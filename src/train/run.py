"""Run an experiment: expand its grid, skip finished runs, train the rest.

    python -m train.run tier_mask --benchmark sachs            # everything pending
    python -m train.run given_edges --benchmark sachs --jobs 4  # 4 at a time
    python -m train.run given_edges --benchmark sachs --shard 2/5  # 2nd fifth of the grid
    python -m train.run tier_mask --benchmark causalman --noise none 0.1
    python -m train.run tier_mask --benchmark sachs --configs nonlinear --seeds 0 1

The grid is the learners (``--configs``: config sets or learner names from
``config.toml``) x the experiment's axes x the noise levels x the seed pairs.
``--seeds`` sets both the training seed and the domain-knowledge seed
(``dk_seed``: which edges are masked or given) unless ``--dk-seeds`` crosses them
separately. Every run is stored under an ID hashed from all of its inputs (see
``train.store``), so re-running a command only trains what is missing. Failed
runs are skipped unless ``--retry-failed``. ``--status`` prints counts only.

``--shard K/N`` trains only every N-th run of the grid, starting at the K-th
(K = 1..N), so N copies of a command with shards 1/N .. N/N split the grid
between them without overlap (e.g. one per cluster job). A shard re-run after
an interruption skips the runs it already finished.

From Python (e.g. a notebook)::

    from train import run
    grid = run.build_grid("tier_mask", "sachs", seeds=[0])
    run.train(grid, jobs=4)

Results go to ``data/results/<benchmark>/<experiment>/runs/``; the tools in
``analysis`` read them from there.
"""

from __future__ import annotations

import argparse
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import benchmarks
from train import config as config_mod
from train import store
from train.experiments import PREPARE, resolve_axes
from train.fit import describe, fit, load_data

RESULTS_DIR = benchmarks.REPO_ROOT / "data" / "results"


@dataclass(frozen=True)
class Run:
    exp_dir: Path
    identity: dict
    settings: dict

    @property
    def id(self) -> str:
        return store.run_id(self.identity, self.settings)


def _learners(cfg: config_mod.Config, names: list[str]) -> list[dict]:
    """Config-set names and learner names, expanded to learner identities."""
    out = []
    for name in names:
        if name in cfg.config_sets:
            out += [cfg.learners[n] for n in cfg.config_sets[name]]
        elif name in cfg.learners:
            out.append(cfg.learners[name])
        else:
            raise ValueError(f"{name!r} is neither a config set nor a learner in the config")
    return out


def build_grid(experiment: str, benchmark: str, configs: list[str] = ("linear",),
               noise: list[float | None] | None = None, seeds: list[int] | None = None,
               dk_seeds: list[int] | None = None, results_dir: Path = RESULTS_DIR,
               config: config_mod.Config | None = None) -> list[Run]:
    """Every run of ``experiment`` on ``benchmark``, in a stable order.

    ``configs``: config sets and/or learner names; ``noise`` (default: the
    benchmark's levels) and ``seeds`` (default: the config's) as on the command
    line; ``dk_seeds`` crosses domain-knowledge seeds with ``seeds`` instead of
    tying them."""
    cfg = config or config_mod.load()
    if experiment not in PREPARE or experiment not in cfg.experiments:
        raise ValueError(f"unknown experiment {experiment!r}; known: {sorted(cfg.experiments)}")
    if benchmark not in cfg.benchmarks:
        raise ValueError(f"unknown benchmark {benchmark!r}; known: {sorted(cfg.benchmarks)}")
    bench = cfg.benchmarks[benchmark]
    noise = bench.noise if noise is None else noise
    seeds = cfg.seeds if seeds is None else seeds
    seed_pairs = [(s, s) for s in seeds] if dk_seeds is None else list(product(seeds, dk_seeds))
    axes = {**cfg.experiments[experiment], **bench.axes.get(experiment, {})}
    axes = resolve_axes(axes, load_data({"benchmark": benchmark, "noise": None,
                                         "noise_seed": cfg.noise_seed,
                                         "propagate_noise": cfg.propagate_noise}))
    fingerprints = {n: load_data({"benchmark": benchmark, "noise": n, "noise_seed": cfg.noise_seed,
                                  "propagate_noise": cfg.propagate_noise}).fingerprint()
                    for n in noise}
    exp_dir = results_dir / benchmark / experiment
    grid = []
    for learner, combo, level, (seed, dk_seed) in product(
            _learners(cfg, list(configs)), product(*axes.values()), noise, seed_pairs):
        identity = {"experiment": experiment, **learner, **dict(zip(axes, combo)),
                    "noise": level, "n_samples": cfg.n_samples, "dk_seed": dk_seed, "seed": seed}
        settings = {
            "benchmark": benchmark,
            "data": fingerprints[level],
            "noise_seed": cfg.noise_seed,
            "propagate_noise": cfg.propagate_noise,
            "layout_dataset": bench.layout_dataset,
            "cpdag_metrics": bench.cpdag_metrics,
            "w_thresholds": cfg.w_thresholds,
            "method_options": cfg.method_options.get(learner["method"], {}),
            "optimizer_options": cfg.optimizer_options.get(learner["method"], {}),
        }
        grid.append(Run(exp_dir, identity, settings))
    # Repeated values (e.g. seeds 0 0) must not run the same thing twice.
    return list({r.id: r for r in grid}.values())


def state(run: Run) -> str:
    """'pending', 'ok' or 'failed'. Refuses to continue if a different run's
    record sits under this ID (a hash collision: never silently reuse it)."""
    record = store.read_record(run.exp_dir, run.id)
    if record is None:
        return "pending"
    if not store.same_inputs(record, run.identity, run.settings):
        raise SystemExit(f"Run {run.id} in {run.exp_dir} holds a different run's results.")
    return record["status"]


def execute(run: Run) -> tuple[str, str, float | None]:
    """Train ``run`` and store its outputs, logging to ``log.txt``. A failure is
    recorded (status 'failed' + the error) rather than raised."""
    out = store.run_dir(run.exp_dir, run.id)
    out.mkdir(parents=True, exist_ok=True)
    record = {"run_id": run.id, "identity": run.identity, "settings": run.settings}
    with open(out / "log.txt", "w") as log, redirect_stdout(log), redirect_stderr(log):
        try:
            metrics, graph, seconds = fit(run.identity, run.settings)
        except Exception as exc:
            traceback.print_exc()
            error = f"{type(exc).__name__}: {exc}"
            store.save_run(run.exp_dir, record | {"status": "failed", "error": error,
                                                  "train_seconds": None})
            return run.id, error, None
    store.save_run(run.exp_dir, record | {"status": "ok", "error": None,
                                          "train_seconds": seconds}, metrics, graph)
    return run.id, "ok", seconds


def _report(outcome: tuple[str, str, float | None]) -> None:
    rid, status, seconds = outcome
    print(f"  {rid} " + (f"done in {seconds:.0f}s" if status == "ok" else f"FAILED: {status}"),
          flush=True)


def train(runs: list[Run], jobs: int = 1, retry_failed: bool = False,
          status_only: bool = False) -> None:
    """Train every pending run of ``runs`` (and failed ones with
    ``retry_failed``), ``jobs`` at a time, printing counts first and then each
    run as it starts and finishes."""
    states = [state(r) for r in runs]
    n_failed = states.count("failed")
    print(f"{len(runs)} runs: {states.count('ok')} done, {n_failed} failed, "
          f"{states.count('pending')} pending", flush=True)
    if n_failed and not retry_failed:
        print(f"  skipping {n_failed} failed run(s); see runs/<id>/log.txt, "
              "or pass --retry-failed to re-run them")
    if status_only:
        return
    runnable = {"pending", "failed"} if retry_failed else {"pending"}
    todo = [r for r, s in zip(runs, states) if s in runnable]
    start = time.time()
    # Every run trains in a fresh process, so its result can't depend on what
    # trained before it: dagrad keeps state across calls (DAGMA updates its
    # default `s` schedule in place when it retries).
    # Runs are handed to the pool only as workers free up, so "started" is
    # printed when a run actually starts.
    with ProcessPoolExecutor(jobs, max_tasks_per_child=1) as pool:
        try:
            queue, running = list(enumerate(todo, 1)), {}
            while queue or running:
                while queue and len(running) < jobs:
                    i, r = queue.pop(0)
                    running[pool.submit(execute, r)] = r
                    print(f"  [{i}/{len(todo)}] started {r.id}: {describe(r.identity)}",
                          flush=True)
                # Short timed waits: on Windows an untimed one can't be interrupted.
                done, _ = wait(running, timeout=0.5, return_when=FIRST_COMPLETED)
                for f in done:
                    running.pop(f)
                    _report(f.result())
        except KeyboardInterrupt:
            # Ctrl-C / a notebook interrupt reaches only this process; without
            # this the pool would wait for the runs in flight to finish. A killed
            # run has no run.json, so it counts as pending next time.
            pool.terminate_workers()
            print("Interrupted; the unfinished runs stay pending.", flush=True)
            raise
    print(f"Trained {len(todo)} run(s) in {time.time() - start:.0f}s")


# ---- CLI ---------------------------------------------------------------------

def _noise_level(value: str) -> float | None:
    return None if value.lower() == "none" else float(value)


def _shard(value: str) -> tuple[int, int]:
    k, n = (int(x) for x in value.split("/"))
    if not 1 <= k <= n:
        raise argparse.ArgumentTypeError(f"need 1 <= K <= N, got {value}")
    return k, n


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\nFrom Python")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--configs", nargs="+", default=["linear"],
                        help="Config sets and/or learner names from the config (default: linear).")
    parser.add_argument("--noise", nargs="+", type=_noise_level, default=None,
                        help="Noise levels: fractions, or 'none' for the data as built "
                             "(default: the benchmark's levels in the config).")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Training seeds, also the dk_seeds unless --dk-seeds "
                             "(default: the config's).")
    parser.add_argument("--dk-seeds", type=int, nargs="+", default=None,
                        help="Domain-knowledge seeds, crossed with --seeds.")
    parser.add_argument("--jobs", type=int, default=1, help="Runs trained at a time (default 1).")
    parser.add_argument("--shard", type=_shard, default=(1, 1), metavar="K/N",
                        help="Only the K-th of N disjoint parts of the grid.")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Also re-run runs that failed before.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                        help="Default: <repo>/data/results.")
    parser.add_argument("--config", type=Path, default=config_mod.CONFIG_PATH,
                        help="Protocol file (default: the shipped train/config.toml).")
    parser.add_argument("--status", action="store_true", help="Print counts and exit.")
    args = parser.parse_args()

    grid = build_grid(args.experiment, args.benchmark, args.configs, args.noise, args.seeds,
                      args.dk_seeds, args.results_dir, config_mod.load(args.config))
    k, n = args.shard
    print(f"{args.experiment} on {args.benchmark}"
          + (f", shard {k}/{n} of {len(grid)} runs" if n > 1 else "") + " -> "
          + str(args.results_dir / args.benchmark / args.experiment), flush=True)
    train(grid[k - 1::n], args.jobs, args.retry_failed, args.status)


if __name__ == "__main__":
    main()
