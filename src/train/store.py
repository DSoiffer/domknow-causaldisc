"""On-disk layout of an experiment's runs -- the one place every tool reads it.

Each run lives in ``<exp_dir>/runs/<run_id>/``:

- ``run.json``: ``{"run_id", "identity", "settings", "status", "error",
  "train_seconds"}``, written last, so its presence marks a finished attempt;
- ``metrics.csv``: one row per w_threshold (successful runs only);
- ``graph.npz``: ``W`` (|W|, unthresholded), ``variables``,
  ``B_true``, ``given`` (successful runs only);
- ``log.txt``: the training output.

``identity`` holds the scalar parameters that tell runs of an experiment apart
(learner config, axis values, noise, seeds); they become the result columns.
``settings`` holds everything else a run depends on (benchmark, dataset
fingerprint, threshold grid, method options). The run ID hashes both, and the
runner compares the stored record in full before treating a run as done, so
runs with any differing input can never share results.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

RUNS_DIR = "runs"


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True)


def run_id(identity: dict, settings: dict) -> str:
    """16 hex chars of the SHA-256 of the full (identity, settings) record."""
    payload = canonical({"identity": identity, "settings": settings})
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def same_inputs(record: dict, identity: dict, settings: dict) -> bool:
    """Whether a stored record was produced from exactly these inputs."""
    return (canonical(record["identity"]) == canonical(identity)
            and canonical(record["settings"]) == canonical(settings))


def run_dir(exp_dir: Path, rid: str) -> Path:
    return exp_dir / RUNS_DIR / rid


def read_record(exp_dir: Path, rid: str) -> dict | None:
    path = run_dir(exp_dir, rid) / "run.json"
    return json.loads(path.read_text()) if path.exists() else None


def save_run(exp_dir: Path, record: dict, metrics: pd.DataFrame | None = None,
             graph: dict | None = None) -> None:
    """Write a run's outputs, ``run.json`` last (atomically)."""
    out = run_dir(exp_dir, record["run_id"])
    out.mkdir(parents=True, exist_ok=True)
    if metrics is not None:
        metrics.to_csv(out / "metrics.csv", index=False)
    if graph is not None:
        np.savez_compressed(out / "graph.npz", **graph)
    tmp = out / "run.json.tmp"
    tmp.write_text(json.dumps(record, indent=2))
    tmp.replace(out / "run.json")


def records(exp_dir: Path) -> list[dict]:
    """Every finished run's record, ordered by run ID."""
    return [json.loads(p.read_text())
            for p in sorted((exp_dir / RUNS_DIR).glob("*/run.json"))]


def shared_identities(recs: list[dict]) -> dict[str, list[dict]]:
    """Records that share all identity parameters (so they differ only in
    settings, e.g. an older dataset or threshold grid), keyed by identity."""
    groups: dict[str, list[dict]] = {}
    for r in recs:
        groups.setdefault(canonical(r["identity"]), []).append(r)
    return {k: v for k, v in groups.items() if len(v) > 1}


def load_results(exp_dir: Path) -> pd.DataFrame:
    """All runs as one table: one row per run x w_threshold, and a single row
    for a failed run, whose ``status`` holds the error.

    Warns about records that share every identity parameter: the analysis
    groups runs by identity, so they are averaged as if they were extra seeds."""
    recs = records(exp_dir)
    if not recs:
        raise SystemExit(f"No finished runs in {exp_dir / RUNS_DIR}")
    shared = shared_identities(recs)
    if shared:
        differing = sorted({k for group in shared.values()
                            for k in set().union(*(r["settings"] for r in group))
                            if len({canonical(r["settings"].get(k)) for r in group}) > 1})
        example = next(iter(shared.values()))
        print(f"WARNING: {exp_dir}: {len(shared)} parameter set(s) have more than one run "
              f"record; they are averaged together as if they were extra seeds (settings "
              f"that differ: {differing or 'none'}; e.g. runs "
              f"{', '.join(r['run_id'] for r in example)}).")
    frames = []
    for r in recs:
        head = {"run_id": r["run_id"], **r["identity"]}
        if r["status"] == "ok":
            m = pd.read_csv(run_dir(exp_dir, r["run_id"]) / "metrics.csv")
            frames.append(pd.DataFrame([head] * len(m)).join(m)
                          .assign(train_seconds=r["train_seconds"]))
        else:
            frames.append(pd.DataFrame([{**head, "status": r["error"]}]))
    return pd.concat(frames, ignore_index=True)


def load_graph(exp_dir: Path, rid: str) -> dict | None:
    """The run's saved graph arrays, or None if it has none (failed run)."""
    path = run_dir(exp_dir, rid) / "graph.npz"
    if not path.exists():
        return None
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def edge_set(W: np.ndarray, names: list[str], threshold: float) -> set:
    """Directed edges (by variable name) with weight >= ``threshold`` (> 0 for 0)."""
    i, j = np.nonzero(np.asarray(W) >= threshold) if threshold > 0 else np.nonzero(W)
    return {(names[a], names[b]) for a, b in zip(i, j)}


def load_graphs(exp_dir: Path, run_ids) -> dict[str, dict]:
    """{run_id: {W, names, true, given}} for every run with a saved graph, with
    the true and given edges as name sets."""
    out = {}
    for rid in run_ids:
        z = load_graph(exp_dir, rid)
        if z is None:  # failed run
            continue
        names = [str(n) for n in z["variables"]]
        out[rid] = {"W": z["W"], "names": names,
                    "true": edge_set(z["B_true"], names, 0),
                    "given": edge_set(z["given"], names, 0)}
    return out
