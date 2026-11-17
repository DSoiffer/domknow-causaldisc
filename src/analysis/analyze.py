"""Load an experiment's runs as the tables the plotting tools read.

``load(exp_dir)`` returns:

  - ``results``: every run, one row per w_threshold (a failed run is a single
    row whose ``status`` holds the error);
  - ``summary``: per group of runs that differ only in (seed, dk_seed), the
    mean/std/min/max of every metric at each w_threshold, and ``n_ok``, the
    number of successful runs in the group;
  - ``identity_cols``: the experiment's identity keys (the result columns).

Reading thousands of ``runs/<id>/`` directories is slow, so ``results`` is
cached in ``<exp_dir>/all_results.csv``. The cache is used while it lists
exactly the runs on disk with the same outcome (ok / failed) and rebuilt
otherwise; it is written to a temporary file and renamed, so a concurrent
reader never sees a partial table. Only readers write it; training never does.
A run re-trained in place with the same outcome isn't detected: force a
rebuild with

    python -m analysis.analyze data/results/<benchmark>/<name> [...]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import NamedTuple

import pandas as pd


from train import store

RESULTS_CSV = "all_results.csv"
_NON_METRIC = {"run_id", "w_threshold", "train_seconds", "status"}


class Tables(NamedTuple):
    results: pd.DataFrame
    summary: pd.DataFrame
    identity_cols: list[str]


def build_summary(df: pd.DataFrame, identity_cols: list[str]) -> pd.DataFrame:
    """Mean/std/min/max of every metric per group (identity minus the seeds,
    plus w_threshold), over the successful runs in ``df``."""
    if "status" in df.columns:
        df = df[df["status"].isna()]
    group_cols = [c for c in identity_cols if c not in ("seed", "dk_seed")]
    group_cols += ["w_threshold"]
    metric_cols = [
        c
        for c in df.columns
        if c not in identity_cols and c not in _NON_METRIC
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    grouped = df.groupby(group_cols, dropna=False)
    summary = grouped[metric_cols].agg(["mean", "std", "min", "max"])
    summary.columns = [f"{m}_{stat}" for m, stat in summary.columns]
    summary.insert(0, "n_ok", grouped["fdr"].count())
    return summary.reset_index()


def _outcomes(df: pd.DataFrame) -> set[tuple[str, bool]]:
    """(run_id, succeeded) for every run in a results table."""
    ok = df["status"].isna() if "status" in df.columns else pd.Series(True, index=df.index)
    return set(zip(df["run_id"], ok))


def load(exp_dir: Path, rebuild: bool = False) -> Tables:
    """The experiment's tables (see the module docstring), refreshing the
    cached results first if runs were added, removed or re-run since."""
    recs = store.records(exp_dir)
    if not recs:
        raise SystemExit(f"No finished runs in {exp_dir / store.RUNS_DIR}")
    identity_cols = list(dict.fromkeys(k for r in recs for k in r["identity"]))
    path = exp_dir / RESULTS_CSV
    on_disk = {(r["run_id"], r["status"] == "ok") for r in recs}
    results = None if rebuild or not path.exists() else pd.read_csv(path, low_memory=False)
    if results is None or _outcomes(results) != on_disk:
        tmp = path.with_name(f"{RESULTS_CSV}.{os.getpid()}.tmp")
        store.load_results(exp_dir).to_csv(tmp, index=False)
        tmp.replace(path)
        print(f"{exp_dir}: {len(recs)} runs -> {RESULTS_CSV}")
        # Read back, so a fresh table and a cached one are parsed identically.
        results = pd.read_csv(path, low_memory=False)
    return Tables(results, build_summary(results, identity_cols), identity_cols)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("exp_dirs", nargs="+", type=Path, help="Experiment directories.")
    args = parser.parse_args()
    for exp_dir in args.exp_dirs:
        tables = load(exp_dir, rebuild=True)
        failed = tables.results["status"].notna().sum() if "status" in tables.results else 0
        print(f"  {failed} failed run(s); {len(tables.summary)} summary groups")


if __name__ == "__main__":
    main()
