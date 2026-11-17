"""How many of the *given* edges does the learner actually recover?

For the runs and threshold that ``analysis.lines`` plots at each (line, x)
point (``x``, ``where``, ``labels``, ``best_over`` as there; the setting is
chosen by best ``shd``), counts per run how many given edges ``u -> v`` are
predicted (``|W[u, v]|`` at or above the threshold) and how many are learned
reversed (``v -> u`` predicted), then averages over seeds. Only points with
edges given (x > 0) are used.

Whether a reverse can be learned at all depends on ``mask_reverse``: with it,
the reverse of each given edge is forbidden.

Usage (one plot per call; e.g. one negative-L1 reward at a time):
    python -m analysis.diagnostics.given_edge_recovery data/results/sachs/given_edges
        --where normalization=zscore model=linear noise=none given_mode=hinge mask_reverse=true
        --best-over w_threshold given_hinge_margin given_hinge_weight
Writes <exp_dir>/plots/given_edge_recovery.png (or --out).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import analyze, lines, style
from train import store


def table(exp_dir: Path, x: str = "frac_given", where: dict | None = None,
          labels: list[str] = ("base", "loss_fn"),
          best_over: list[str] = ("w_threshold",)) -> pd.DataFrame:
    """One row per (label, x > 0): mean ``n_given``, ``n_recovered`` and
    ``n_reverse`` over seeds, ``n`` (runs with a saved graph) and the chosen
    ``w_threshold``."""
    labels, best_over = list(labels), list(best_over)
    runs = lines.select(analyze.load(exp_dir).results, where)
    runs = runs[runs[x] > 0]
    points = lines.reduce(runs, x, y="shd", labels=labels, best_over=best_over)
    runs = lines.chosen_runs(runs, points, x, labels, best_over)
    graphs = store.load_graphs(exp_dir, sorted(runs["run_id"]))
    rows = []
    for rid, label, xval, t in runs[["run_id", "label", x, "w_threshold"]].itertuples(index=False):
        g = graphs.get(rid)
        if g is None:  # failed run: no graph
            continue
        idx = {n: i for i, n in enumerate(g["names"])}
        W = np.asarray(g["W"])
        given = list(g["given"])
        rows.append({"label": label, "x": xval, "w_threshold": t, "n_given": len(given),
                     "n_recovered": sum(W[idx[u], idx[v]] >= t for u, v in given),
                     "n_reverse": sum(W[idx[v], idx[u]] >= t for u, v in given)})
    per_run = pd.DataFrame(rows)
    out = (per_run.groupby(["label", "x", "w_threshold"], as_index=False)
           .agg(n_given=("n_given", "mean"), n_recovered=("n_recovered", "mean"),
                n_reverse=("n_reverse", "mean"), n=("n_given", "size")))
    out.attrs.update(x=x)
    return out


def draw(tbl: pd.DataFrame, ax: plt.Axes | None = None, title: str | None = None) -> plt.Axes:
    """Given edges recovered (solid) and learned reversed (dotted) per line,
    against the number given (dashed black: the ceiling)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 4.8))
    for label, look in style.line_style(sorted(tbl["label"].unique())).items():
        line = tbl[tbl["label"] == label].sort_values("x")
        ax.plot(line["x"], line["n_recovered"], marker=look["marker"], color=look["color"],
                label=look["name"])
        ax.plot(line["x"], line["n_reverse"], marker="x", ls=":", color=look["color"], alpha=0.7)
    ceiling = tbl.groupby("x")["n_given"].mean()
    ax.plot(ceiling.index, ceiling.to_numpy(), "k--", marker="s", ms=4, label="given (ceiling)")
    x = tbl.attrs.get("x", "x")
    ax.set_xlabel(style.AXIS_NAMES.get(x, x))
    ax.set_ylabel("given edges (solid: recovered, dotted: reversed)")
    ax.set_ylim(bottom=0)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("exp_dir", type=Path)
    parser.add_argument("--x", default="frac_given")
    parser.add_argument("--where", nargs="*", default=[], metavar="COL=VALUE",
                        help="runs to use; a comma list means any, 'none' = unset")
    parser.add_argument("--labels", nargs="+", default=["base", "loss_fn"])
    parser.add_argument("--best-over", nargs="+", default=["w_threshold"])
    parser.add_argument("--out", type=Path, default=None,
                        help="default: <exp_dir>/plots/given_edge_recovery.png")
    args = parser.parse_args()
    tbl = table(args.exp_dir, args.x, lines.parse_where(args.where), args.labels, args.best_over)
    out = args.out or args.exp_dir / "plots" / "given_edge_recovery.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    ax = draw(tbl, title=f"{args.exp_dir.parent.name}/{args.exp_dir.name}: given-edge recovery")
    ax.figure.tight_layout()
    ax.figure.savefig(out, dpi=150)
    print(tbl.round(2).to_string(index=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
