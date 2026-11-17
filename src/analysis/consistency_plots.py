"""Consistency figures and tables for an experiment, from the saved graphs.

Every view uses the runs and threshold that ``analysis.lines`` plots at each
(line, x) point (``x``, ``where``, ``labels``, ``best_over`` as there; the
setting is chosen by best ``shd``). Outputs in ``<exp_dir>/plots/`` (or
``--out``):

  - ``consistency.png``: across-seed consistency of each base config vs the
    axis -- mean pairwise Jaccard between the seeds' predicted edge sets, and
    the fraction of unknown true edges always / never recovered.
  - ``config_agreement.png``: consistency *between* base configs -- mean
    Jaccard between different configs' predictions on the same (axis value,
    seed), as a curve vs the axis and as a config x config heatmap.
  - ``edge_recovery.png``: per real (true, not-given) edge, how consistently
    it is recovered across all runs of the experiment; the most and least
    consistently recovered edges are printed.
  - ``edge_false_positives.png``: the top-20 non-true edges by how often they
    are wrongly predicted (rate over runs where the edge isn't a true edge).

Usage:
    python -m analysis.consistency_plots data/results/sachs/given_edges --x frac_given
        --where normalization=zscore model=linear noise=none given_mode=hinge,none mask_reverse=true,none
        --best-over w_threshold given_hinge_margin given_hinge_weight
"""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


from analysis import analyze, lines, style
from train import store


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _group_consistency(runs: list[dict], threshold: float) -> tuple[dict, list[dict]]:
    """Stability + per-edge stats for one group of runs (differing only in seed)
    at one threshold.

    A run contributes its predicted edge set (|W| >= threshold; W is
    stored as absolute values) and its true/given edge sets. Unknown true
    edges are scored per run, so groups whose true graphs differ across runs
    (e.g. graph_growth) are handled.
    """
    for r in runs:
        r["pred"] = store.edge_set(r["W"], r["names"], threshold)

    jaccards = [_jaccard(a["pred"], b["pred"]) for a, b in combinations(runs, 2)]

    all_edges = set().union(*(r["pred"] | r["true"] for r in runs))
    edge_rows = []
    for edge in sorted(all_edges):
        n_unknown = sum(edge in r["true"] and edge not in r["given"] for r in runs)
        n_recovered = sum(
            edge in r["pred"] and edge in r["true"] and edge not in r["given"]
            for r in runs
        )
        edge_rows.append(
            {
                "src": edge[0],
                "dst": edge[1],
                "n_runs": len(runs),
                "n_true": sum(edge in r["true"] for r in runs),
                "n_given": sum(edge in r["given"] for r in runs),
                "n_pred": sum(edge in r["pred"] for r in runs),
                "n_false_pred": sum(
                    edge in r["pred"] and edge not in r["true"] for r in runs
                ),
                # recovery rate among the runs where the edge was there to discover
                "n_true_unknown": n_unknown,
                "recovery_rate": n_recovered / n_unknown if n_unknown else np.nan,
            }
        )

    rates = [e["recovery_rate"] for e in edge_rows if e["n_true_unknown"] > 0]
    group_row = {
        "n_runs": len(runs),
        "pred_jaccard_mean": float(np.mean(jaccards)) if jaccards else np.nan,
        "pred_nnz_mean": float(np.mean([len(r["pred"]) for r in runs])),
        "n_unknown_edges": len(rates),
        "n_always_recovered": sum(r == 1.0 for r in rates),
        "n_sometimes_recovered": sum(0.0 < r < 1.0 for r in rates),
        "n_never_recovered": sum(r == 0.0 for r in rates),
    }
    return group_row, edge_rows


def consistency_tables(
    runs: pd.DataFrame, graphs: dict[str, dict], sel: dict, x_col: str,
    group_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per group of runs differing only in seed, at the group's selected
    threshold: (one row of stability stats per group, one row per group x edge)."""
    group_rows, edge_rows = [], []
    for key, grp in runs.groupby(group_cols, dropna=False):
        ident = dict(zip(group_cols, key), label=grp["label"].iloc[0])
        t = sel.get((ident["label"], ident[x_col]))
        members = [dict(graphs[r]) for r in grp["run_id"] if r in graphs]
        if t is None or not members:
            continue
        row, edges = _group_consistency(members, t)
        group_rows.append({**ident, **row})
        edge_rows += [{**ident, **e} for e in edges]
    return pd.DataFrame(group_rows), pd.DataFrame(edge_rows)


def plot_seed_consistency(
    cons: pd.DataFrame, x_col: str, palette: dict, out_png: Path
) -> None:
    """Across-seed consistency per base config vs the axis."""
    cons = cons.assign(
        frac_always=cons["n_always_recovered"] / cons["n_unknown_edges"],
        frac_never=cons["n_never_recovered"] / cons["n_unknown_edges"],
    )
    panels = [
        ("pred_jaccard_mean", "graph stability across seeds\n(mean pairwise Jaccard)"),
        ("frac_always", "unknown true edges\nalways recovered (frac)"),
        ("frac_never", "unknown true edges\nnever recovered (frac)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (col, title) in zip(axes, panels):
        for label, grp in cons.groupby("label"):
            grp = grp.sort_values(x_col)
            ax.plot(grp[x_col], grp[col], marker="o", markersize=4,
                    label=label, color=palette[label])
        ax.set_xlabel(x_col)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle("Across-seed consistency (at best-shd threshold per point)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_config_agreement(
    runs: pd.DataFrame,
    graphs: dict[str, dict],
    sel: dict,
    x_col: str,
    labels: list[str],
    out_png: Path,
) -> None:
    """Agreement between different base configs' predictions on the same
    (axis value, seed): curve vs the axis + config x config heatmap."""
    seed_cols = [c for c in ("seed", "dk_seed") if c in runs.columns]
    pair_jacc: dict[tuple, list[float]] = {}  # (x, label_a, label_b) -> [jaccard]
    for (x, *_), grp in runs.groupby([x_col, *seed_cols]):
        preds = []
        for rid, lbl in zip(grp["run_id"], grp["label"]):
            g = graphs.get(rid)
            if g is not None:
                preds.append((lbl, store.edge_set(g["W"], g["names"], sel[(lbl, x)])))
        for (la, ea), (lb, eb) in combinations(sorted(preds), 2):
            pair_jacc.setdefault((x, la, lb), []).append(_jaccard(ea, eb))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), width_ratios=[1.1, 1])

    # Curve: mean agreement across all config pairs, min-max band over pairs.
    xs = sorted({k[0] for k in pair_jacc})
    per_x = [[np.mean(v) for k, v in pair_jacc.items() if k[0] == x] for x in xs]
    ax1.plot(xs, [np.mean(v) for v in per_x], marker="o", color="C0")
    ax1.fill_between(xs, [min(v) for v in per_x], [max(v) for v in per_x],
                     alpha=0.15, color="C0")
    ax1.set_xlabel(x_col)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_title("mean Jaccard between configs\n(band = min-max over config pairs)")
    ax1.grid(alpha=0.3)

    # Heatmap: mean agreement per config pair (over axis values and seeds).
    # Cells are annotated with their comparison count: a config whose runs
    # mostly failed contributes only the few (x, seed) cells it survived, so
    # its means are not comparable to fully-covered cells.
    n = len(labels)
    M, counts = np.full((n, n), np.nan), np.zeros((n, n), dtype=int)
    np.fill_diagonal(M, 1.0)
    pair_all: dict[tuple, list[float]] = {}
    for (_, la, lb), v in pair_jacc.items():
        pair_all.setdefault((la, lb), []).extend(v)
    for (la, lb), v in pair_all.items():
        i, j = labels.index(la), labels.index(lb)
        M[i, j] = M[j, i] = np.mean(v)
        counts[i, j] = counts[j, i] = len(v)
    annot = np.where(np.isfinite(M), np.char.add(
        np.vectorize(lambda v: f"{v:.2f}\n")(np.nan_to_num(M)),
        np.vectorize(lambda c: f"n={c}")(counts)), "")
    np.fill_diagonal(annot, "1.00")
    sns.heatmap(pd.DataFrame(M, index=labels, columns=labels), vmin=0, vmax=1,
                annot=annot, fmt="", cmap="viridis",
                annot_kws={"fontsize": 6}, ax=ax2)
    ax2.tick_params(labelsize=7)
    ax2.set_title(f"mean Jaccard per config pair\n"
                  f"(over {x_col} values and seeds; n = #comparisons)")
    fig.suptitle("Between-config consistency (at best-shd threshold per point)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _stacked_edge_plot(
    numdf: pd.DataFrame, den: pd.Series, out_png: Path, *,
    palette: dict, title: str, xlabel: str, top: int | None = None,
) -> pd.DataFrame:
    """Horizontal bars of per-edge rate, stacked by base config.

    ``numdf`` has columns ``src, dst, label, num`` (a config's numerator for an
    edge); ``den`` is a Series indexed by ``(src, dst)`` of the total denominator
    (pooled over configs/runs). Each config's segment width is ``num / total
    den``, so the segments sum to the edge's overall rate and the fraction of the
    bar in a config's color is that config's share of the numerator. Sorted by
    rate; keeps the top ``top``. Returns the per-edge ``rate``/``n`` table.
    """
    den = den[den > 0]
    rate_by_cfg = (
        numdf.merge(den.rename("den"), left_on=["src", "dst"], right_index=True)
        .assign(frac=lambda d: d["num"] / d["den"])
        .pivot_table(index=["src", "dst"], columns="label", values="frac", fill_value=0.0)
    )
    totals = rate_by_cfg.sum(axis=1).sort_values(ascending=False)
    if top is not None:
        totals = totals.head(top)
    rate_by_cfg = rate_by_cfg.loc[totals.index]
    cols = [c for c in palette if c in rate_by_cfg.columns]  # stable order + legend

    edges = list(totals.index)
    names = [f"{s} -> {d}" for s, d in edges]
    y = np.arange(len(edges))
    fig, ax = plt.subplots(figsize=(9, max(3, 0.32 * len(edges))))
    left = np.zeros(len(edges))
    for cfg in cols:
        w = rate_by_cfg[cfg].to_numpy()
        ax.barh(y, w, left=left, color=palette[cfg], label=cfg, height=0.8)
        left += w
    for yi, e in zip(y, edges):
        ax.text(totals[e] + 0.01, yi, f"n={int(den[e])}", va="center", fontsize=6)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()  # highest rate on top
    ax.set_xlim(0, 1.1)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend(fontsize=6, ncol=min(3, len(cols)), loc="lower right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return pd.DataFrame({"rate": totals.round(4), "n": den.loc[edges].astype(int)})


def edge_recovery_table(edges: pd.DataFrame, out_png: Path, palette: dict) -> None:
    """Per real (true, not-given) edge: how consistently it is recovered, with
    each bar stacked by which base config did the recovering."""
    edges = edges.assign(n_recovered=edges["recovery_rate"] * edges["n_true_unknown"])
    numdf = (edges.groupby(["src", "dst", "label"], as_index=False)["n_recovered"].sum()
             .rename(columns={"n_recovered": "num"}))
    den = edges.groupby(["src", "dst"])["n_true_unknown"].sum()
    summ = _stacked_edge_plot(
        numdf, den, out_png, palette=palette,
        title="Per-edge recovery consistency (stacked by base config)",
        xlabel="recovery rate (over runs where the edge was there to discover)",
    )
    print("Most consistently recovered edges:")
    print(summ.head(5).to_string())
    print("Least consistently recovered edges:")
    print(summ.tail(5).to_string())


def edge_false_positive_table(
    graphs: dict[str, dict], runs: pd.DataFrame, sel: dict, x_col: str,
    out_png: Path, palette: dict, top: int = 20,
) -> None:
    """Top non-true directed edges by false-positive rate, from the saved graphs,
    each bar stacked by which base config did the wrong predicting.

    For every run, *every* ordered pair of present nodes that is not a true edge
    there is an opportunity; it is a false positive if predicted (at the run's
    best-shd threshold). So the denominator includes runs where the edge was
    correctly *not* predicted -- an unbiased rate, unlike a per-edge-table aggregation.
    """
    opp, false_pos = Counter(), Counter()  # opp: per edge; false_pos: per (edge, config)
    for rid, lbl, xval in runs[["run_id", "label", x_col]].itertuples(index=False):
        g = graphs.get(rid)
        if g is None or (lbl, xval) not in sel:
            continue
        names, true = g["names"], g["true"]
        pred = store.edge_set(g["W"], names, sel[(lbl, xval)])
        present_pairs = {(a, b) for a in names for b in names if a != b}
        opp.update(present_pairs - true)
        false_pos.update((a, b, lbl) for (a, b) in (pred - true))

    den = pd.Series(dict(opp))
    den.index = pd.MultiIndex.from_tuples(den.index, names=["src", "dst"])
    numdf = pd.DataFrame(
        [{"src": a, "dst": b, "label": l, "num": c} for (a, b, l), c in false_pos.items()]
    )
    summ = _stacked_edge_plot(
        numdf, den, out_png, palette=palette, top=top,
        title=f"Top {top} most-often falsely predicted (non-)edges (stacked by base config)",
        xlabel="false-positive rate (over runs where both endpoints exist and the edge is not true)",
    )
    print(f"Most often falsely predicted edges (top {min(top, len(summ))}):")
    print(summ.head(5).to_string())


def report(exp_dir: Path, x: str, where: dict | None = None,
           labels: list[str] = ("base", "loss_fn"), best_over: list[str] = ("w_threshold",),
           out_dir: Path | None = None) -> None:
    """Write the four consistency figures for the points ``analysis.lines``
    would plot (see its ``table`` for the arguments)."""
    labels, best_over = list(labels), list(best_over)
    print(f"=== {exp_dir} ({where}) ===")
    runs = lines.select(analyze.load(exp_dir).results, where)
    points = lines.reduce(runs, x, y="shd", labels=labels, best_over=best_over)
    runs = lines.chosen_runs(runs, points, x, labels, best_over)
    sel = {(lbl, xv): t for lbl, xv, t in points[["label", "x", "w_threshold"]].itertuples(index=False)}
    names = sorted(points["label"].unique())
    palette = {lbl: look["color"] for lbl, look in style.line_style(names).items()}
    out_dir = out_dir or exp_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    graphs = store.load_graphs(exp_dir, sorted(runs["run_id"]))
    group_cols = [*labels, x, *[c for c in best_over if c != "w_threshold"]]
    cons, edges = consistency_tables(runs, graphs, sel, x, group_cols)
    plot_seed_consistency(cons, x, palette, out_dir / "consistency.png")
    plot_config_agreement(runs, graphs, sel, x, names, out_dir / "config_agreement.png")
    edge_recovery_table(edges, out_dir / "edge_recovery.png", palette)
    edge_false_positive_table(graphs, runs, sel, x, out_dir / "edge_false_positives.png", palette)
    print(f"Wrote consistency.png, config_agreement.png, edge_recovery.png, "
          f"edge_false_positives.png -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("exp_dir", type=Path)
    parser.add_argument("--x", required=True, help="the experiment axis, e.g. frac_given")
    parser.add_argument("--where", nargs="*", default=[], metavar="COL=VALUE",
                        help="runs to use; a comma list means any, 'none' = unset")
    parser.add_argument("--labels", nargs="+", default=["base", "loss_fn"])
    parser.add_argument("--best-over", nargs="+", default=["w_threshold"])
    parser.add_argument("--out", type=Path, default=None, help="default: <exp_dir>/plots")
    args = parser.parse_args()
    report(args.exp_dir, args.x, lines.parse_where(args.where), args.labels, args.best_over,
           args.out)


if __name__ == "__main__":
    main()
