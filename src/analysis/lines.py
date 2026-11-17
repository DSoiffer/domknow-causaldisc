"""Metric-vs-x plots, one axes per call, with every choice explicit.

Three stages, each usable on its own (e.g. from a notebook):

- ``select(results, where)``: the runs to plot;
- ``reduce(runs, x, y, ...)``: one row per (line, x): the mean / min / max of
  ``y`` over seeds and the number of seeds ``n``, after choosing the setting of
  the ``best_over`` columns (the threshold, and any hyperparameter grid) by a
  selection ``rule``;
- ``draw(table)``: the lines on one axes (``analysis.heatmap`` draws one line's
  metric over two axes instead).

``table(exp_dir, ...)`` loads an experiment and runs the first two stages (plus
the no-knowledge fallback); ``plot(...)`` also draws it::

    from analysis import lines
    tm = lines.table("data/results/sachs/tier_mask", x="mask_fraction",
                     where={"normalization": "zscore", "model": "linear"})
    ge = lines.table("data/results/sachs/given_edges", x="frac_given",
                     where={"normalization": "zscore", "model": "linear",
                            "given_mode": ["hinge", None],
                            "mask_reverse": [True, None]},
                     best_over=["w_threshold", "given_hinge_margin", "given_hinge_weight"],
                     baseline=tm)
    lines.draw(ge)

Seeds whose estimate is cyclic at a threshold have no SHD there: by default
(``cyclic="drop"``) they are left out of that point's mean (``n`` counts the
seeds that remain); ``cyclic="fill"`` scores them at their own smallest acyclic
threshold instead.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import analyze, style

SEED_COLS = ("seed", "dk_seed")
RULES = ("best", "pareto", "global")
CYCLIC = ("drop", "fill")
_KEYS = ["label", "x"]
_STEMS = ("fdr", "tpr", "fpr", "shd", "nnz")


def _stem(metric: str) -> str:
    return metric.removeprefix("full_").removeprefix("skel_").removeprefix("cpdag_")


def direction(metric: str) -> int:
    """+1 if higher is better (tpr), -1 if lower is (fdr, fpr, shd), else 0."""
    stem = _stem(metric)
    return 1 if stem == "tpr" else -1 if stem in ("fdr", "fpr", "shd") else 0


def select(results: pd.DataFrame, where: dict | None = None) -> pd.DataFrame:
    """The successful runs matching ``where`` ({column: value}). A value may be
    a list (any of them); ``None`` matches an unset (NaN) entry."""
    df = results[results["status"].isna()] if "status" in results.columns else results
    for col, want in (where or {}).items():
        wants = want if isinstance(want, list) else [want]
        values = [w for w in wants if w is not None]
        keep = df[col].isin(values)
        if None in wants:
            keep |= df[col].isna()
        df = df[keep]
    if df.empty:
        raise ValueError(f"no successful runs match {where}")
    return df


def parse_where(items: list[str]) -> dict:
    """``where`` from command-line items like ``noise=0.1`` or
    ``given_mode=hinge,none`` (a comma list means any of them): ``none`` ->
    None, ``true``/``false`` -> bool, numbers -> int or float, else strings."""
    def value(v: str):
        if v.lower() in ("none", "true", "false"):
            return {"none": None, "true": True, "false": False}[v.lower()]
        for cast in (int, float):
            try:
                return cast(v)
            except ValueError:
                pass
        return v
    out = {}
    for item in items:
        key, _, raw = item.partition("=")
        vals = [value(v) for v in raw.split(",")]
        out[key] = vals if len(vals) > 1 else vals[0]
    return out


def chosen_runs(runs: pd.DataFrame, points: pd.DataFrame, x: str, labels: list[str],
                best_over: list[str]) -> pd.DataFrame:
    """One row per run behind ``points`` (a ``reduce`` table): the runs at each
    point's chosen hyperparameters, with that point's ``w_threshold`` and
    ``label``. For tools that look past the metrics, e.g. at the saved graphs."""
    hyper = [c for c in best_over if c != "w_threshold"]
    one = runs.drop_duplicates("run_id").drop(columns="w_threshold")
    one = one.assign(label=one[list(labels)].map(str).agg("/".join, axis=1))
    chosen = points[["label", "x", *hyper, "w_threshold"]].rename(columns={"x": x})
    return one.merge(chosen, on=["label", x, *hyper])


def reduce(runs: pd.DataFrame, x: str, y: str = "full_shd",
           labels: list[str] = ("base", "loss_fn"),
           best_over: list[str] = ("w_threshold",),
           rank_by: list[str] = ("shd", "skel_shd"), rule: str = "best",
           cyclic: str = "drop") -> pd.DataFrame:
    """One row per (line label, x value) with columns ``label``, ``x``, the
    label columns, ``y`` (mean over seeds), ``lo``/``hi`` (min/max over seeds),
    ``n`` (seeds with a value) and the chosen ``best_over`` values.

    ``cyclic``: a seed whose estimate is cyclic at the chosen threshold has no
    ``y`` there. "drop" leaves it out of the point; "fill" scores it at its own
    smallest acyclic threshold instead (its ``dag_<y>``), so every seed counts.
    Either way the setting is chosen on the ``rank_by`` means as below.

    Every setting of the ``best_over`` columns is scored by the seed-means of
    ``rank_by`` (compared in order, each in its own direction: higher tpr,
    lower fdr/fpr/shd; missing = worst; ties go to the smallest ``best_over``
    values in order). ``rank_by=[y]`` chooses by the plotted metric itself. The
    ``rule``:

    - "best": per point, the best-ranked setting;
    - "pareto": hyperparameters (``best_over`` besides ``w_threshold``) as for
      "best"; then per point, a threshold at least as good as every other on
      every directional metric (fdr, tpr, fpr, shd and their variants), if one
      exists (column ``dominant`` True), else the threshold best on ``y`` (ties:
      the other directional metrics in column order; an ``nnz`` metric uses its
      ``shd``);
    - "global": hyperparameters as for "best"; then one threshold per line
      across all of its points: the one with a ``rank_by[0]`` value at the most
      points, then the best ``rank_by`` means over the points.

    Every other column that varies (besides seeds and metrics) must be fixed by
    the caller (``where``), or a point would mix different configurations."""
    labels, best_over, rank_by = list(labels), list(best_over), list(rank_by)
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}, got {rule!r}")
    if rule != "best" and "w_threshold" not in best_over:
        raise ValueError(f"rule {rule!r} chooses w_threshold; put it in best_over")
    if cyclic not in CYCLIC:
        raise ValueError(f"cyclic must be one of {CYCLIC}, got {cyclic!r}")
    if cyclic == "fill":
        if f"dag_{y}" not in runs.columns:
            raise ValueError(f"cyclic='fill' needs a dag_{y} column (each run's metrics at "
                             "its smallest acyclic threshold)")
        runs = runs.assign(_y=runs[y].fillna(runs[f"dag_{y}"]))
    else:
        runs = runs.assign(_y=runs[y])
    point = [*labels, x]
    keys = [*point, *best_over]
    metric_cols = [c for c in runs.columns if c.startswith("dag_") or _stem(c) in _STEMS]
    grouped = runs.groupby(keys, dropna=False)
    others = [c for c in runs.columns if c not in
              {*keys, *SEED_COLS, *metric_cols, "_y", "run_id", "status", "train_seconds"}]
    free = [c for c, k in grouped[others].nunique(dropna=False).max().items() if k > 1]
    if free:
        raise ValueError(f"columns {free} vary within a point; fix them with `where`, "
                         "or add them to `labels` or `best_over`")

    directional = ([c for c in metric_cols if direction(c) and not c.startswith("dag_")]
                   if rule == "pareto" else [])
    means = grouped[list(dict.fromkeys([*rank_by, *directional]))].mean()
    agg = means.reset_index().merge(
        grouped["_y"].agg(y="mean", lo="min", hi="max", n="count").reset_index(), on=keys)
    undirected = [m for m in rank_by if not direction(m)]
    if undirected:
        raise ValueError(f"cannot rank by {undirected}: neither higher nor lower is better")
    ranks = [f"_rank{i}" for i in range(len(rank_by))]
    agg[ranks] = (agg[rank_by] * [-direction(m) for m in rank_by]).fillna(np.inf).to_numpy()
    best = agg.sort_values([*ranks, *best_over], kind="stable").drop_duplicates(point)
    if rule == "best":
        chosen = best
    else:
        hyper = [c for c in best_over if c != "w_threshold"]
        cand = agg.merge(best[[*point, *hyper]], on=[*point, *hyper]) if hyper else agg
        chosen = (_pareto(cand, point, y, directional) if rule == "pareto"
                  else _global(cand, labels, ranks))
    chosen = chosen.assign(label=chosen[labels].map(str).agg("/".join, axis=1), x=chosen[x])
    extra = ["dominant"] if rule == "pareto" else []
    out = chosen[[*_KEYS, *[c for c in labels if c != "x"], "y", "lo", "hi", "n",
                  *best_over, *extra]]
    return out.sort_values(_KEYS).reset_index(drop=True)


def _pareto(cand: pd.DataFrame, point: list[str], y: str, metrics: list[str]) -> pd.DataFrame:
    """Per point: the smallest Pareto-dominant threshold over ``metrics``, else
    the threshold best on ``y`` (see ``reduce``)."""
    scores = (cand[metrics] * [direction(m) for m in metrics]).fillna(-np.inf)
    y_key = y if direction(y) else y.replace("nnz", "shd")
    order = [y_key, *[m for m in metrics if m != y_key]]
    rows = []
    for _, grp in cand.groupby(point, dropna=False):
        s = scores.loc[grp.index]
        dominant = grp[(s >= s.max() - 1e-12).all(axis=1)]
        if len(dominant):
            rows.append(dominant.sort_values("w_threshold").iloc[[0]].assign(dominant=True))
        else:
            ranked = grp.assign(**{f"_o{i}": -s[m] for i, m in enumerate(order)})
            pick = ranked.sort_values([*(f"_o{i}" for i in range(len(order))), "w_threshold"],
                                      kind="stable").iloc[[0]]
            rows.append(pick.drop(columns=[f"_o{i}" for i in range(len(order))])
                        .assign(dominant=False))
    return pd.concat(rows)


def _global(cand: pd.DataFrame, labels: list[str], ranks: list[str]) -> pd.DataFrame:
    """Per line: one threshold for all of its points (see ``reduce``)."""
    rows = []
    for _, grp in cand.groupby(labels, dropna=False):
        by_t = grp.replace(np.inf, np.nan).groupby("w_threshold")
        stats = by_t[ranks].mean().fillna(np.inf)
        stats["_coverage"] = by_t[ranks[0]].count()
        stats = stats.reset_index().sort_values(
            ["_coverage", *ranks, "w_threshold"],
            ascending=[False, *[True] * len(ranks), True], kind="stable")
        rows.append(grp[grp["w_threshold"] == stats["w_threshold"].iloc[0]])
    return pd.concat(rows)


def with_baseline(tbl: pd.DataFrame, baseline: pd.DataFrame | None) -> pd.DataFrame:
    """``tbl`` with each line's x = 0 point taken from ``baseline`` (another
    experiment's table) where ``tbl`` has none: x = 0 means no domain
    knowledge, so the two experiments' points are equivalent."""
    if baseline is None:
        return tbl
    have = set(tbl.loc[tbl["x"] == 0, "label"])
    extra = baseline[(baseline["x"] == 0) & baseline["label"].isin(set(tbl["label"]) - have)]
    return pd.concat([tbl, extra], ignore_index=True).sort_values(_KEYS).reset_index(drop=True)


def table(exp_dir: str | Path, x: str, y: str = "full_shd", where: dict | None = None,
          labels: list[str] = ("base", "loss_fn"), best_over: list[str] = ("w_threshold",),
          rank_by: list[str] = ("shd", "skel_shd"), rule: str = "best",
          cyclic: str = "drop", baseline: pd.DataFrame | None = None) -> pd.DataFrame:
    """Load an experiment and reduce it to plotted points (see ``reduce`` and
    ``with_baseline``)."""
    runs = select(analyze.load(Path(exp_dir)).results, where)
    tbl = with_baseline(reduce(runs, x, y, labels, best_over, rank_by, rule, cyclic), baseline)
    tbl.attrs.update(x=x, y=y)  # axis labels for draw and heatmap
    return tbl


def draw(tbl: pd.DataFrame, ax: plt.Axes | None = None, band: bool = True,
         title: str | None = None, xlabel: str | None = None, ylabel: str | None = None,
         reference: float | None = None, legend: bool = True) -> plt.Axes:
    """One line per label (x ascending), with a min-max band over seeds and,
    optionally, a dashed horizontal ``reference`` line (e.g. the true edge
    count). Points with no value (every seed cyclic) leave a gap; with the
    "pareto" rule, points without a dominant threshold are drawn hollow."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 4.8))
    looks = style.line_style(sorted(tbl["label"].unique()))
    for label, look in looks.items():
        line = tbl[tbl["label"] == label].sort_values("x")
        ax.plot(line["x"], line["y"], marker=look["marker"], color=look["color"],
                label=look["name"], lw=1.6, ms=6)
        if "dominant" in line.columns:  # hollow: no threshold best on every metric
            hollow = line[~line["dominant"].astype(bool)]
            ax.plot(hollow["x"], hollow["y"], ls="none", marker=look["marker"], ms=6,
                    mfc="white", mec=look["color"])
        if band:
            ax.fill_between(line["x"], line["lo"], line["hi"], color=look["color"], alpha=0.12)
    if reference is not None:
        ax.axhline(reference, ls="--", lw=1.2, color="gray", alpha=0.5)
    ax.set_xlabel(xlabel or style.AXIS_NAMES.get(tbl.attrs.get("x", ""), tbl.attrs.get("x", "")))
    ax.set_ylabel(ylabel or style.metric_name(tbl.attrs.get("y", "")))
    if title:
        ax.set_title(title)
    if legend:
        ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def plot(exp_dir: str | Path, x: str, y: str = "full_shd", where: dict | None = None,
         labels: list[str] = ("base", "loss_fn"), best_over: list[str] = ("w_threshold",),
         rank_by: list[str] = ("shd", "skel_shd"), rule: str = "best",
         cyclic: str = "drop", baseline: pd.DataFrame | None = None,
         out: str | Path | None = None,
         **draw_kw) -> plt.Axes:
    """``table`` then ``draw``; saves the figure to ``out`` if given."""
    ax = draw(table(exp_dir, x, y, where, labels, best_over, rank_by, rule, cyclic, baseline),
              **draw_kw)
    if out is not None:
        ax.figure.tight_layout()
        ax.figure.savefig(out)
    return ax
