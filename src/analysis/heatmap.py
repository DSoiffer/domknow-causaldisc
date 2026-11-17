"""One line's metric over two axes (e.g. the experiment axis by noise), as a
heatmap of the points ``analysis.lines`` selects::

    from analysis import heatmap, lines
    t = lines.table("data/results/causalman/tier_mask", x="mask_fraction",
                    where={"normalization": "zscore", "model": "linear"},
                    labels=["base", "loss_fn", "noise"])
    heatmap.draw(t[(t["base"] == "dagma_adam") & (t["loss_fn"] == "l2")], rows="noise")
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import lines, style


def draw(tbl: pd.DataFrame, rows: str, ax: plt.Axes | None = None,
         title: str | None = None, fmt: str = ".3g") -> plt.Axes:
    """One line's metric over two axes: x (columns) by ``rows`` (e.g. "noise";
    unset values show as "none"). ``tbl`` must hold one value per (rows, x):
    build it with ``rows`` among the ``labels`` and keep one line, e.g.
    ``t = lines.table(..., labels=["base", "loss_fn", "noise"]); t[t.base == "dagma_adam"]``.
    Colors run from worse (dark) to better (light)."""
    values = tbl[rows]
    grid = (tbl.assign(_row=values.astype(object).where(values.notna(), "none"))
            .pivot(index="_row", columns="x", values="y")
            .reindex((["none"] if values.isna().any() else []) + sorted(values.dropna().unique())))
    if ax is None:
        _, ax = plt.subplots(figsize=(1.0 + 0.8 * grid.shape[1], 1.2 + 0.5 * grid.shape[0]))
    y = tbl.attrs.get("y", "")
    higher_better = lines.direction(y) >= 0
    cells = grid.to_numpy(dtype=float)
    im = ax.imshow(cells, aspect="auto", cmap="viridis" if higher_better else "viridis_r")
    for (i, j), v in np.ndenumerate(cells):
        if v == v:  # dark text on the light (yellow) end of the colormap
            light = im.norm(v) > 0.6 if higher_better else im.norm(v) < 0.4
            ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=8,
                    color="black" if light else "white")
    ax.set_xticks(range(grid.shape[1]), [f"{c:g}" for c in grid.columns])
    ax.set_yticks(range(grid.shape[0]), [str(r) for r in grid.index])
    ax.set_xlabel(style.AXIS_NAMES.get(tbl.attrs.get("x", ""), tbl.attrs.get("x", "")))
    ax.set_ylabel(style.AXIS_NAMES.get(rows, rows))
    ax.figure.colorbar(im, ax=ax, label=style.metric_name(y))
    if title:
        ax.set_title(title)
    return ax
