"""How lines look: display name, color and marker per line label, and display
names for axis columns. Labels are "/"-joined label-column values (by default
"<base>/<loss_fn>"); unknown labels get a color from the default cycle and no
display name change."""

from __future__ import annotations

import matplotlib.pyplot as plt

# label -> (display name, marker), in legend order. Colors follow this order.
LINES: dict[str, tuple[str, str]] = {
    "dagma_adam/l2": ("DAGMA", "o"),
    "dagma_adam/logdetll_nv": ("DAGMA-NV", "s"),
    "dagma_adam_logll/logll": ("DAGMA-LL", "^"),
    "notears_lbfgs/l2": ("NOTEARS", "D"),
    "notears_adam/l2": ("NOTEARS-Adam", "v"),
    "notears_adam/logdetll_nv": ("NOTEARS-NV-Adam", "X"),
    "dagma_adam_nl/l2": ("DAGMA-MLP", "o"),
    "dagma_adam_logll_nl/logll": ("DAGMA-LL-MLP", "^"),
    "notears_lbfgs_nl/l2": ("NOTEARS-MLP", "D"),
    "notears_adam_nl/l2": ("NOTEARS-Adam-MLP", "v"),
    "notears_adam_logll_nl/logll": ("NOTEARS-LL-Adam-MLP", "P"),
}

AXIS_NAMES: dict[str, str] = {
    "mask_fraction": "Fraction of known-bad edges masked",
    "frac_given": "Fraction of true edges given",
    "frac_colliders": "Fraction of colliders given",
    "n_colliders": "Number of colliders kept",
    "noise": "Noise level",
}


_METRIC_WORDS = {"shd": "SHD", "tpr": "TPR", "fdr": "FDR", "fpr": "FPR", "nnz": "edges"}


def metric_name(metric: str) -> str:
    """Axis label for a metric column: 'full_skel_tpr' -> 'skel TPR'."""
    return " ".join(_METRIC_WORDS.get(w, w) for w in metric.removeprefix("full_").split("_"))


def line_style(labels: list[str]) -> dict[str, dict]:
    """{label: {"name", "color", "marker"}} for every label, consistent across
    plots: known labels keep their slot in ``LINES``; others follow after."""
    cycle = plt.get_cmap("tab20").colors
    known = [lbl for lbl in LINES if lbl in labels]
    ordered = known + sorted(set(labels) - set(known))
    out = {}
    for lbl in ordered:
        slot = list(LINES).index(lbl) if lbl in LINES else len(LINES) + ordered.index(lbl)
        name, marker = LINES.get(lbl, (lbl, "o"))
        # tab20's dark shades first (even indices), then its light ones.
        color = cycle[(2 * slot) % 20 + (slot // 10) % 2]
        out[lbl] = {"name": name, "color": color, "marker": marker}
    return out
