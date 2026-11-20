# Analysis: tables and figures from trained runs

Everything here reads an experiment directory written by
[train.run](../train/README.md) (`data/results/<benchmark>/<experiment>/`) and
never trains.

They all load an experiment through `analysis.analyze.load`, which returns the
per-run results (one row per run x w_threshold) and the summary over seeds. The
results table is cached in `<exp_dir>/all_results.csv` and rebuilt when runs are
added, removed or re-run. `python -m analysis.analyze <dir>` forces a rebuild, e.g.
after re-training a run in place.

## Plotting a metric vs domain knowledge given (`analysis.lines`)

From Python (a notebook, say):

```python
from analysis import lines
Z = {"normalization": "zscore", "noise": None, "model": "linear"}
tm = lines.table("data/results/sachs/tier_mask", x="mask_fraction", where=Z)
ge = lines.table("data/results/sachs/given_edges", x="frac_given", y="full_shd",
                 where={**Z, "given_mode": ["hinge", None], "mask_reverse": [True, None]},
                 best_over=["w_threshold", "given_hinge_margin", "given_hinge_weight"],
                 baseline=tm)
lines.draw(ge, reference=17)  # dashed line at the true edge count
```
Args:
- `where` picks the runs ({column: value}; a list means any of them, `None`
  means unset). If any other column still takes more than one value within a
  point (besides the seeds), an error is raised.
- `y` is the plotted metric (default `full_shd`; see "Metrics" below);
  `labels` the columns that make a line (default `base`, `loss_fn`; add
  `tier_source` for causalAssembly's two masks); [style.py](style.py) gives
  each line its display name, color and marker.
- `best_over`: per point, the setting of these columns (the threshold, and a
  hyperparameter grid if any) with the best seed-mean of `rank_by` is kept. Ties go to the smallest values. The table records the setting chosen.
  `rank_by` defaults to `shd`, then `skel_shd`, whatever `y` is, so a TPR plot
  shows the TPR at the best-SHD threshold. `rank_by=[y]` chooses by the plotted
  metric instead (each metric in its own direction: higher TPR, lower
  FDR/FPR/SHD; not `nnz`). Note that the best threshold for TPR alone is
  simply the lowest one that leaves the estimate acyclic.
- `rule` changes how the threshold is chosen (hyperparameters stay per point):
  `"best"` (default, as above); `"pareto"`, per point a threshold at least as
  good as every other on every metric if one exists, else the one best on `y`
  (drawn hollow; column `dominant`); `"global"`, one threshold per line across
  all of its points.
- `baseline`: another experiment's table whose x = 0 points (no knowledge) can
  be used to fill in for x = 0 points for this one, all else being equal.

One seed only: `where={..., "seed": 0}`.

### Metrics

Every run is scored at each threshold on five metrics: `tpr`, `fdr`, `fpr`,
`shd` (lower is better except `tpr`) and `nnz` (estimated edge count). Each
comes in variants, by prefix:

- no prefix: directed edges, **excluding the given edges** (when an experiment
  gives some, they are removed from both truth and estimate; otherwise this
  equals `full_`)
- `full_`: directed edges over the whole graph
- `skel_` / `full_skel_`: the same on the skeleton (edge direction ignored)
- `cpdag_`: against the true graph's Markov equivalence class (only for
  benchmarks with `cpdag_metrics` in the config: Sachs)
- `dag_` (any of the above): at the run's smallest threshold that leaves its
  estimate acyclic (what `cyclic="fill"` uses)

By default, full_shd is plotted.

Each point is the mean over seeds. Optionally, min-max bands can also be disabled. A seed whose estimate is cyclic at the chosen threshold has no SHD there:
`cyclic="drop"` (default) leaves it out, `cyclic="fill"` scores it at its own
smallest acyclic threshold (the run's `dag_*` metrics). The threshold is chosen
the same way under both.

[heatmap.py](heatmap.py) draws one line's metric over two axes instead, e.g.
the domain knowledge strength by noise: build the table with the second axis among the
`labels`, keep one line, and call `heatmap.draw(tbl, rows="noise")`.

Note that choosing the threshold (and hyperparameters) per point by the results is
oracle selection, it is not realistic to expect a tuning procedure to perform as well in practice.

## Other plotters

There are two other plotting capabilities provided. Both use the same runs and threshold a `lines` plot would (same `--x`, `--where`,
`--labels`, `--best-over`; `--where` takes `COL=VALUE`, a comma list meaning any
of them and `none` unset) and write to `<dir>/plots/` unless `--out` is given:

| command | what |
|---|---|
| `analysis.consistency_plots <dir> --x X ...` | across-seed and between-config agreement of the predicted edge sets, per-edge recovery, top false positives |
| `analysis.diagnostics.given_edge_recovery <dir> ...` | how many of the given edges are recovered (`--reverse`: also how many are learned reversed, for runs without `mask_reverse`). Works on `given_edges` and `given_edges_neg_l1`; for the latter, fix one reward per call (`--where lambda1_given=0.1 ...`) |

## Visualizing a run's learned graph

`python -m visualize.graph run <exp_dir> <run_id> [--threshold T]` draws a run's
estimate against the ground truth (true positives grey, false positives red,
missed edges dashed orange), at its best-SHD threshold by default. See
[visualize/graph.py](../visualize/graph.py).
