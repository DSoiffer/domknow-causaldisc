# Training: the domain-knowledge experiments

- [config.toml](config.toml): learner settings and config
  sets, per-benchmark settings, each experiment's axes, thresholds and seeds.
  Read by [config.py](config.py); pass another file with `--config` (or
  `config=config.load(path)` from Python).
- [experiments.py](experiments.py): what each experiment does to one run.
- [fit.py](fit.py): trains one run, [run.py](run.py) builds grids and trains
  the pending runs

Each experiment crosses a set of learners (`--configs`: config sets or learner
names) with its own axes, the benchmark's noise levels and a set of seeds. Every
grid point is one run that trains once, evaluates over the shared `w_threshold`
grid, and saves both metrics and the predicted graph.

## Experiments

| name | axes | what it tests |
|---|---|---|
| `tier_mask` | `mask_fraction`, `tier_source` | forbidding edges from later to earlier tiers, keeping a fraction of the forbidden set (paper, Figure 1) |
| `given_edges` | `frac_given`, `given_hinge_margin`, `given_hinge_weight` | a fraction of true edges given, encouraged by a hinge on \|W\| (paper, Figure 2) |
| `given_edges_neg_l1` | `frac_given`, `lambda1_given` | the same, encouraged by a negative L1 weight |
| `given_colliders` | `frac_colliders` | both incoming edges of a fraction of the colliders in the ground-truth graph are given (negative L1) |
| `graph_growth` | `n_colliders` (1, 2, ..., all) | gradually grows the graph  collider-by-collider to study where learning breaks down; other variables are marginalized via latent projection |


Benchmarks: `causalman`, `causalassembly`, `sachs`, and optionally
`causalman_sim` / `causalman_sim_pooled` (built from the CausalMan simulator;
see the top-level README). Each benchmark's table in the config sets its
default noise levels, whether CPDAG metrics are scored, and extra axes for an
experiment on it: the `tier_source` values for `tier_mask` (CausalMan's
reference layout, causalAssembly's stations and TierSort, TierSort for the
others). All of an experiment's runs on a benchmark share one directory.

A `prepare(ds, cfg, rng)` in `experiments.py` returns the (possibly
transformed) `Dataset`, the given edges, an optional allow-mask and the dagrad
options that encode the given edges. Everything else (training, metrics,
saving) is generic. Adding a new experiment means creating one prepare function in `PREPARE` +
one `[experiments.<name>]` table of axes in the config.

## Scoring

When edges are given, recovery is scored on `E_unknown = E* \ E_given`: given
positions are removed from both the ground truth and the estimate (given
*pairs* for skeleton metrics). Plain metric columns (`fdr`, `tpr`, `skel_*`,
...) are the E_unknown versions; `full_*` columns score the whole
graph (the paper's given-edge figure uses `full_shd`).

## Seeds

There are two unique seeds per run: `dk_seed` selects the domain knowledge (which forbidden or
given edges), `seed` seeds training. `--seeds` sets both to the same by default, pass
`--dk-seeds` to cross them independently.

The linear NOTEARS/DAGMA algorithms are deterministic (they start from W = 0 and
draw no random numbers), so `seed` changes nothing for them. Their seed-to-seed
variation is entirely `dk_seed`, and where no knowledge is subsampled (mask
fraction 0 or 1) every seed gives the same run. Only the MLP variants use `seed`.

## Noise

The noise level setting primarily exists because the CausalMan benchmark contains mostly deterministic functions, which makes the graph unlearnable. Injecting (not too much) noise typically makes the algorithms learn more effectively. 

Noise levels are just another crossed axis (`--noise`, default from the
benchmark). `none` trains on the dataset as built; a fraction `F` resamples the
dataset's fitted SCM (`DataGenerator`) with `N(0, F*std(node))` added to every
non-root node, same row count. Noisy pickles are generated once and cached under
`data/datasets/_noisy/` (keyed by level + `noise_seed`). All levels land in the
same experiment dir, recorded in the `noise` column.

The `causalman_sim` benchmarks are built with noise already inside the
simulator's structural equations (1% per node by default), so their only noise
level is `none`; a different amount means rebuilding them
(`benchmarks.causalman_sim.build(noise_frac=...)`). Resampling from a fitted
SCM would not work for them anyway: their mechanisms (ratios, ReLUs) are
outside what `DataGenerator` fits.

## Usage

```bash
python -m train.run given_edges --benchmark sachs [--jobs 4] [--shard 1/3]
```

or from Python (e.g. a notebook):

```python
from train import run
grid = run.build_grid("tier_mask", "sachs", seeds=[0])
run.train(grid, jobs=4)
```

then analyze the results with the tools in [../analysis/](../analysis/README.md).

`train.run` trains every run of the grid that isn't finished, so
re-running a command resumes it, and `--status` prints counts. It prints each
run as it starts and finishes. `--jobs N` trains
N runs at a time; `--shard K/N` takes only every N-th run of the grid (starting
at the K-th), so N copies with shards `1/N` .. `N/N`, e.g. one per cluster job,
split the grid without overlap.

Each run is stored under an ID that hashes all of its inputs: the `identity`
(learner config, axis values, noise, seeds: the result columns) and the
`settings` (benchmark, a fingerprint of the exact dataset, the threshold grid,
method options). Before skipping a run as done, the runner compares the stored
inputs in full, so differing runs can never share results. Changing a setting
(say, rebuilding a dataset) therefore adds new runs next to the old ones with the
same parameters. The analysis averages those together as if they were extra seeds
and prints a warning; remove the stale `runs/<id>` directories or use a fresh
`--results-dir` if that isn't wanted.
[store.py](store.py) is the only code that reads or writes this layout:

- `runs/<run_id>/run.json`: identity, settings, status (`ok`/`failed` + error), and
  training time
- `runs/<run_id>/metrics.csv`: one row per w_threshold; every row also carries
  the run's smallest acyclic threshold (`dag_threshold`) and its metrics there
  (`dag_*`), for scoring estimates that are cyclic at a plotted threshold
- `runs/<run_id>/graph.npz`: the learned weights `W` (\|W\|,
  unthresholded, zero diagonal), `variables`, `B_true`, `given`
- `runs/<run_id>/log.txt`: training output
- `all_results.csv`: every run, one row per threshold; a cache the analysis
  tools build and refresh when the runs change (see
  [analysis/analyze.py](../analysis/analyze.py))

