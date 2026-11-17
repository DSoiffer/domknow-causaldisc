# Domain knowledge in differentiable structure learning

Code for *The Effects of Domain Knowledge on Differentiable Structure Learning in
Industrial Scenarios*. It measures how two kinds of
prior knowledge change what NOTEARS, DAGMA and GOLEM-style likelihood variants
recover:

- **known non-existent edges** (`tier_mask`): forbid edges from later to earlier
  tiers (e.g., assembly-line stations), keeping a growing fraction of the forbidden set
  (Figure 1 in the paper);
- **known existent edges** (`given_edges`): hand the learner a fraction of the true
  edges, encouraged by a hinge on their magnitude (Figure 2 in the paper);

on three benchmarks: CausalMan (simplified, with injected noise), causalAssembly,
and Sachs.

## Setup

It is recommended that you [install uv](https://docs.astral.sh/uv/getting-started/installation/). Then run

```bash
uv sync --extra causalassembly
```

The `causalassembly` extra is only needed to build the causalAssembly dataset (see
[License](#license)), plain `uv sync` installs everything else.

Run commands with `uv run python -m ...` (or activate `.venv`). 

The paths below are relative to the repository root. 

## Data

Everything under `data/` is generated or downloaded; nothing there is in git.

1. **CausalMan** (manual download). Get the three parts of `data.zip` from
   <https://zenodo.org/records/14697668> (11 GB; CC BY 4.0), join them, and extract
   the three files the paper uses:

   ```bash
   cat data.zip.001 data.zip.002 data.zip.003 > data.zip
   unzip data.zip dataset0/dataset0_fully_observable.csv dataset0/ground_truth.gml \
       dataset0/ground_truth_dataset0.gml_complete.html -d data/
   ```

   The HTML file is the reference graph layout. Its vertical layers are the tiers
   for the CausalMan tier mask.
2. **causalAssembly** ships with the `causalAssembly` package (the `causalassembly`
   extra), and the **Sachs** dataset is downloaded on first use.
3. Build the three datasets:

   ```bash
   python -m benchmarks        # -> data/datasets/{causalman,causalassembly,sachs}.pkl
   ```

   The CausalMan simplification (one product configuration, the ancestors of the
   press-fit station, no tolerance/pass-fail variables) is documented in
   [src/benchmarks/causalman.py](src/benchmarks/causalman.py). Noisy CausalMan data
   is resampled from a fitted SCM the first time an experiment needs it.
4. **Optional: `causalman_sim`**, a benchmark built from the more newly released CausalMan
   simulator (<https://github.com/boschresearch/CausalMan>, license is AGPL-3.0) rather than
   the Zenodo data. We take one sub-batch of its `causalman_micro` SCM with 1% process
   noise added inside the structural equations (5000 rows, 33 variables, 38 edges);
   `causalman_sim_pooled` pools all 12 environments instead (14 841 rows, 42
   variables, 62 edges). How both are built, and where the pooled ground truth is
   not fully accurate, is documented in
   [src/benchmarks/causalman_sim.py](src/benchmarks/causalman_sim.py).

   The simulator is not a project dependency. A full clone is ~8.5 GB (data for
   all four model sizes), so the commands below make a sparse clone into
   `CausalMan/` that keeps only the micro model (~190 MB), then build both
   benchmarks with the simulator added to the environment for that one command.
   `core.longpaths` is needed on Windows, where some of its file paths pass the
   260-character limit.

   ```bash
   git clone -c core.longpaths=true --filter=blob:none --sparse https://github.com/boschresearch/CausalMan.git CausalMan
   cd CausalMan
   printf '%s\n' '/*' '!/pre_generated_datasets/*' '/pre_generated_datasets/causalman_micro/' \
     '!/causalman/dataset_objects/*' '/causalman/dataset_objects/__init__.py' \
     '/causalman/dataset_objects/causalman_micro/' | git sparse-checkout set --no-cone --stdin
   cd ..
   uv run --with-editable ./CausalMan python -m benchmarks causalman_sim causalman_sim_pooled
   ```

## Examples

[notebooks/examples.ipynb](notebooks/examples.ipynb) trains one seed of DAGMA
on CausalMan and causalAssembly (forbidden edges), Sachs (given edges) and
CausalMan (given colliders), plots each result, and draws a benchmark's true
graph and one run's estimate against it: a short tour of `train.run`,
`analysis.lines` and `visualize.graph`.

## Training causal discovery

Training code is in [src/train/](src/train/README.md).

`train.run` trains every run of an experiment's grid that isn't finished yet.
The grid crosses the learner configs (`--configs`, default: the `linear` set),
the experiment's axes, the benchmark's noise levels and the seeds (10 by
default), all set in [src/train/config.toml](src/train/config.toml). `--jobs N`
trains N runs at a time:

```bash
for b in causalman causalassembly sachs; do
  python -m train.run tier_mask   --benchmark $b    # Figure 1
  python -m train.run given_edges --benchmark $b    # Figure 2
done
```


To analyze / plot trained runs, use code in `analysis`.  

Each panel of the paper's figures is then one `analysis.lines` plot of an
experiment directory, e.g. Sachs's given-edges panel:

```python
from analysis import lines
Z = {"normalization": "zscore", "noise": None, "model": "linear"}
tm = lines.table("data/results/sachs/tier_mask", x="mask_fraction", where=Z)
ge = lines.table("data/results/sachs/given_edges", x="frac_given",
                 where={**Z, "given_mode": ["hinge", None], "mask_reverse": [True, None]},
                 best_over=["w_threshold", "given_hinge_margin", "given_hinge_weight"],
                 baseline=tm)
lines.draw(ge, reference=17)
```

(CausalMan: `"noise": 0.1`; causalAssembly's tier_mask: add `"tier_source":
"station"` or `"dag"` for TierSort.) Figure 1 scores a seed whose estimate is
cyclic at the chosen threshold at its own smallest acyclic threshold: pass
`cyclic="fill"` to `lines.table` for its panels. By default (`"drop"`) such
seeds are left out, as in Figure 2.

Each run is stored under an ID hashed from all of its inputs, so re-running a
command picks up where it stopped, and changing any setting produces new runs
instead of overwriting old ones. Failed runs are skipped (with a note) unless you
pass `--retry-failed`. To split a grid across machines or cluster jobs, give N
copies of the command `--shard 1/N` .. `--shard N/N`; each trains a disjoint part.

This is a big job. Each `tier_mask` launch is 540 runs (causalAssembly has two mask
kinds, station and TierSort, so it launches 1080), and each `given_edges` launch is
2160 (4 fractions × a 3×3 hinge grid). Linear runs take from under a minute (Sachs)
to ~15 GPU-minutes (causalAssembly) each, a few hundred GPU-hours in total. To try the
pipeline first, use fewer seeds (`--seeds 0`) or see the example notebook.

Results go to `data/results/<benchmark>/<experiment>/runs/`. See
[src/train/README.md](src/train/README.md) for the experiments, scoring and
on-disk outputs, and [src/analysis/README.md](src/analysis/README.md) for the
tables and plots.

## Layout

```
notebooks/              usage examples (examples.ipynb)
dagrad/                 modified fork of github.com/Duntrain/dagrad
                        (adds per-edge L1 weights and the given-edge hinge; see
                        "Changes in this fork" in dagrad/README.md)
src/
  dataset.py            Dataset: data + ground-truth graph
  data_generator.py     fit an SCM to a Dataset and resample it with noise
  benchmarks/           build the datasets (python -m benchmarks), layout.py
                        reads the CausalMan reference layout (its tiers)
  train/                training code, experiment definitions, configs
  analysis/             figures from trained runs: lines (metric vs x, one plot per
                        call), heatmap, consistency_plots
  visualize/            interactive graph renderings: a dataset's ground truth, or
                        one run's estimate against it (python -m visualize.graph)
```



## License

The code is under three licenses. There are two licenses that conflict, so programs involving those are separated.

- **`dagrad/`** has its own non-commercial license ([dagrad/LICENSE](dagrad/LICENSE)).
  The experiment code depends on it, so in practice the experiments are for
  non-commercial use only.
- **[src/benchmarks/causalassembly.py](src/benchmarks/causalassembly.py)** and
  **[src/benchmarks/causalman_sim.py](src/benchmarks/causalman_sim.py)**: AGPL-3.0
  ([LICENSE-AGPL-3.0](LICENSE-AGPL-3.0)), because they use the AGPL-3.0
  `causalAssembly` package and CausalMan simulator. Each runs only as a separate
  data-preparation step (`python -m benchmarks causalassembly` /
  `causalman_sim`), which writes a pickled dataset; no other code imports them,
  and the experiments read the pickles without either package installed.
- **Everything else**: MIT ([LICENSE](LICENSE)).

The CausalMan data are CC BY 4.0 (cite the CausalMan paper).
