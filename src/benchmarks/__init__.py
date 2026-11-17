"""The three benchmarks, each loaded as a `Dataset`.

- ``causalman``: simplified CausalMan "dataset0" (`benchmarks.causalman`);
  needs the Zenodo download in ``data/dataset0/`` (see README).
- ``causalassembly``: the causalAssembly production line (shipped with the
  ``causalAssembly`` package).
- ``sachs``: observational Sachs protein signaling (downloaded on first use).
- ``causalman_sim`` / ``causalman_sim_pooled`` (optional): one sub-batch of the
  CausalMan simulator, or all of them pooled (`benchmarks.causalman_sim`); need
  its sparse clone (see README).

Build the pickles the experiments read (from ``src/``)::

    python -m benchmarks                   # the first three
    python -m benchmarks sachs causalman   # a subset

Experiments then get data through :func:`load`, which also produces noisy
variants on demand.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

from data_generator import DataGenerator, noisy_like

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = REPO_ROOT / "data" / "datasets"


def pickle_path(name: str) -> Path:
    """Where ``python -m benchmarks`` writes the pickled `Dataset` for ``name``."""
    return DATASETS_DIR / f"{name}.pkl"


def _noise_tag(fraction: float, propagate: bool) -> str:
    """Filename token for a noise level: whole percents as 3 digits (0.05 ->
    005), finer fractions in full (0.003 -> 0p003); 'm' marks measurement noise."""
    pct = fraction * 100
    tag = (f"{int(round(pct)):03d}" if abs(pct - round(pct)) < 1e-9
           else f"{fraction:g}".replace(".", "p"))
    return ("" if propagate else "m") + tag


def load(name: str, noise: float | None = None, noise_seed: int = 0,
         propagate_noise: bool = True):
    """The benchmark ``name`` as built by ``python -m benchmarks``, or with
    ``noise`` a copy resampled from its fitted SCM with N(0, noise * std(node))
    added to every non-root node (``data_generator.noisy_like``). Noisy copies
    are generated once and cached under ``data/datasets/_noisy/``."""
    base = pickle_path(name)
    if not base.exists():
        raise SystemExit(f"{base} not found; build it with `python -m benchmarks {name}`.")
    path = base if noise is None else (
        DATASETS_DIR / "_noisy" / f"{name}_noise{_noise_tag(noise, propagate_noise)}_s{noise_seed}.pkl")
    if not path.exists():
        with open(base, "rb") as f:
            clean = pickle.load(f)
        ds = noisy_like(DataGenerator.fit(clean, verbose=False), clean, noise,
                        seed=noise_seed, propagate_noise=propagate_noise)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")  # concurrent jobs may race; rename is atomic
        with open(tmp, "wb") as f:
            pickle.dump(ds, f)
        tmp.replace(path)
        return ds
    with open(path, "rb") as f:
        return pickle.load(f)
