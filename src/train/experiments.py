"""What each experiment does to one run.

An experiment's ``prepare(ds, cfg, rng)`` turns one grid point (``cfg``: the
run's identity and settings) into the inputs it trains on, a `Prepared`:

- ``ds``: the dataset, possibly transformed (graph_growth marginalizes variables);
- ``given_edges``: edges handed to the learner as domain knowledge (recovery is
  scored on E_unknown = E* \\ E_given);
- ``mask``: an optional allow-mask over ``ds.variables`` (1 = allowed);
- ``options``: extra dagrad ``general_options`` encoding the given edges.

All selection randomness comes from ``rng``, seeded by the run's ``dk_seed``.
The grids themselves (the axes) are in ``config.toml``. To add an experiment,
write a prepare function, add it to ``PREPARE`` and give it a table of axes in
the config.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dataset import Dataset
from train.knowledge import (
    build_dag_tier_mask,
    build_given_edges_lambda,
    build_station_dag_tier_mask,
    build_station_tier_mask,
    build_vertical_mask,
    find_colliders,
    given_matrix,
)


@dataclass(frozen=True)
class Prepared:
    ds: Dataset
    given_edges: list[tuple[str, str]] = field(default_factory=list)
    mask: np.ndarray | None = None
    options: dict = field(default_factory=dict)


def _true_edges(ds: Dataset) -> list[tuple[str, str]]:
    return sorted((str(u), str(v)) for u, v in ds.directed_subgraph().edges())


def _choose(items: list, fraction: float, rng: np.random.Generator) -> list:
    k = int(round(fraction * len(items)))
    idx = rng.choice(len(items), size=k, replace=False)
    return [items[i] for i in sorted(idx)]


# ---- Forbidden edges: tier masks -------------------------------------------------

_TIER_MASKS = {"station": build_station_tier_mask, "dag": build_dag_tier_mask,
               "station_dag": build_station_dag_tier_mask}


def tier_mask(ds: Dataset, cfg: dict, rng: np.random.Generator) -> Prepared:
    """Forbid edges from later to earlier tiers (``tier_source``), keeping a
    ``mask_fraction`` of the forbidden set."""
    frac = cfg["mask_fraction"]
    if frac == 0.0:
        return Prepared(ds)
    source = cfg["tier_source"]
    if source == "reference":
        mask = build_vertical_mask(ds, dataset=cfg["layout_dataset"],
                                   subsample_fraction=frac, rng=rng)
    else:
        mask = _TIER_MASKS[source](ds, subsample_fraction=frac, rng=rng)
    return Prepared(ds, mask=mask)


# ---- Given edges -------------------------------------------------------------------

def _reverse_forbidden_mask(ds: Dataset, given_edges: list[tuple[str, str]]) -> np.ndarray:
    """Allow-mask that forbids only the reverse of each given edge (given u -> v
    forbids v -> u); self-loops are forbidden by convention."""
    idx = {n: i for i, n in enumerate(ds.variables)}
    mask = np.ones((len(idx), len(idx)), dtype=float)
    np.fill_diagonal(mask, 0.0)
    for u, v in given_edges:
        mask[idx[v], idx[u]] = 0.0
    return mask


def _encode_given(ds: Dataset, given_edges: list[tuple[str, str]], cfg: dict) -> dict:
    """dagrad options that encourage the given edges (``given_mode``):
    - "hinge": no sparsity on them, plus a loss-side magnitude floor
      rho * sum max(0, m - |W|)^2 (m, rho = given_hinge_margin / given_hinge_weight);
    - "neg_l1": L1 weight -lambda1_given on them (None = the learner's lambda1)."""
    given = given_matrix(ds.variables, given_edges)
    lambda1 = cfg["lambda1"]
    if cfg["given_mode"] == "hinge":
        options = {"lambda1": np.where(given, 0.0, lambda1), "given_hinge_mask": given,
                   "given_hinge_margin": cfg["given_hinge_margin"],
                   "given_hinge_weight": cfg["given_hinge_weight"]}
    elif cfg["given_mode"] == "neg_l1":
        reward = cfg["lambda1_given"]
        options = {"lambda1": build_given_edges_lambda(
            given, lambda1, lambda1 if reward is None else reward)}
    else:
        raise ValueError(f"unknown given_mode {cfg['given_mode']!r}")
    # Nonlinear models honor a per-edge lambda1 (and the hinge) only when the
    # L1 acts on edge magnitudes.
    if cfg["model"] == "nonlinear":
        options["reg_on_adj"] = True
    return options


def _give(ds: Dataset, given_edges: list[tuple[str, str]], cfg: dict) -> Prepared:
    """Hand ``given_edges`` to the learner; with ``mask_reverse``, also forbid
    their reverses, so an encouraged u -> v can't simply be learned as v -> u
    (no real edge is lost: a DAG can't contain both directions)."""
    if not given_edges:
        return Prepared(ds)
    mask = _reverse_forbidden_mask(ds, given_edges) if cfg["mask_reverse"] else None
    return Prepared(ds, given_edges, mask, _encode_given(ds, given_edges, cfg))


def given_edges(ds: Dataset, cfg: dict, rng: np.random.Generator) -> Prepared:
    """Give a ``frac_given`` of the true edges."""
    return _give(ds, _choose(_true_edges(ds), cfg["frac_given"], rng), cfg)


def given_colliders(ds: Dataset, cfg: dict, rng: np.random.Generator) -> Prepared:
    """Give both incoming edges of a ``frac_colliders`` of the colliders."""
    chosen = _choose(find_colliders(ds.directed_subgraph()), cfg["frac_colliders"], rng)
    return _give(ds, sorted({e for a, b, c in chosen for e in ((a, c), (b, c))}), cfg)


# ---- Graph growth -------------------------------------------------------------------

def graph_growth(ds: Dataset, cfg: dict, rng: np.random.Generator) -> Prepared:
    """Keep the variables of ``n_colliders`` randomly ordered colliders and
    marginalize the rest (latent projection: the data stay faithful to the
    marginal graph; induced bidirected edges are left out of the scored truth)."""
    colliders = find_colliders(ds.directed_subgraph())
    keep: set[str] = set()
    for i in rng.permutation(len(colliders))[: cfg["n_colliders"]]:
        keep |= set(colliders[i])
    return Prepared(ds.hide_variables([v for v in ds.variables if v not in keep]))


PREPARE = {
    "tier_mask": tier_mask,
    "given_edges": given_edges,
    "given_edges_neg_l1": given_edges,
    "given_colliders": given_colliders,
    "graph_growth": graph_growth,
}


def resolve_axes(axes: dict[str, list], ds: Dataset) -> dict[str, list]:
    """The config's axes with dataset-dependent values filled in:
    ``n_colliders = "all"`` -> 1, 2, ..., the dataset's collider count."""
    if axes.get("n_colliders") == "all":
        axes = {**axes, "n_colliders": list(range(1, len(find_colliders(ds.directed_subgraph())) + 1))}
    return axes
