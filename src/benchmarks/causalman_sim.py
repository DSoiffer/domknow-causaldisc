# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This file uses the CausalMan simulator (boschresearch/CausalMan, Copyright (c)
# Robert Bosch GmbH, AGPL-3.0) and is distributed under the GNU Affero General
# Public License, version 3 or later; see LICENSE-AGPL-3.0 at the repository root.
# It is a standalone data-preparation step: `python -m benchmarks causalman_sim`
# writes a pickled `Dataset` that the experiment code reads, and nothing outside
# `benchmarks` imports this module.
"""Build a benchmark from the CausalMan simulator's ``causalman_micro`` SCM.

The raw simulator output is not usable for structure learning as is: every
non-source conditional is a point mass, about 40% of the nodes are constant,
and several variable pairs are exact bijections of each other. `build` fixes
that, in this order:

1. **process noise**: an independent ``Normal(0, noise_frac * sd)`` term is
   added inside each continuous non-source node's structural equation before
   sampling, so it propagates to descendants (measurement noise on top of a
   deterministic node would break the Markov property instead);
2. drop the **quality layer** (``*_MpGood``, ``ProcessResult``), a pure sink;
3. drop the **leak branch**, a dead end feeding nothing;
4. drop the two **childless** non-constant nodes (`DEAD_END_NODES`);
5. drop **constants**;
6. **contract deterministic ties**, keeping one node per group; noise cannot
   break these;
7. drop **isolated** nodes.

Two regimes:

- **one sub-batch** (default: 5000 rows, 33 variables, 38 edges). The
  simulator's two ground-truth levels coincide here, so the graph is
  unambiguous.
- **pooled** (``pooled=True``): every (sub-batch, parallel path) sampled and
  concatenated, a mixture of 12 environments (`environment_key`). The ground
  truth is level 2, which adds the mechanism-switching edges (e.g. supplier ->
  modulus). The built `Dataset` carries
  ``environment`` (each row's environment index) and ``environment_labels``,
  for methods that use environments; they are aligned with ``ds.data``'s rows
  and are not carried over by `Dataset` operations.

  Limitation: the pooled ground truth is not fully accurate. The section ID
  (``PSec_C2_pSection_ID_num``) is an exact threshold function of each of its
  children ``PF_M1_T{1,2}_s0`` and ``PF_M1_KstiffMachine`` (its two source
  distributions are ~16 sd apart and are not noised, being sources), so its 6
  outgoing edges are indistinguishable from edges out of those children; the
  one to ``PF_M1_KstiffMachine_REF`` has no effect at all (`unlearnable_edges`
  finds it). And the supplier IDs are dependent through the production
  schedule (which supplier combinations run together), which the graph does
  not model: no edge or common cause links them.

Variables are left unscaled; the learners z-score them.

Ties are found by reading the SCM symbolically rather than testing sampled
columns: that is exact, and it refuses to contract pairs that are only
*usually* tied, such as a ReLU whose input rarely crosses zero (a data test
calls that pair tied and would delete a real edge). Pooled, the supplier IDs
and the moduli they set are also tied, with no equation between them (both are
per-batch constants); `discrete_ties` finds those in the data.

Needs the simulator, which is not a project dependency. Sparse-clone it into
``CausalMan/`` (see README) and build with
``uv run --with-editable ./CausalMan python -m benchmarks causalman_sim``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Literal

import causalman
import networkx as nx
import numpy as np
import pandas as pd
import sympy
from causalman.utils.data import clean_df
from causalman.utils.graph import sample_CausalGraph
from causalman.utils.serialization import load_pickle
from sympy.stats import Normal
from sympy.stats.rv import RandomSymbol

from dataset import Dataset

VARIANT = "causalman_micro"
_ROOT = Path(causalman.__file__).parent / "dataset_objects" / VARIANT

# Childless in the full graph and outside the quality and leak layers: computed
# by the SCM but consumed by nothing.
DEAD_END_NODES = ("HU_Force_Lim_Deform", "HU_Ch1_Vchamber")

# Suffixes that lose when picking a tied group's representative: keep `Kstiff`
# over `Kstiff_inv`, `deltaD_max` over `deltaD_mean`.
TIE_LOSING_SUFFIXES = ("_inv", "_mean")


def _load_dag(batch: str, path_idx: int, level: int):
    if not _ROOT.is_dir():
        raise FileNotFoundError(f"No {VARIANT} dataset objects at {_ROOT}; see the README "
                                "for the sparse-checkout commands.")
    with open(_ROOT / "batch_data" / f"batch_{batch}" / f"dag_level_{level}_{path_idx}.pkl",
              "rb") as fh:
        return load_pickle(fh)


@cache
def _batch_info() -> pd.DataFrame:
    return pd.read_csv(_ROOT / f"{VARIANT}_1_batch_info.csv").set_index("subbatch_ID_unique")


def subbatch_paths() -> list[tuple[str, int]]:
    """Every (sub-batch, parallel path) the simulator has an SCM for (101)."""
    return [(d.name.removeprefix("batch_"), p)
            for d in sorted(x for x in (_ROOT / "batch_data").iterdir() if x.is_dir())
            for p in (0, 1) if (d / f"dag_level_1_{p}.pkl").exists()]


def environment_key(batch: str, path_idx: int) -> tuple[str, str, str, int]:
    """``(T1 supplier, T2 supplier, HU supplier, section ID)``: the regime a
    (sub-batch, path) runs in. There are 12; given them, only the batch label
    still varies. ``path_idx`` is not the section (both sections occur at
    each index), so the key reads the section ID from the SCM."""
    dag = _load_dag(batch, path_idx, level=1)
    section = dag.nodes["PSec_C2_pSection_ID_num"]["source_distribution"].rhs
    section_id = int(next(iter(section.pspace.distribution.dict)))
    row = _batch_info().loc[batch, ["T1_MV_Supplier", "T2_MV_Supplier", "HU_Block_Supplier"]]
    return (str(row.iloc[0]), str(row.iloc[1]), str(row.iloc[2]), section_id)


# ---------------------------------------------------------------- symbolic ties


@dataclass(frozen=True)
class Tie:
    """``node`` is a deterministic bijection of ``parent``."""

    node: str
    parent: str
    reason: str
    expr: str  # the simplified right-hand side


def _constant_substitutions(dag, sample_constants: dict[str, float]) -> dict:
    """``{symbol: value}`` for everything constant in this sample.

    Both kinds matter. Sources whose distribution is a plain number:
    ``deltaD_mean = beta_asym*deltaD_max + deltaD_min*(1-beta_asym)`` collapses
    to ``deltaD_max`` only because ``beta_asym = 1``. And columns constant in
    this sample: within one sub-batch the supplier is fixed, so ``E_eff`` is
    constant and ``KstiffPF`` becomes a rescaling of ``deltaD_mean`` (pooled,
    it varies again, so this is read off the sample).
    """
    subs = {}
    for _, data in dag.nodes(data=True):
        src = data.get("source_distribution")
        if data.get("term") is not None or src is None:
            continue
        if not src.rhs.atoms(RandomSymbol) and not src.rhs.free_symbols:
            subs[src.lhs] = src.rhs
    for name, value in sample_constants.items():
        subs.setdefault(sympy.Symbol(name), sympy.Float(value))
    return subs


def _is_bijection_in(expr, sym, lo: float, hi: float) -> tuple[bool, str]:
    """Whether ``expr`` is strictly monotone in ``sym`` over the observed range.

    Piecewise / Max / Min / Heaviside / Abs are rejected outright: those are
    ReLU-style gates, not bijections, even when the data never hits the kink.
    """
    if expr.atoms(sympy.Piecewise, sympy.Max, sympy.Min, sympy.Heaviside, sympy.Abs):
        return False, "piecewise/clipping"
    deriv = sympy.diff(expr, sym)
    if deriv.is_zero:
        return False, "constant in the parent"
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    signs = set()
    for value in np.linspace(lo - pad, hi + pad, 33):
        try:
            d = complex(deriv.subs(sym, float(value)).evalf())
        except (TypeError, ValueError):
            return False, "derivative not numerically evaluable"
        if abs(d.imag) > 1e-9 or not np.isfinite(d.real) or abs(d.real) < 1e-12:
            return False, "derivative undefined or vanishing on the observed range"
        signs.add(d.real > 0)
    if len(signs) != 1:
        return False, "derivative changes sign"
    return True, "strictly monotone"


def structural_ties(dag, ranges: dict[str, tuple[float, float]],
                    sample_constants: dict[str, float]) -> list[Tie]:
    """Pairs the SCM makes exact bijections of each other, after folding in
    constants: an equation that reduces to a strictly monotone function of one
    other variable (``Kstiff = 1/Kstiff_inv``), or two nodes with the same
    simplified equation. ``ranges`` (``{node: (min, max)}`` from the sample)
    limits the monotonicity check to where the data lives."""
    subs = _constant_substitutions(dag, sample_constants)
    simplified = {}
    for node, data in dag.nodes(data=True):
        if data.get("term") is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                simplified[node] = sympy.simplify(data["term"].rhs.subs(subs))

    ties = []
    for node, expr in simplified.items():
        free = [s for s in expr.free_symbols if str(s) in dag.nodes]
        if len(free) != 1 or str(free[0]) not in ranges:
            continue
        lo, hi = ranges[str(free[0])]
        ok, why = _is_bijection_in(expr, free[0], lo, hi)
        if ok:
            ties.append(Tie(node, str(free[0]), why, str(expr)))

    by_expr: dict[str, list[str]] = {}
    for node, expr in simplified.items():
        if expr.free_symbols:
            by_expr.setdefault(sympy.srepr(expr), []).append(node)
    for members in by_expr.values():
        head, *rest = sorted(members)
        ties += [Tie(other, head, "identical equation", str(simplified[head])) for other in rest]
    return ties


def discrete_ties(data: pd.DataFrame, max_unique: int = 20) -> list[Tie]:
    """1-1 maps between low-cardinality columns, found in the data (pooled:
    supplier ID <-> modulus). Limited to at most ``max_unique`` levels: a
    continuous column has a distinct value per row and would pass trivially."""
    low = [c for c in data.columns if 1 < data[c].nunique() <= max_unique]
    ties = []
    for i, a in enumerate(low):
        for b in low[i + 1:]:
            n = data[a].nunique()
            if n == data[b].nunique() == len(data[[a, b]].drop_duplicates()):
                ties.append(Tie(b, a, f"discrete bijection ({n} levels)", f"1-1 map with {a}"))
    return ties


def tie_groups(ties: list[Tie]) -> list[set[str]]:
    """The connected groups of pairwise ties."""
    g = nx.Graph()
    g.add_edges_from((t.node, t.parent) for t in ties)
    return [set(c) for c in nx.connected_components(g)]


def pick_representative(group: set[str], observable: set[str]) -> str:
    """The member of a tied group to keep: an observable one, then one without
    a losing suffix, then alphabetical. Not cosmetic: keeping ``Kstiff`` leaves
    ``delta_sgrad = Force / Kstiff`` (reciprocal); keeping ``Kstiff_inv`` would
    make it bilinear."""
    def rank(name: str):
        return (name not in observable, sum(name.endswith(s) for s in TIE_LOSING_SUFFIXES), name)
    return min(group, key=rank)


# ---------------------------------------------------------------- process noise


def _looks_discrete(column: pd.Series, max_unique: int = 10) -> bool:
    """Integer-valued with few levels: a flag or an ID, not a measurement.
    Cardinality alone would not do: pooled, ``E_eff`` takes two values (the
    supplier is binary) but is a continuous modulus that must be noised."""
    values = pd.to_numeric(column, errors="coerce").dropna()
    if values.empty or values.nunique() > max_unique:
        return False
    return bool(np.allclose(np.mod(values.to_numpy(float), 1.0), 0.0))


def _noise_scales(frame: pd.DataFrame) -> tuple[dict[str, float], set[str]]:
    """Per-node sd, and the discrete nodes, measured on ``frame``."""
    return ({c: float(frame[c].std()) for c in frame.columns},
            {c for c in frame.columns if _looks_discrete(frame[c])})


def add_process_noise(dag, scales: dict[str, float], discrete: set[str], frac: float):
    """A copy of ``dag`` with ``Normal(0, frac * scales[node])`` added to each
    non-source node's equation, and the number of nodes noised. Skips constants
    (no scale), discrete nodes (noise would destroy a 0/1 flag) and equations
    already carrying a random symbol (the engine allows one)."""
    noisy = dag.copy()
    applied = 0
    for node in noisy.nodes:
        term = noisy.nodes[node].get("term")
        if term is None or node not in scales or term.rhs.atoms(RandomSymbol):
            continue
        sd = scales[node]
        if not np.isfinite(sd) or sd == 0.0 or node in discrete:
            continue
        noisy.nodes[node]["term"] = sympy.Eq(term.lhs, term.rhs + Normal(f"e_{node}", 0, frac * sd))
        applied += 1
    return noisy, applied


def _sample(dag, n: int, seed: int) -> pd.DataFrame:
    frame = sample_CausalGraph(dag, n, random_state=seed)
    return clean_df(frame, list(frame.columns)).reset_index(drop=True)


def _sample_noisy(batch: str, path_idx: int, n: int, seed: int, noise_frac: float,
                  scales=None) -> tuple[pd.DataFrame, int]:
    """Sample one (sub-batch, path) with process noise; ``scales`` (sds and
    discrete nodes) defaults to this SCM's own, from a 500-row noiseless sample."""
    dag = _load_dag(batch, path_idx, level=1)
    if noise_frac == 0:
        return _sample(dag, n, seed), 0
    dag, applied = add_process_noise(dag, *(scales or _noise_scales(_sample(dag, 500, seed))),
                                     noise_frac)
    return _sample(dag, n, seed), applied


def _pooled_sizes(combos: list[tuple[str, int]], mode: str, n_samples: int) -> dict:
    """Rows per (sub-batch, path). "faithful": each sub-batch's real production
    run size (``batch_size_run``, ~14.8k rows in all; environments get 234 to
    2722), split over its paths. "balanced": ``n_samples`` per environment,
    split over its combinations (environments hold 2 to 17, so equal rows per
    combination would still leave an 8.5x spread)."""
    if mode == "faithful":
        paths = pd.Series([b for b, _ in combos]).value_counts()
        return {(b, p): max(1, round(float(_batch_info().loc[b, "batch_size_run"]) / paths[b]))
                for b, p in combos}
    if mode != "balanced":
        raise ValueError(f"pooled_sizes must be 'faithful' or 'balanced', got {mode!r}")
    per_env: dict[tuple, list] = {}
    for combo in combos:
        per_env.setdefault(environment_key(*combo), []).append(combo)
    return {c: max(1, round(n_samples / len(m))) for m in per_env.values() for c in m}


def unlearnable_edges(ds: Dataset) -> list[tuple[str, str]]:
    """Level-2-only (mechanism-switching) edges of ``ds`` whose child's source
    distribution is the same in every (sub-batch, path): the parent never
    changes it, so no method can recover the edge. Most such children are
    constants and dropped; this checks rather than assumes it."""
    combos = subbatch_paths()
    l2_only = set(_load_dag(*combos[0], level=2).edges) - set(_load_dag(*combos[0], level=1).edges)
    children = {c for _, c in l2_only}
    seen: dict[str, set[str]] = {}
    for combo in combos:
        dag = _load_dag(*combo, level=1)
        for child in children & set(dag.nodes):
            src = dag.nodes[child].get("source_distribution")
            if src is not None:
                rhs = src.rhs
                seen.setdefault(child, set()).add(
                    str(rhs.pspace.distribution) if hasattr(rhs, "pspace") else str(rhs))
    frozen = {c for c, values in seen.items() if len(values) == 1}
    return [(p, c) for p, c in ds.graph.edges if (p, c) in l2_only and c in frozen]


# ------------------------------------------------------------------------ build


def build(*, pooled: bool = False, pooled_sizes: Literal["faithful", "balanced"] = "faithful",
          batch: str = "1_0_111_8_1", path_idx: int = 0, n_samples: int = 5000,
          seed: int = 42, noise_frac: float = 0.01, keep_leak: bool = False,
          keep_quality: bool = False, keep_dead_ends: bool = False,
          contract_ties: bool = True, drop_isolated: bool = True,
          verbose: bool = True) -> Dataset:
    """Sample one sub-batch (``batch``, parallel path ``path_idx``;
    ``n_samples`` rows), or with ``pooled`` every one (``pooled_sizes``, see
    `_pooled_sizes`; ``n_samples`` is then per environment and only used by
    "balanced"), and apply the steps in the module docstring. The ``keep_*`` /
    ``contract_ties`` / ``drop_isolated`` flags skip a step (ablations);
    ``noise_frac=0`` leaves the SCM deterministic."""
    env = None
    if pooled:
        combos = subbatch_paths()
        # Pooled noise scales: E_eff is constant within each sub-batch but not
        # across them, so per-sub-batch sds would skip it and leave it degenerate.
        scales = (_noise_scales(pd.concat([_sample(_load_dag(*c, level=1), 40, seed)
                                           for c in combos], ignore_index=True))
                  if noise_frac > 0 else None)
        sizes = _pooled_sizes(combos, pooled_sizes, n_samples)
        frames, keys, applied = [], [], 0
        for combo in combos:
            frame, k = _sample_noisy(*combo, sizes[combo], seed, noise_frac, scales)
            frames.append(frame)
            keys += [environment_key(*combo)] * len(frame)
            applied = max(applied, k)
        data = pd.concat(frames, ignore_index=True)
        labels = sorted(set(keys))
        env = np.array([labels.index(k) for k in keys])
        ref = combos[0]
    else:
        data, applied = _sample_noisy(batch, path_idx, n_samples, seed, noise_frac)
        ref = (batch, path_idx)

    truth = _load_dag(*ref, level=2)
    graph = nx.DiGraph()
    graph.add_nodes_from((str(n), {"observable": bool(d.get("Observable", False))})
                         for n, d in truth.nodes(data=True))
    graph.add_edges_from((str(u), str(v)) for u, v in truth.edges())
    ds = Dataset(data=data.loc[:, list(graph.nodes)], graph=graph, graph_type="dag")
    steps = [(f"sampled ({applied} nodes noised)", ds)]

    # Ties come from the noiseless equations: noise neither creates nor breaks one.
    ranges = {c: (float(data[c].min()), float(data[c].max())) for c in ds.variables}
    constants = {c: float(data[c].iloc[0]) for c in ds.variables if data[c].std() == 0.0}
    ties = structural_ties(_load_dag(*ref, level=1), ranges, constants) + discrete_ties(ds.data)

    def drop(label, names):
        nonlocal ds
        ds = ds.drop_columns(sorted({n for n in names if n in ds.variables}))
        steps.append((label, ds))

    if not keep_quality:
        drop("- quality layer", [c for c in ds.variables if "MpGood" in c or "ProcessResult" in c])
    if not keep_leak:
        drop("- leak branch", [c for c in ds.variables if "leak" in c.lower()])
    if not keep_dead_ends:
        drop("- dead ends", DEAD_END_NODES)
    ds = ds.drop_singletons()
    steps.append(("- constants", ds))
    if contract_ties:
        observable = {n for n, d in ds.graph.nodes(data=True) if d["observable"]}
        for group in tie_groups(ties):
            present = group & set(ds.variables)
            if len(present) > 1:
                keep = pick_representative(present, observable)
                ds = ds.contract_nodes(sorted(present - {keep}))
                if verbose:
                    print(f"  tie: keep {keep}, contract {sorted(present - {keep})}")
        steps.append(("contract ties", ds))
    if drop_isolated:
        drop("- isolated", [n for n in ds.graph.nodes if ds.graph.degree(n) == 0])

    if verbose:
        for label, step in steps:
            print(f"  {label:32s} {len(step.variables):4d} nodes "
                  f"{step.graph.number_of_edges():4d} edges")
    if pooled:
        bad = unlearnable_edges(ds)
        if bad:
            print(f"  warning: {len(bad)} unlearnable mechanism-switching edges: {bad}")
        ds.environment, ds.environment_labels = env, labels
        if verbose:
            print(f"  {len(labels)} environments, {np.bincount(env).min()}-"
                  f"{np.bincount(env).max()} rows each")
    return ds


def load_causalman_sim() -> Dataset:
    """The default build: one sub-batch."""
    return build()


def load_causalman_sim_pooled() -> Dataset:
    """All sub-batches pooled, at their real production sizes."""
    return build(pooled=True)
