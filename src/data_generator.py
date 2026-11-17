"""Reverse-engineered data-generating process for a `Dataset`.

`DataGenerator.fit(dataset)` learns one structural equation per node of the
ground-truth DAG:

* **root nodes** (no parents)  -> a fitted Gaussian ``N(mean, std)``;
* **linear nodes**             -> an (exact) linear function of their parents,
  ``y = intercept + sum_i coef_i * parent_i``;
* **multiplicative nodes**     -> a product of their parents, ``y = coef * prod_i parent_i``
  (auto-detected as a fallback when the linear fit fails -- this is how the
  ``*_Force = LmvPF * KstiffPF`` nodes are captured).

Coefficients are snapped to integers when they sit within ``round_tol`` of one
(this removes ~1e-10 least-squares dust, e.g. a multiplicative coefficient of
``1.0000000000`` or an intercept of ``2.5e-10``).

Any non-root node that is neither an (exact) linear nor a multiplicative
function of its parents is reported as a failure (printed and stored in
``.failures``); it gets no model, and `generate` will refuse to run until it is
resolved.

`generate(n, noise=...)` produces a new `Dataset` with the same structure by
ancestral sampling, with an optional per-node noise knob. The noisy benchmark
variants come from `noisy_like`, via ``benchmarks.load(name, noise=...)``.

For CausalMan, fitting the equations from data is now mostly unnecessary:
Bosch has released the simulator itself (boschresearch/CausalMan), with the
same structural causal model as the Zenodo data, and `benchmarks.causalman_sim`
builds benchmarks from it with noise injected inside its exact equations. The
fit here remains what makes noisy copies of the Zenodo-based ``causalman``
benchmark (whose 24 variables, one product configuration, are mostly latent in
the simulator's observable set) and of any other dataset whose mechanisms it
can recover.

To see the fitted equations (printed by ``fit``)::

    from benchmarks import load
    from data_generator import DataGenerator
    DataGenerator.fit(load("causalman"))
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Union

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

from dataset import Dataset, GraphType

# noise spec accepted by `generate`:
#   None                         -> no extra noise
#   float                        -> additive N(0, std) on every node
#   {node: std}                  -> additive N(0, std) per node (missing -> 0)
#   callable(node, n, rng)->arr  -> full control; return an array to add (or None)
NoiseSpec = Union[None, float, Mapping[str, float], Callable[[str, int, np.random.Generator], np.ndarray]]


def _round_near_int(x: float, tol: float) -> float:
    """Snap ``x`` to the nearest integer if it is within ``tol`` of one."""
    r = round(x)
    return float(r) if abs(x - r) <= tol else float(x)


# --------------------------------------------------------------------------- #
# Structural-equation models. Each exposes `.parents` and `.evaluate(...)`.
@dataclass
class RootGaussian:
    mean: float
    std: float
    # goodness of the Gaussian fit (measured against the fitted N(mean, std)):
    skew: float = 0.0          # 0 for a true normal
    excess_kurt: float = 0.0   # 0 for a true normal
    ks_stat: float = 0.0       # KS D: max |empirical CDF - normal CDF| (0 = perfect)
    ppcc: float = 1.0          # normal probability-plot correlation (1 = perfect)
    n_obs: int = 0
    gaussian_ok: bool = True   # passed all tolerances below
    parents: list[str] = field(default_factory=list)

    @classmethod
    def fit(cls, y: np.ndarray, *, skew_tol: float, kurt_tol: float, ks_tol: float, round_tol: float) -> "RootGaussian":
        mean, std = float(np.mean(y)), float(np.std(y, ddof=0))
        mean = _round_near_int(mean, round_tol)
        sk, ek = float(stats.skew(y)), float(stats.kurtosis(y))
        if std > 0:
            ks = float(stats.kstest(y, "norm", args=(mean, std)).statistic)
            (_osm, _osr), (_slope, _icpt, r) = stats.probplot(y, dist="norm")
            ppcc = float(r)
        else:  # constant column: degenerate but trivially "fits"
            ks, ppcc = 0.0, 1.0
        ok = abs(sk) <= skew_tol and abs(ek) <= kurt_tol and ks <= ks_tol
        return cls(mean, std, sk, ek, ks, ppcc, int(len(y)), ok)

    def evaluate(self, n, parent_vals, rng):
        return rng.normal(self.mean, self.std, size=n)

    def describe(self):
        tag = "" if self.gaussian_ok else "  <- POOR GAUSSIAN FIT"
        return (f"~ N(mean={self.mean:.6g}, std={self.std:.6g})  "
                f"[skew={self.skew:+.2f} exkurt={self.excess_kurt:+.2f} "
                f"KS={self.ks_stat:.3f} ppcc={self.ppcc:.4f}]{tag}")


@dataclass
class LinearModel:
    parents: list[str]
    intercept: float
    coefs: list[float]

    def evaluate(self, n, parent_vals, rng):
        out = np.full(n, self.intercept, dtype=float)
        for c, p in zip(self.coefs, self.parents):
            out = out + c * parent_vals[p]
        return out

    def describe(self):
        terms = [f"{self.intercept:.6g}"] + [
            f"{c:.6g}*{p}" for c, p in zip(self.coefs, self.parents)
        ]
        return "= " + " + ".join(terms)


@dataclass
class MultiplicativeModel:
    parents: list[str]
    coef: float

    def evaluate(self, n, parent_vals, rng):
        out = np.full(n, self.coef, dtype=float)
        for p in self.parents:
            out = out * parent_vals[p]
        return out

    def describe(self):
        body = " * ".join(self.parents)
        prefix = "" if self.coef == 1 else f"{self.coef:.6g} * "
        return f"= {prefix}{body}"


NodeModel = Union[RootGaussian, LinearModel, MultiplicativeModel]


class DataGenerator:
    """A fitted structural causal model that can synthesize new data."""

    def __init__(
        self,
        graph: nx.DiGraph,
        variables: list[str],
        models: dict[str, NodeModel],
        topo_order: list[str],
        *,
        graph_type: GraphType = "dag",
        failures: list[tuple[str, str]] | None = None,
    ) -> None:
        self.graph = graph
        self.variables = variables
        self.models = models
        self.topo_order = topo_order
        self.graph_type = graph_type
        self.failures = failures or []

    # ----------------------------------------------------------------- fit
    @classmethod
    def fit(
        cls,
        dataset: Dataset,
        *,
        linear_tol: float = 1e-7,
        round_tol: float = 1e-8,
        root_skew_tol: float = 0.5,
        root_kurt_tol: float = 1.0,
        root_ks_tol: float = 0.05,
        force_linear: bool = False,
        verbose: bool = True,
    ) -> "DataGenerator":
        """Learn one structural equation per node from observed data.

        Parameters
        ----------
        linear_tol
            A node counts as an exact linear/multiplicative function of its
            parents if ``std(residual) / std(node) < linear_tol``.
        force_linear
            Give every non-root node its OLS linear fit, exact or not (no
            tolerance check, no multiplicative fallback) -- for testing how
            much the multiplicative nonlinearities matter to downstream
            learning.
        round_tol
            Coefficients within this distance of an integer are snapped to it.
        root_skew_tol, root_kurt_tol, root_ks_tol
            A root's Gaussian fit is flagged ``gaussian_ok=False`` (and warned
            about) unless ``|skew| <= root_skew_tol`` and
            ``|excess_kurt| <= root_kurt_tol`` and ``KS <= root_ks_tol``. The
            root is still modeled as Gaussian; this only reports fit quality.
        """
        graph = dataset.directed_subgraph()  # structural DAG
        data = dataset.data
        variables = dataset.variables

        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("DataGenerator.fit requires an acyclic directed graph.")

        models: dict[str, NodeModel] = {}
        failures: list[tuple[str, str]] = []

        for node in graph.nodes():
            y = data[node].to_numpy(dtype=float)
            parents = sorted(graph.predecessors(node))

            # --- root: fit a Gaussian (and assess how well it fits) -------
            if not parents:
                models[node] = RootGaussian.fit(
                    y, skew_tol=root_skew_tol, kurt_tol=root_kurt_tol, ks_tol=root_ks_tol, round_tol=round_tol
                )
                continue

            Pcols = [data[pc].to_numpy(dtype=float) for pc in parents]
            y_scale = y.std(ddof=0)
            denom = y_scale if y_scale > 0 else 1.0

            # --- try exact linear-in-parents ------------------------------
            # (force_linear: accept the OLS fit regardless of how good it is)
            design = np.column_stack([np.ones(len(y))] + Pcols)
            beta, *_ = np.linalg.lstsq(design, y, rcond=None)
            lin_rel = np.std(y - design @ beta) / denom
            if force_linear or lin_rel < linear_tol:
                intercept = _round_near_int(float(beta[0]), round_tol)
                coefs = [_round_near_int(float(b), round_tol) for b in beta[1:]]
                models[node] = LinearModel(parents=parents, intercept=intercept, coefs=coefs)
                continue

            # --- fall back to multiplicative (product of parents) ---------
            prod = np.ones(len(y))
            for pc in Pcols:
                prod = prod * pc
            pp = float(np.dot(prod, prod))
            coef = float(np.dot(prod, y) / pp) if pp > 0 else np.inf
            mul_rel = np.std(y - coef * prod) / denom
            if np.isfinite(coef) and mul_rel < linear_tol:
                models[node] = MultiplicativeModel(parents=parents, coef=_round_near_int(coef, round_tol))
                continue

            failures.append(
                (node, f"not linear (rel={lin_rel:.2e}) nor multiplicative (rel={mul_rel:.2e}) in parents {parents}")
            )

        topo_order = list(nx.topological_sort(graph))
        gen = cls(graph, variables, models, topo_order, graph_type=dataset.graph_type, failures=failures)
        gen.summary(verbose)
        return gen

    # ------------------------------------------------------------- generate
    def generate(
        self,
        n: int,
        *,
        noise: NoiseSpec = None,
        propagate_noise: bool = True,
        random_state: int | None = None,
        as_dataset: bool = True,
        noise_roots: bool = False
    ) -> Union[Dataset, pd.DataFrame]:
        """Synthesize ``n`` rows by ancestral sampling.

        Parameters
        ----------
        noise
            Optional extra noise added to each node as it is generated. See
            ``NoiseSpec``. By default no extra noise is added (roots still carry
            their fitted Gaussian variation; non-roots are deterministic).
        propagate_noise
            If True (default), a node's children see its noised value, so noise
            flows through the structural equations. If False, children are
            computed from the clean value and noise is applied only to the
            node's own output (measurement-style noise).
        """
        missing = [node for node in self.topo_order if node not in self.models]
        if missing:
            raise RuntimeError(
                f"Cannot generate: {len(missing)} node(s) have no fitted model "
                f"(failures): {missing}. Resolve these before generating."
            )

        rng = np.random.default_rng(random_state)
        noise_arr = self._resolve_noise(noise)

        compute_vals: dict[str, np.ndarray] = {}  # what children read
        output_vals: dict[str, np.ndarray] = {}  # what we record
        for node in self.topo_order:
            model = self.models[node]
            parent_vals = {p: compute_vals[p] for p in model.parents}
            clean = model.evaluate(n, parent_vals, rng)
            if isinstance(model, RootGaussian) and not noise_roots:
                add = None
            else:
                add = noise_arr(node, n, rng)  # can still be None
            noised = clean if add is None else clean + add
            compute_vals[node] = noised if propagate_noise else clean
            output_vals[node] = noised

        df = pd.DataFrame({node: output_vals[node] for node in self.topo_order})[self.variables]
        if not as_dataset:
            return df
        return Dataset(data=df, graph=self.graph.copy(), graph_type=self.graph_type)

    def _resolve_noise(self, noise: NoiseSpec) -> Callable[[str, int, np.random.Generator], np.ndarray | None]:
        if noise is None:
            return lambda node, n, rng: None
        if callable(noise):
            return noise  # type: ignore[return-value]
        if isinstance(noise, Mapping):
            def per_node(node, n, rng, _m=noise):
                std = float(_m.get(node, 0.0))
                return rng.normal(0.0, std, size=n) if std > 0 else None
            return per_node
        if isinstance(noise, (int, float)):
            std = float(noise)
            return lambda node, n, rng: (rng.normal(0.0, std, size=n) if std > 0 else None)
        raise TypeError(f"Unsupported noise spec: {type(noise)!r}")

    # ---------------------------------------------------------------- utils
    def summary(self, verbose) -> None:
        roots = [n for n, m in self.models.items() if isinstance(m, RootGaussian)]
        lins = [n for n, m in self.models.items() if isinstance(m, LinearModel)]
        muls = [n for n, m in self.models.items() if isinstance(m, MultiplicativeModel)]
        bad_roots = [n for n in roots if not self.models[n].gaussian_ok]
        if verbose:
            print(f"DataGenerator: {len(roots)} root (Gaussian; {len(bad_roots)} poor fit), "
                f"{len(lins)} linear, {len(muls)} multiplicative, {len(self.failures)} failed "
                f"(of {len(self.models) + len(self.failures)} nodes)")
            for node in self.topo_order:
                m = self.models.get(node)
                if m is not None:
                    print(f"  {node:32s} {m.describe()}")
        for node, reason in self.failures:
            print(f"  [FAIL] {node:27s} {reason}")
        if bad_roots:
            print(f"  WARNING: {len(bad_roots)} root(s) are a poor Gaussian fit "
                  f"(see [skew/exkurt/KS/ppcc] above): {bad_roots}")

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "DataGenerator":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} does not contain a DataGenerator")
        return obj


def noisy_like(
    gen: DataGenerator, base: Dataset, noise_fraction: float,
    *, n: int | None = None, seed: int = 0, propagate_noise: bool = True,
) -> Dataset:
    """Resample ``base``'s fitted SCM (``gen``) with ``N(0, noise_fraction*std)``
    added to each non-root node. ``n`` defaults to ``base``'s row count."""
    spec = {node: noise_fraction * float(base.data[node].std(ddof=0)) for node in gen.variables}
    return gen.generate(len(base.data) if n is None else n,
                        noise=spec, noise_roots=False, random_state=seed,
                        propagate_noise=propagate_noise)
