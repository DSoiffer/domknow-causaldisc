from __future__ import annotations

import itertools
import re
from pathlib import Path

import networkx as nx
import numpy as np

from benchmarks.layout import reference_spec
from dataset import Dataset


def find_colliders(
    graph: nx.DiGraph, *, include_shielded: bool = False
) -> list[tuple[str, str, str]]:
    """Return (A, B, C) for each A -> C <- B collider in the graph."""
    out: list[tuple[str, str, str]] = []
    for c in graph.nodes:
        parents = sorted(str(p) for p in graph.predecessors(c))
        for a, b in itertools.combinations(parents, 2):
            if not include_shielded:
                if graph.has_edge(a, b) or graph.has_edge(b, a):
                    continue
            out.append((a, b, str(c)))
    return out


def _node_y(node: str, positions: dict[str, tuple[float, float]]) -> float:
    """Vertical coordinate of a node in the reference layout (larger ``y`` is
    lower in the drawing: vis-network exports place the origin at the top)."""
    if node not in positions:
        raise ValueError(f"Reference layout is missing a position for node {node!r}.")
    return positions[node][1]


def build_vertical_mask(
    ds: Dataset,
    reference_html: str | Path | None = None,
    *,
    dataset: str = "dataset0",
    observability: str = "full",
    allow_same_level: bool = True,
    subsample_fraction: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Build a directed allow-mask from a reference layout's vertical order.

    A node may only send directed edges *upward* -- to nodes drawn higher in
    the reference graph (smaller ``y``). Downward edges (the reverse direction)
    are forbidden, and so are self-loops (the diagonal is always ``0``).

    The returned matrix is indexed by ``ds.variables`` -- the column order of
    the training inputs -- so it lines up directly with dagrad's ``X``. The
    dataset may have had nodes removed; only
    its current variables appear in the mask. Each variable must have a
    position in the reference layout.

    Parameters
    ----------
    ds:
        Dataset whose variables define the mask's rows/columns.
    reference_html:
        Path to the reference vis-network HTML export. Resolved relative to the
        dataset directory first, then the repo root. If ``None``, the dataset's
        default reference for the given observability is used.
    dataset:
        Dataset directory name (under ``data/``) used to resolve the default
        reference and relative paths.
    observability:
        ``"full"`` or ``"partial"``; selects the default reference file.
    allow_same_level:
        If ``True`` (default), edges between nodes at the *same* vertical level
        are allowed in both directions (only strictly-downward edges are
        forbidden). If ``False``, only strictly-upward edges are allowed, so
        same-level pairs are forbidden in both directions.
    subsample_fraction:
        Fraction in ``[0, 1]`` of the (off-diagonal) masked entries to *retain*
        as forbidden. ``1.0`` (default) keeps the full mask; smaller values
        randomly flip the rest back to allowed, giving a weaker mask. Self-loops
        (the diagonal) are never counted toward the fraction and are always kept
        forbidden.
    rng:
        Optional ``np.random.Generator`` used to choose which masked entries to
        drop. If ``None``, a fresh default generator is used.

    Returns
    -------
    np.ndarray
        A ``(p, p)`` float allow-mask (``1`` = allowed, ``0`` = forbidden),
        where ``p == len(ds.variables)``.
    """
    positions = reference_spec(dataset, observability, reference_html)["positions"]

    names = [str(n) for n in ds.variables]
    ys = np.array([_node_y(name, positions) for name in names], dtype=float)

    # mask[i, j] allows i -> j when j is further up than i (smaller y).
    strictly_up = ys[None, :] < ys[:, None]
    if allow_same_level:
        same_level = ys[None, :] == ys[:, None]
        allowed = strictly_up | same_level
    else:
        allowed = strictly_up
    mask = allowed.astype(float)
    np.fill_diagonal(mask, 0.0)  # no self-loops
    return _subsample_forbidden_mask(mask, subsample_fraction, rng)


def _subsample_forbidden_mask(
    mask: np.ndarray, subsample_fraction: float, rng: np.random.Generator | None
) -> np.ndarray:
    """Retain ``subsample_fraction`` of an allow-mask's forbidden off-diagonal
    entries, flipping the rest back to allowed (``1.0`` keeps the full mask).
    Self-loops on the diagonal are excluded from the count and always kept
    forbidden. Mutates and returns ``mask``."""
    if not 0.0 <= subsample_fraction <= 1.0:
        raise ValueError(
            f"subsample_fraction must be in [0, 1], got {subsample_fraction}"
        )
    if subsample_fraction < 1.0:
        if rng is None:
            rng = np.random.default_rng()
        forbidden = mask == 0.0
        np.fill_diagonal(forbidden, False)
        idx = np.flatnonzero(forbidden)
        n_drop = int(round((1.0 - subsample_fraction) * idx.size))
        if n_drop > 0:
            mask.flat[rng.choice(idx, size=n_drop, replace=False)] = 1.0
    return mask


def build_dag_tier_mask(
    ds: Dataset,
    *,
    allow_same_level: bool = True,
    subsample_fraction: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Directed allow-mask from the true DAG's topological tiers.

    The no-reference-layout analogue of `build_vertical_mask` (used for real
    benchmarks such as Sachs that ship no exported layout): each variable's tier
    is its index in ``nx.topological_generations`` of the directed ground-truth
    graph, and an edge ``i -> j`` is allowed iff ``j`` sits in a later-or-equal
    tier -- forbidding only edges that run *backward* across tiers. The mask is
    indexed by ``list(ds.variables)``. ``allow_same_level`` (allow both
    directions between same-tier nodes), ``subsample_fraction`` and ``rng``
    behave exactly as in `build_vertical_mask`; the diagonal is always
    forbidden.
    """
    dag = ds.directed_subgraph()
    tier = {str(n): i for i, gen in enumerate(nx.topological_generations(dag)) for n in gen}
    names = [str(n) for n in ds.variables]
    t = np.array([tier[n] for n in names], dtype=float)

    allowed = t[None, :] >= t[:, None] if allow_same_level else t[None, :] > t[:, None]
    mask = allowed.astype(float)
    np.fill_diagonal(mask, 0.0)  # no self-loops
    return _subsample_forbidden_mask(mask, subsample_fraction, rng)


def build_station_tier_mask(
    ds: Dataset,
    *,
    allow_same_level: bool = True,
    subsample_fraction: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Directed allow-mask from CausalAssembly station numbers.

    The assembly line only flows forward, so each variable's tier is the integer
    ``k`` in its ``Station<k>_...`` name and an edge ``i -> j`` is allowed iff
    ``j``'s station is later-or-equal -- forbidding every edge that would run
    from a higher-numbered station to a lower-numbered one. Unlike
    `build_dag_tier_mask` (per-node topological generations), the tiers here are
    the 5 stations, so within-station structure is left entirely free (both
    directions when ``allow_same_level``). Mask is indexed by
    ``list(ds.variables)``; the diagonal is always forbidden. ``subsample_fraction``
    / ``rng`` behave as in `build_dag_tier_mask`. Every node must carry a
    ``Station<k>`` prefix.
    """
    names = [str(n) for n in ds.variables]

    def station_num(name: str) -> int:
        m = re.match(r"Station(\d+)", name)
        if m is None:
            raise ValueError(
                f"node {name!r} has no 'Station<k>' prefix; station tiering "
                "only applies to the CausalAssembly benchmark."
            )
        return int(m.group(1))

    t = np.array([station_num(n) for n in names], dtype=float)
    allowed = t[None, :] >= t[:, None] if allow_same_level else t[None, :] > t[:, None]
    mask = allowed.astype(float)
    np.fill_diagonal(mask, 0.0)  # no self-loops
    return _subsample_forbidden_mask(mask, subsample_fraction, rng)


def build_station_dag_tier_mask(
    ds: Dataset,
    *,
    allow_same_level: bool = True,
    subsample_fraction: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Union of the station and DAG tier masks: forbid an edge if *either*
    forbids it.

    The strongest correct tier prior available here -- it layers the
    fine-grained topological-generation ordering on top of the station ordering,
    so it forbids strictly more than either alone while (because both component
    masks remove zero true edges) still removing none. ``subsample_fraction`` is
    applied to the combined forbidden set, so it thins the union rather than each
    mask independently.
    """
    station = build_station_tier_mask(ds, allow_same_level=allow_same_level)
    dag = build_dag_tier_mask(ds, allow_same_level=allow_same_level)
    mask = station * dag  # allowed (1) iff both allow; forbidden (0) if either does
    np.fill_diagonal(mask, 0.0)  # no self-loops
    return _subsample_forbidden_mask(mask, subsample_fraction, rng)


def build_given_edges_lambda(
    given: np.ndarray, lambda1: float, lambda1_given: float
) -> np.ndarray:
    """Per-edge L1 weight matrix encoding 'directly given' edges.

    Non-given edges get the usual sparsity penalty ``lambda1``; given edges
    (``given[i, j]`` truthy) get ``-lambda1_given``, *encouraging* them to
    exist. The diagonal keeps ``lambda1`` (dagrad holds self-loops at zero).
    """
    given = np.asarray(given, dtype=bool)
    return np.where(given, -float(lambda1_given), float(lambda1))


def given_matrix(names: list[str], given_edges: list[tuple[str, str]]) -> np.ndarray:
    """(d, d) boolean matrix of the given directed edges, indexed by ``names``."""
    index = {n: i for i, n in enumerate(names)}
    mask = np.zeros((len(names), len(names)), dtype=bool)
    for u, v in given_edges:
        mask[index[u], index[v]] = True
    return mask


__all__ = [
    "build_dag_tier_mask",
    "build_given_edges_lambda",
    "build_station_dag_tier_mask",
    "build_station_tier_mask",
    "build_vertical_mask",
    "find_colliders",
    "given_matrix",
]
