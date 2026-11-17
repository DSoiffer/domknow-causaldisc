"""Core `Dataset` object coupling a data frame with its ground-truth graph.

One column per variable (= graph node); the column order is the variable order
used everywhere downstream (training inputs, masks, saved graphs). Every
preprocessing method returns a new `Dataset`, keeping the data frame and the
graph in sync.
"""

from __future__ import annotations

import hashlib
import itertools
from os import PathLike
from pathlib import Path
from typing import Iterable, Literal

import networkx as nx
import numpy as np
import pandas as pd

StandardizeMethod = Literal["zscore", "minmax01", "minmax11", "center"]

GraphType = Literal["dag", "admg"]

DEFAULT_DROP_COLUMNS: tuple[str, ...] = ("partID", "batch_ID_num")


def _edge_is_bidirected(edge_data: dict) -> bool:
    """Whether a graph edge represents a bidirected edge."""
    return edge_data.get("directed", True) is False


class NotADAGError(ValueError):
    """Raised when a graph passed to / produced by `Dataset` contains a cycle."""


class Dataset:
    """A data frame paired with its ground-truth graph, one column per node.

    Instances are treated as immutable: preprocessing methods return new
    `Dataset` objects via `_replace(...)` rather than mutating in place.
    Invariants are checked at construction time and after every `_replace`.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        graph: nx.DiGraph,
        *,
        graph_type: GraphType = "dag",
    ) -> None:
        self.data = data
        self.graph = graph
        self.graph_type = graph_type
        self._check_invariants()

    @property
    def variables(self) -> list[str]:
        """The variable (= column = graph node) names, in column order."""
        return [str(c) for c in self.data.columns]

    def directed_subgraph(self) -> nx.DiGraph:
        """Read-only view containing only the directed edges of the graph.
        Used for the ADMG acyclicity check."""
        sub = nx.DiGraph()
        sub.add_nodes_from(self.graph.nodes())
        for u, v, edge_data in self.graph.edges(data=True):
            if not _edge_is_bidirected(edge_data):
                sub.add_edge(u, v)
        return sub

    def bidirected_edges(self) -> list[tuple[str, str]]:
        """Bidirected edges; empty for a plain DAG."""
        return [
            (u, v)
            for u, v, edge_data in self.graph.edges(data=True)
            if _edge_is_bidirected(edge_data)
        ]

    # ------------------------------------------------------------------ I/O

    @classmethod
    def from_files(
        cls,
        data_csv: str | PathLike[str],
        graph_gml: str | PathLike[str],
        *,
        graph_type: GraphType = "dag",
    ) -> "Dataset":
        """Load a data CSV and a GML ground-truth graph into a `Dataset`.

        Standard non-causal columns in `DEFAULT_DROP_COLUMNS` (e.g. `partID`,
        `batch_ID_num`) are dropped from the CSV up front and the drop is
        announced. After that, the CSV column set must exactly equal the graph
        node set; any mismatch is an error. Columns are put in graph-node order.

        Pass ``graph_type='admg'`` for partially observable ground truth whose
        GML encodes bidirected edges.
        """
        data = pd.read_csv(Path(data_csv))
        graph_raw = nx.read_graphml(Path(graph_gml))
        graph = graph_raw if isinstance(graph_raw, nx.DiGraph) else nx.DiGraph(graph_raw)

        graph_node_set = {str(n) for n in graph.nodes()}
        dropped_defaults = sorted(
            {c for c in DEFAULT_DROP_COLUMNS if c in data.columns or c in graph_node_set}
        )
        if dropped_defaults:
            data = data.drop(columns=[c for c in dropped_defaults if c in data.columns])
            graph.remove_nodes_from([n for n in dropped_defaults if n in graph_node_set])
            print(f"Dropped default non-causal columns on load: {dropped_defaults}")

        node_names = [str(n) for n in graph.nodes()]
        csv_cols = set(data.columns)
        graph_nodes = set(node_names)
        if csv_cols != graph_nodes:
            missing_in_csv = sorted(graph_nodes - csv_cols)
            extra_in_csv = sorted(csv_cols - graph_nodes)
            raise ValueError(
                "CSV columns and graph nodes must match exactly (after default drops). "
                f"Graph nodes missing from CSV: {missing_in_csv[:10]} | "
                f"CSV columns missing from graph: {extra_in_csv[:10]}"
            )

        return cls(data=data[node_names], graph=graph, graph_type=graph_type)

    # ----------------------------------------------------- preprocessing ops

    def drop_columns(self, names: Iterable[str]) -> "Dataset":
        """Remove the given variables from the dataset and the graph.

        Each name must exist; unknown names raise. Incident edges are removed
        with the nodes.
        """
        names = list(names)
        unknown = [n for n in names if n not in self.variables]
        if unknown:
            raise ValueError(f"Cannot drop unknown variables: {unknown}")

        new_graph = self.graph.copy()
        new_graph.remove_nodes_from(names)
        return self._replace(data=self.data.drop(columns=names), graph=new_graph)

    def contract_nodes(self, names: Iterable[str]) -> "Dataset":
        """Remove one or more nodes but preserve paths through them.

        Each node is contracted in turn on a shared graph copy: for every
        parent ``P`` and child ``C`` of the node, an edge ``P -> C`` is added
        before the node is removed, so any path that previously routed through
        it still exists. Contracting a chain works because bypass edges added
        for an earlier node feed into the next node's parent set. The result is
        independent of the order within ``names``. All names must be existing
        variables. Bypass edges cannot introduce a cycle because ``P`` precedes
        ``C`` in any topological order of the original DAG.
        """
        names = list(names)
        unknown = [n for n in names if n not in self.variables]
        if unknown:
            raise ValueError(f"Cannot contract unknown variables: {unknown}")

        new_graph = self.graph.copy()
        for name in names:
            parents = list(new_graph.predecessors(name))
            children = list(new_graph.successors(name))
            new_graph.add_edges_from((p, c) for p in parents for c in children if p != c)
            new_graph.remove_node(name)

        return self._replace(data=self.data.drop(columns=names), graph=new_graph)

    def standardize(self, method: StandardizeMethod = "minmax01") -> "Dataset":
        """Scale every variable with the chosen method.

        - ``"zscore"``: subtract mean, divide by population std.
        - ``"minmax01"``: rescale to ``[0, 1]``.
        - ``"minmax11"``: rescale to ``[-1, 1]``.
        - ``"center"``: subtract mean.

        Operates on the whole frame in one vectorized pandas pass. Constant
        columns (zero std / zero range) are left unchanged rather than
        producing NaN/inf. Only data values change; the graph does not.
        """
        block = self.data.astype(float)
        if method == "zscore":
            sigma = block.std(ddof=0).replace(0.0, 1.0)
            scaled = (block - block.mean()) / sigma
        elif method in ("minmax01", "minmax11"):
            low = block.min()
            span = (block.max() - low).replace(0.0, 1.0)
            unit = (block - low) / span
            scaled = unit if method == "minmax01" else unit * 2.0 - 1.0
        elif method == "center":
            scaled = block - block.mean()
        else:
            raise ValueError(
                f"Unknown method {method!r}; expected 'zscore', 'minmax01', 'minmax11', or 'center'."
            )
        return self._replace(data=scaled)

    def stratify_by(self, name: str) -> dict[object, "Dataset"]:
        """Split into one `Dataset` per distinct value of variable ``name``.

        Returns a dict keyed by the stratifying value (sorted). The
        stratifying variable is kept in each split (it is constant within a
        split and can be dropped later as a singleton).
        """
        if name not in self.variables:
            raise ValueError(f"Cannot stratify on unknown variable: {name}")

        splits: dict[object, "Dataset"] = {}
        for value, idx in self.data.groupby(name, sort=True).groups.items():
            subset = self.data.loc[idx].reset_index(drop=True)
            splits[value] = self._replace(data=subset)
        return splits

    def _latent_project(self, keep: Iterable[str]) -> "Dataset":
        """Canonical latent projection: keep the given variables, marginalize
        the rest.

        The source ADMG is expanded into its **canonical DAG** -- each source
        bidirected edge ``a <-> b`` is replaced by an unobserved confounder
        ``h`` with ``h -> a`` and ``h -> b`` -- and the standard DAG latent
        projection is run on that expanded DAG (all original non-kept
        variables and all introduced confounders are latent). This makes the
        construction equivalent to the canonical ADMG latent projection
        (Verma-Pearl / Richardson-Spirtes), so all paths through bidirected
        confounding are propagated correctly.

        The projected graph has:

        - directed edge ``u -> v`` between two kept variables iff a directed
          path ``u -> ... -> v`` exists in the source with every internal node
          latent;
        - bidirected edge ``u <-> v`` whenever the expanded DAG has a latent
          node that is a common ancestor of ``u`` and ``v`` via directed
          paths with all internal nodes latent. This covers latent common
          causes already in the source (preserved as ``u <-> v``), latent
          common causes induced by marginalization, AND confounding routed
          through chains that mix directed and bidirected edges.

        Single-edge-per-pair convention: if both a directed and a bidirected
        edge would be assigned to the same pair (e.g. ``u`` is an ancestor of
        ``v`` via a latent path AND ``u, v`` also share an unrelated latent
        common cause), only the directed edge is kept -- this representation
        does not model edge multiplicities. The collapsed case loses the
        "direct effect is also confounded" annotation. Limitation: a full ADMG
        would keep both (a "bow"); this only matters to code that uses the
        bidirected edges, which nothing does yet (scoring uses directed edges).

        Result is ``graph_type='admg'`` if any bidirected edges remain, else
        ``'dag'``.
        """
        # Dedupe while preserving first occurrence so duplicate inputs do not
        # produce duplicated data columns (which would trip the invariants).
        seen: set[str] = set()
        keep_list: list[str] = []
        for name in keep:
            if name in seen:
                continue
            seen.add(name)
            keep_list.append(name)
        unknown = [n for n in keep_list if n not in self.variables]
        if unknown:
            raise ValueError(f"Cannot keep unknown variables: {unknown}")
        keep_set = set(keep_list)

        # Build the canonical DAG of the source ADMG.
        canonical = self.directed_subgraph()
        existing_names: set[str] = set(canonical.nodes())
        for i, (a, b) in enumerate(self.bidirected_edges()):
            h = f"__bidir_{i}__"
            while h in existing_names:
                h += "_"
            existing_names.add(h)
            canonical.add_node(h)
            canonical.add_edge(h, a)
            canonical.add_edge(h, b)

        def observed_reach(start: str) -> set[str]:
            """Kept nodes reachable from ``start`` in the canonical DAG via
            paths whose internal nodes are all latent. ``start`` excluded."""
            out: set[str] = set()
            visited: set[str] = set()
            stack = list(canonical.successors(start))
            while stack:
                x = stack.pop()
                if x in visited:
                    continue
                visited.add(x)
                if x in keep_set:
                    out.add(x)  # observed; don't traverse through it
                else:
                    stack.extend(canonical.successors(x))
            return out

        new_graph = nx.DiGraph()
        new_graph.add_nodes_from(keep_list)
        for u in keep_list:
            for v in observed_reach(u):
                new_graph.add_edge(u, v)

        # Bidirected edges arise from every latent node (original latent or
        # synthetic confounder) whose observed-reach contains >=2 kept nodes.
        # We enforce a single-edge-per-pair convention: if a directed edge
        # already connects the pair (in either direction), the bidirected one
        # is dropped (no multi-edges in this representation). The directed
        # edge carries strictly more structural information; the trade-off is
        # that "direct effect plus separate latent confounding" collapses to
        # just the directed edge (see the docstring's limitation note).
        directed_pair_keys = {frozenset((u, v)) for u, v in new_graph.edges()}
        bidir: set[frozenset[str]] = set()
        for w in set(canonical.nodes()) - keep_set:
            for a, b in itertools.combinations(sorted(observed_reach(w)), 2):
                pair = frozenset((a, b))
                if pair in directed_pair_keys:
                    continue
                bidir.add(pair)
        for pair in bidir:
            a, b = sorted(pair)
            new_graph.add_edge(a, b, directed=False)

        new_graph_type: GraphType = "admg" if bidir else "dag"
        return Dataset(data=self.data.loc[:, keep_list], graph=new_graph, graph_type=new_graph_type)

    def subsample_variables(self, n: int, random_state: int | None = None) -> "Dataset":
        """Randomly keep ``n`` of the variables, treating the rest as latent.

        The dropped variables are marginalized via latent projection, so the
        ground-truth graph still matches the data (bidirected edges appear
        wherever marginalization induces latent confounding).
        """
        total = len(self.variables)
        if not 0 <= n <= total:
            raise ValueError(f"n must be in [0, {total}], got {n}")
        rng = np.random.default_rng(random_state)
        all_names = self.variables
        chosen = rng.choice(total, size=n, replace=False)
        keep = [all_names[i] for i in sorted(chosen)]
        return self._latent_project(keep)

    def hide_variables(self, names: Iterable[str]) -> "Dataset":
        """Remove the named variables from the data and marginalize them in
        the ground-truth graph via latent projection.

        The remaining variables and any induced bidirected edges from latent
        confounding form the new ADMG (or DAG, if no confounding is induced).
        """
        hide = list(names)
        unknown = [n for n in hide if n not in self.variables]
        if unknown:
            raise ValueError(f"Cannot hide unknown variables: {unknown}")
        keep = [n for n in self.variables if n not in set(hide)]
        return self._latent_project(keep)

    def drop_singletons(self, print_dropped: bool = False) -> "Dataset":
        """Drop all variables that take only a single value."""
        singletons = [c for c in self.data.columns if self.data[c].nunique(dropna=False) == 1]
        if print_dropped:
            print(f"Dropping singletons: {singletons}")
        return self.drop_columns(singletons)

    def random_sample(self, n: int | None = None, random_state: int | None = None) -> "Dataset":
        """Return a new `Dataset` with a random sample of `n` rows from this one.
        `random_state=None` means no shuffling, `n=None` means keep all rows."""
        if n is None and random_state is None:
            return self
        if random_state is None:
            return self._replace(data=self.data.head(n))
        frac = 1.0 if n is None else None
        sampled = self.data.sample(n=n, frac=frac, random_state=random_state).reset_index(drop=True)
        return self._replace(data=sampled)

    def fingerprint(self) -> str:
        """Short hash of the full contents (data incl. column and row order,
        graph), so results can record exactly which data they used. Built from
        raw float64 bytes, so it does not depend on the pandas version."""
        h = hashlib.sha256(repr([str(c) for c in self.data.columns]).encode())
        h.update(np.ascontiguousarray(self.data.index.to_numpy(dtype=np.int64)).tobytes())
        h.update(np.ascontiguousarray(self.data.to_numpy(dtype=np.float64)).tobytes())
        h.update(repr(sorted(self.graph.edges(data=True))).encode())
        h.update(self.graph_type.encode())
        return h.hexdigest()[:16]

    # ------------------------------------------------------------- internal

    def _replace(
        self,
        *,
        data: pd.DataFrame | None = None,
        graph: nx.DiGraph | None = None,
    ) -> "Dataset":
        """Return a new `Dataset` with selected components swapped.

        Preprocessing methods funnel through this so that invariants are
        re-checked uniformly. Also warns if the resulting graph has more than
        one weakly connected component.
        """
        new_ds = Dataset(
            data=self.data if data is None else data,
            graph=self.graph if graph is None else graph,
            graph_type=self.graph_type,
        )
        n_components = nx.number_weakly_connected_components(new_ds.graph)
        if n_components > 1:
            print(
                f"Warning: graph has {n_components} weakly connected components "
                f"after this operation."
            )
        return new_ds

    def _check_invariants(self) -> None:
        graph_nodes = {str(n) for n in self.graph.nodes()}
        columns = self.variables
        if len(set(columns)) != len(columns):
            raise ValueError("Data has duplicate column names.")
        if graph_nodes != set(columns):
            raise ValueError(
                "Data columns must match graph nodes. "
                f"Graph nodes missing from data: {sorted(graph_nodes - set(columns))[:10]} | "
                f"Columns missing from graph: {sorted(set(columns) - graph_nodes)[:10]}"
            )

        has_bidirected = bool(self.bidirected_edges())
        if self.graph_type == "dag":
            if has_bidirected:
                raise ValueError("graph_type='dag' but the graph contains bidirected edges.")
            if not nx.is_directed_acyclic_graph(self.graph):
                raise NotADAGError("Graph contains a cycle; must be a DAG.")
        else:  # admg: only the directed part must be acyclic
            if not has_bidirected:
                print(
                    "Warning: graph_type='admg' but the graph has no "
                    "bidirected edges; it is effectively a DAG."
                )
            if not nx.is_directed_acyclic_graph(self.directed_subgraph()):
                raise NotADAGError("ADMG directed edges contain a cycle.")

        nan_counts = self.data.isna().sum()
        nan_cols = nan_counts[nan_counts > 0]
        if not nan_cols.empty:
            preview = {c: int(n) for c, n in list(nan_cols.items())[:10]}
            print(f"Warning: data contains NaN values in {len(nan_cols)} column(s): " f"{preview}")


__all__ = [
    "DEFAULT_DROP_COLUMNS",
    "Dataset",
    "GraphType",
    "NotADAGError",
    "StandardizeMethod",
]
