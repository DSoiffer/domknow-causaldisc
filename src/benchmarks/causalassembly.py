# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This file uses the causalAssembly package (Copyright (c) 2023 Robert Bosch GmbH,
# AGPL-3.0) and is distributed under the GNU Affero General Public License, version
# 3 or later; see LICENSE-AGPL-3.0 at the repository root. It is a standalone
# data-preparation step: `python -m benchmarks causalassembly` writes a pickled
# `Dataset` that the experiment code reads, and nothing outside `benchmarks`
# imports this module.
"""Load the CausalAssembly assembly-line benchmark as a `Dataset`.

Semisynthetic, observational-only, all-continuous data from
boschresearch/causalAssembly -- a manufacturing assembly-line benchmark whose
ground-truth DAG is established from the physics of each station:

* **data**: ``ProductionLineGraph.get_data()`` -- 15581 x 98 continuous samples,
  drawn from per-node Distributional Random Forests fit to real Bosch
  assembly-line measurements (so the released data are guaranteed Markov to the
  published DAG; the raw recordings themselves are proprietary). Loading this
  shipped sample needs no R/rpy2 (that is only required to refit DRFs).
* **graph**: ``ProductionLineGraph.get_ground_truth()`` -- 98 nodes, 485 edges,
  a DAG over 5 stations (``Station1``..``Station5``). Only the structure is used.

All 98 variables are continuous and the data columns already match the graph
node names one-to-one, so the result is an ordinary `Dataset` that flows through
the experiments code unchanged.
Unlike CausalMan, no node is a deterministic function of its parents; but the
mechanisms are nonlinear/nonparametric, so the linear NOTEARS/DAGMA configs are
genuinely misspecified against it.

Needs the optional dependency: ``uv sync --extra causalassembly``.
"""

from __future__ import annotations

import networkx as nx

from dataset import Dataset


def load_causalassembly() -> Dataset:
    """Return the CausalAssembly benchmark as a `Dataset` (all continuous).

    The `causalAssembly` import is local so the other benchmarks build without
    the optional dependency installed.
    """
    from causalAssembly.models_dag import ProductionLineGraph

    data = ProductionLineGraph.get_data().reset_index(drop=True)
    truth = ProductionLineGraph.get_ground_truth()
    graph_raw = truth.graph if hasattr(truth, "graph") else truth
    graph = graph_raw if isinstance(graph_raw, nx.DiGraph) else nx.DiGraph(graph_raw)

    node_names = [str(n) for n in graph.nodes()]
    csv_cols = set(data.columns)
    graph_nodes = set(node_names)
    if csv_cols != graph_nodes:
        missing_in_csv = sorted(graph_nodes - csv_cols)
        extra_in_csv = sorted(csv_cols - graph_nodes)
        raise ValueError(
            "CausalAssembly data columns and graph nodes must match exactly. "
            f"Graph nodes missing from data: {missing_in_csv[:10]} | "
            f"Data columns missing from graph: {extra_in_csv[:10]}"
        )

    return Dataset(data=data[node_names], graph=graph, graph_type="dag")
