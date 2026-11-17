"""Load the Sachs et al. (2005) protein-signaling benchmark as a `Dataset`.

Real, observational-only, continuous data -- the standard single-cell flow
cytometry benchmark, with interventions excluded:

* **data**: bnlearn's ``sachs.data.txt`` -- 853 purely *observational* continuous
  samples (the general anti-CD3/CD28 stimulus condition; the 8 interventional
  conditions are not included). No Python causal-discovery library exposes this
  file through a loader API (cdt only ships the pooled 7466-row set, bnlearn's
  ``import_example`` returns *synthetic* samples, pywhy/gCastle ship no Sachs
  loader at all), so it is fetched with `pooch` -- the standard scientific-data
  fetcher (SciPy / scikit-image / pywhy use it) -- which caches the download and
  verifies it against a pinned hash.
* **graph**: ``bnlearn.import_DAG('sachs')`` -- the consensus network from the
  paper: 11 nodes, 17 directed edges, a DAG. Only the structure is used.

Data columns and graph nodes share the same 11 variable names (Raf, Mek, Plcg,
PIP2, PIP3, Erk, Akt, PKA, PKC, P38, Jnk), so no remapping is needed; all are
continuous. The result is an ordinary `Dataset`, so it flows through the
experiments / DataGenerator code unchanged.
"""

from __future__ import annotations

import gzip
import logging

import networkx as nx
import pandas as pd
import pooch

from benchmarks import DATASETS_DIR
from dataset import Dataset

DATA_URL = "https://www.bnlearn.com/book-crc/code/sachs.data.txt.gz"
# sha256 of the gzip served by bnlearn.com (pin for integrity + reproducibility).
DATA_HASH = "sha256:3b65a3c4fe642d8f9f9562fe798edbafd5aaefe80caecd7f9ec5dea2fbe86d03"


def _load_data() -> pd.DataFrame:
    """Fetch (cached + hash-verified) and parse the 853x11 observational data."""
    path = pooch.retrieve(DATA_URL, known_hash=DATA_HASH, path=DATASETS_DIR)
    with gzip.open(path, "rt") as f:
        return pd.read_csv(f, sep=r"\s+")


def _load_graph(nodes: list[str]) -> nx.DiGraph:
    """Consensus 17-edge DAG from ``bnlearn.import_DAG`` over the given nodes."""
    import bnlearn as bn

    # bnlearn/pgmpy are chatty; keep the import quiet. Only the structure is used.
    logging.getLogger("pgmpy").setLevel(logging.ERROR)
    dag = bn.import_DAG("sachs", CPD=False)

    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)  # fix node order to the data's column order
    graph.add_edges_from((str(u), str(v)) for u, v in dag["model"].edges())

    if set(graph.nodes()) != set(nodes):
        raise ValueError(
            f"Sachs graph nodes {sorted(graph.nodes())} do not match data "
            f"columns {sorted(nodes)}."
        )
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("bnlearn Sachs ground truth is not a DAG.")
    return graph


def load_sachs() -> Dataset:
    """Return the observational Sachs benchmark as a `Dataset` (all continuous)."""
    data = _load_data()
    nodes = [str(c) for c in data.columns]
    return Dataset(data=data, graph=_load_graph(nodes))
