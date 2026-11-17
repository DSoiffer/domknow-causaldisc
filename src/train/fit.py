"""Train one run: load its data, apply the experiment's domain knowledge, fit
with dagrad once (no thresholding) and score every w_threshold post hoc."""

from __future__ import annotations

import time

import networkx as nx
import numpy as np
import pandas as pd
import torch

import benchmarks
from dagrad import dagrad
from train.config import LEARNER_KEYS
from train.experiments import PREPARE
from train.knowledge import given_matrix
from train.metrics import edge_magnitudes, threshold_metric_rows


def load_data(cfg: dict):
    """The run's dataset (``cfg``: its identity and settings)."""
    return benchmarks.load(cfg["benchmark"], cfg["noise"], cfg["noise_seed"], cfg["propagate_noise"])


def describe(identity: dict) -> str:
    rest = (f"{k}={v}" for k, v in identity.items() if k not in ("experiment", *LEARNER_KEYS))
    return f"{identity['base']}/{identity['loss_fn']} " + " ".join(rest)


def fit(identity: dict, settings: dict) -> tuple[pd.DataFrame, dict, float]:
    """(metrics per w_threshold, graph arrays to store, training seconds)."""
    cfg = {**identity, **settings}
    ds = load_data(cfg)
    if ds.fingerprint() != cfg["data"]:
        raise RuntimeError("dataset changed since this run was defined; re-run the command")
    if cfg["n_samples"] is not None and cfg["n_samples"] < len(ds.data):
        ds = ds.random_sample(cfg["n_samples"], random_state=cfg["dk_seed"])

    prep = PREPARE[cfg["experiment"]](ds, cfg, np.random.default_rng(cfg["dk_seed"]))
    ds = prep.ds.standardize(cfg["normalization"])
    X = ds.data.to_numpy(dtype=float)
    names = ds.variables
    B_true = nx.to_pandas_adjacency(ds.directed_subgraph(), nodelist=names, dtype=int).to_numpy()
    given = given_matrix(names, prep.given_edges)
    options = {"gamma": cfg["gamma"], "var_floor": cfg["var_floor"], "w_threshold": 0.0,
               "lambda1": cfg["lambda1"], **prep.options}
    if prep.mask is not None:
        options["mask"] = prep.mask

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{describe(identity)} device={device} d={len(names)} given={int(given.sum())}")
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    start = time.time()
    W_raw = np.asarray(dagrad(
        X, method=cfg["method"], model=cfg["model"], loss_fn=cfg["loss_fn"],
        optimizer=cfg["optimizer"], reg=cfg["reg"], device=device,
        general_options=options,
        method_options=dict(cfg["method_options"]) | {"verbose": True},
        optimizer_options=dict(cfg["optimizer_options"]),
    ))
    seconds = round(time.time() - start, 3)

    metrics = pd.DataFrame(threshold_metric_rows(
        W_raw, cfg["w_thresholds"], B_true, given, include_cpdag=cfg["cpdag_metrics"]))
    graph = {"W": edge_magnitudes(W_raw), "variables": np.array(names),
             "B_true": B_true, "given": given}
    return metrics, graph, seconds
