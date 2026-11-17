"""Load the experiment protocol from ``config.toml`` (learners, benchmarks,
experiment grids). Unknown keys are rejected; the string ``"none"`` is read as
``None`` wherever a value can be absent.

    from train import config
    cfg = config.load()                 # the shipped config.toml
    cfg = config.load("my_config.toml")  # or another file with the same layout
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

CONFIG_PATH = Path(__file__).with_name("config.toml")

# A learner's identity fields, in result-column order.
LEARNER_KEYS = ["base", "method", "model", "loss_fn", "optimizer",
                "reg", "gamma", "lambda1", "normalization", "var_floor"]
_TOP_KEYS = {"w_thresholds", "seeds", "noise_seed", "propagate_noise", "n_samples",
             "learner_defaults", "learners", "config_sets", "method_options",
             "optimizer_options", "benchmarks", "experiments"}
_BENCHMARK_KEYS = {"noise", "layout_dataset", "cpdag_metrics"}


@dataclass(frozen=True)
class Benchmark:
    noise: list[float | None]
    layout_dataset: str | None = None
    cpdag_metrics: bool = False
    axes: dict[str, dict[str, list]] | None = None  # experiment -> extra axes


@dataclass(frozen=True)
class Config:
    w_thresholds: list[float]
    seeds: list[int]
    noise_seed: int
    propagate_noise: bool
    n_samples: int | None
    learners: dict[str, dict]  # name -> identity fields (LEARNER_KEYS order)
    config_sets: dict[str, list[str]]
    method_options: dict[str, dict]
    optimizer_options: dict[str, dict]
    benchmarks: dict[str, Benchmark]
    experiments: dict[str, dict]  # name -> {axis: values}


def _none(value):
    if isinstance(value, list):
        return [_none(v) for v in value]
    return None if value == "none" else value


def _check(keys, allowed, where: str) -> None:
    unknown = sorted(set(keys) - set(allowed))
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {unknown}")


def load(path: str | Path = CONFIG_PATH) -> Config:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    _check(raw, _TOP_KEYS, str(path))

    defaults = raw["learner_defaults"]
    _check(defaults, LEARNER_KEYS, "learner_defaults")
    learners = {}
    for name, fields in raw["learners"].items():
        _check(fields, LEARNER_KEYS, f"learners.{name}")
        merged = {"base": name, **defaults, **fields}
        missing = [k for k in LEARNER_KEYS if k not in merged]
        if missing:
            raise ValueError(f"learners.{name}: missing {missing}")
        learners[name] = {k: merged[k] for k in LEARNER_KEYS}
    for set_name, names in raw["config_sets"].items():
        _check(names, learners, f"config_sets.{set_name}")

    experiments = {name: {axis: _none(v) for axis, v in axes.items()}
                   for name, axes in raw["experiments"].items()}
    benchmarks = {}
    for name, table in raw["benchmarks"].items():
        _check(table, _BENCHMARK_KEYS | set(experiments), f"benchmarks.{name}")
        benchmarks[name] = Benchmark(
            noise=_none(table["noise"]),
            layout_dataset=table.get("layout_dataset"),
            cpdag_metrics=table.get("cpdag_metrics", False),
            axes={exp: {axis: _none(v) for axis, v in table[exp].items()}
                  for exp in experiments if exp in table})

    return Config(
        w_thresholds=raw["w_thresholds"], seeds=raw["seeds"], noise_seed=raw["noise_seed"],
        propagate_noise=raw["propagate_noise"], n_samples=_none(raw["n_samples"]),
        learners=learners, config_sets=raw["config_sets"],
        method_options=raw["method_options"], optimizer_options=raw["optimizer_options"],
        benchmarks=benchmarks, experiments=experiments)
