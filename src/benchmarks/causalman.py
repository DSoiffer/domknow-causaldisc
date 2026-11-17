"""Load the simplified CausalMan benchmark as a `Dataset`.

Source: the CausalMan "dataset0" (small production line, fully observed) from
https://zenodo.org/records/14697668, unpacked into ``data/dataset0/``. The raw
data mixes several product configurations and contains many categorical,
constant, and pass/fail-check variables; this benchmark simplifies it to a single
continuous sub-system:

1. keep one product configuration (product type, pressing section, supplier),
   which fixes the categorical variables, then drop the now-constant columns;
2. keep only the ancestors of ``Sec_C2_Machine1_ProcessResult`` (the press-fit
   station);
3. drop tolerance limits (``_LTL``/``_UTL``) and pass/fail indicators
   (``_MpGood``, ``ProcessResult``), which are deterministic checks.

The result has 24 continuous variables and 22 edges. Many of its structural
equations are deterministic; the experiments add noise by resampling from a
fitted SCM (``data_generator.noisy_like``, via ``benchmarks.load``).
"""

from __future__ import annotations

import networkx as nx

from benchmarks import REPO_ROOT
from dataset import Dataset

DATASET0_DIR = REPO_ROOT / "data" / "dataset0"
TARGET = "Sec_C2_Machine1_ProcessResult"


def load_causalman() -> Dataset:
    """Return the simplified CausalMan benchmark (all continuous)."""
    ds = Dataset.from_files(
        DATASET0_DIR / "dataset0_fully_observable.csv", DATASET0_DIR / "ground_truth.gml"
    )
    ds = ds.random_sample(n=None, random_state=1)  # shuffle rows

    ds = ds.stratify_by("product_type_ID_num")[92122]
    ds = ds.stratify_by("PSec_C2_pSection_ID_num")[722]
    ds = ds.stratify_by("HU_HU_Block_Supplier_ID_num")[13]
    ds = ds.drop_singletons()

    keep = nx.ancestors(ds.graph, TARGET) | {TARGET}
    ds = ds.drop_columns([v for v in ds.variables if v not in keep])

    checks = ("_MpGood", "ProcessResult", "_LTL", "_UTL")
    return ds.drop_columns([v for v in ds.variables if v.endswith(checks)])
