"""Build benchmark pickles into ``data/datasets/``.

    python -m benchmarks [causalman] [causalassembly] [sachs]
                         [causalman_sim] [causalman_sim_pooled]

With no names, builds the first three. The ``causalman_sim*`` ones need the CausalMan
simulator, which is not a project dependency (see README)::

    uv run --with-editable ./CausalMan python -m benchmarks causalman_sim causalman_sim_pooled
"""

from __future__ import annotations

import argparse
import pickle

from benchmarks import DATASETS_DIR, pickle_path
from benchmarks.causalassembly import load_causalassembly
from benchmarks.causalman import load_causalman
from benchmarks.sachs import load_sachs



def load_causalman_sim():
    from benchmarks.causalman_sim import load_causalman_sim  # needs the simulator
    return load_causalman_sim()


def load_causalman_sim_pooled():
    from benchmarks.causalman_sim import load_causalman_sim_pooled
    return load_causalman_sim_pooled()


LOADERS = {
    "causalman": load_causalman,
    "causalassembly": load_causalassembly,
    "sachs": load_sachs,
    "causalman_sim": load_causalman_sim,
    "causalman_sim_pooled": load_causalman_sim_pooled,
}
DEFAULT = ["causalman", "causalassembly", "sachs"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", choices=list(LOADERS))
    args = parser.parse_args()

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    for name in args.names or DEFAULT:
        ds = LOADERS[name]()
        with open(pickle_path(name), "wb") as f:
            pickle.dump(ds, f)
        print(f"{name}: {ds.data.shape[0]} rows x {len(ds.variables)} variables, "
              f"{ds.directed_subgraph().number_of_edges()} edges -> {pickle_path(name)}")


if __name__ == "__main__":
    main()
