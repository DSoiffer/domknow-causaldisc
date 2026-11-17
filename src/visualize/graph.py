"""Interactive (PyVis) renderings of graphs: a benchmark's ground truth, a
`Dataset`'s graph, or one trained run's estimate against the truth.

Usage:
    python -m visualize.graph benchmark <name> [--out FILE]    # a built benchmark's true graph
    python -m visualize.graph truth [--dataset dataset0]       # ground truth -> data/<dataset>/
    python -m visualize.graph run <exp_dir> <run_id> [--threshold T] [--out FILE]

The pages load vis-network from a CDN. In a notebook, `inline(path)` shows one.

CausalMan's reference layout (``benchmarks.layout``) is used for CausalMan-derived graphs,
including the original CausalMan graphs, and topological layers are used for other graphs.

Colours: nodes laid out in topological layers shade from blue (roots) to green
(deepest layer); nodes in the CausalMan layout keep the reference's colours (all
blue but the red process result). Ground-truth edges are grey-blue, bidirected
ones dashed purple. A run's estimate: true positives grey, false positives red,
missed true edges dashed orange.
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from pyvis.network import Network

import benchmarks
from benchmarks.layout import (
    OBSERVABILITY_CHOICES,
    ReferenceVisualSpec,
    dataset_dir,
    observability_paths,
    reference_spec,
)
from dataset import Dataset
from train import store


def node_group_key(node_id: str) -> str:
    """Group related variables so each level stays visually coherent."""
    patterns = [
        r"^(PF_M\d+_T\d+)",
        r"^(MV\d+)",
        r"^(HU_Ch\d+)",
        r"^(HU_T\d+)",
        r"^(PSec_[^_]+)",
        r"^(Sec_[^_]+)",
        r"^(T\d+)",
    ]
    for pattern in patterns:
        match = re.match(pattern, node_id)
        if match:
            return match.group(1)
    return node_id.split("_", 1)[0]


def compact_layout_positions(
    graph: nx.DiGraph,
    level_gap: float = 155.0,
    node_gap: float = 65.0,
    group_gap: float = 95.0,
) -> tuple[dict[str, tuple[float, float]], dict[str, int]]:
    """Create coordinates using topological layers."""
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Expected a DAG for hierarchical coordinate generation.")

    generations = list(nx.topological_generations(graph))
    level_by_node: dict[str, int] = {}
    for level, generation in enumerate(generations):
        for node in generation:
            level_by_node[str(node)] = level

    group_anchor_level: dict[str, int] = {}
    for level, generation in enumerate(generations):
        for node in generation:
            group = node_group_key(str(node))
            if group not in group_anchor_level:
                group_anchor_level[group] = level

    positions: dict[str, tuple[float, float]] = {}
    for level, generation in enumerate(generations):
        group_to_nodes: dict[str, list[str]] = {}
        for node in generation:
            node_id = str(node)
            group_to_nodes.setdefault(node_group_key(node_id), []).append(node_id)

        ordered_groups = sorted(group_to_nodes, key=lambda g: (group_anchor_level[g], g))

        x = 0.0
        level_nodes: list[str] = []
        for group in ordered_groups:
            group_nodes = sorted(
                group_to_nodes[group],
                key=lambda n: (graph.in_degree(n), graph.out_degree(n), n),
            )
            level_nodes.extend(group_nodes)
            for node_id in group_nodes:
                positions[node_id] = (x, -level * level_gap)
                x += node_gap
            x += group_gap

        if level_nodes:
            min_x = min(positions[node_id][0] for node_id in level_nodes)
            max_x = max(positions[node_id][0] for node_id in level_nodes)
            shift = (min_x + max_x) / 2.0
            for node_id in level_nodes:
                old_x, old_y = positions[node_id]
                positions[node_id] = (old_x - shift, old_y)

    return positions, level_by_node


def validate_reference_node_set(
    graph: nx.DiGraph,
    reference_positions: dict[str, tuple[float, float]],
) -> None:
    """Ensure reference positions cover exactly the graph node names."""
    graph_nodes = {str(node) for node in graph.nodes()}
    ref_nodes = set(reference_positions)
    if graph_nodes != ref_nodes:
        missing_in_reference = sorted(graph_nodes - ref_nodes)
        extra_in_reference = sorted(ref_nodes - graph_nodes)
        raise ValueError(
            "Reference node set does not match graph node set. "
            f"Missing in reference: {missing_in_reference[:10]} | "
            f"Extra in reference: {extra_in_reference[:10]}"
        )


def render_pyvis_graph(
    graph: nx.DiGraph,
    output_html: Path,
    reference_spec: ReferenceVisualSpec | None = None,
    edge_styles: dict[tuple[str, str], dict] | None = None,
) -> None:
    """Render a compact, fixed-position PyVis graph."""
    if reference_spec is not None:
        reference_positions = reference_spec["positions"]
        reference_node_styles = reference_spec["node_styles"]
        validate_reference_node_set(graph, reference_positions)

        positions = reference_positions
        # Layer is only used for color; when using reference coordinates, infer a
        # pseudo-layer from y so the color gradient remains informative.
        ys = sorted({pos[1] for pos in positions.values()})
        y_to_level = {y: idx for idx, y in enumerate(ys)}
        levels = {node_id: y_to_level[positions[node_id][1]] for node_id in positions}
        network_height = reference_spec["network_height"]
        network_width = reference_spec["network_width"]
    else:
        positions, levels = compact_layout_positions(graph)
        reference_node_styles = {}
        network_height = "1200px"
        network_width = "100%"

    max_level = max(levels.values()) if levels else 1

    net = Network(
        height=network_height,
        width=network_width,
        directed=True,
        bgcolor="#ffffff",
        font_color="#222222",
        cdn_resources="remote",
    )

    for node in graph.nodes():
        node_id = str(node)
        level = levels[node_id]
        node_style = reference_node_styles.get(node_id, {})
        hue = int(200 - 90 * (level / max_level))
        color = node_style.get("color", f"hsl({hue}, 75%, 52%)")
        x, y = positions[node_id]
        net.add_node(
            node_id,
            label=node_id,
            x=x,
            y=y,
            fixed=True,
            physics=node_style.get("physics", False),
            shape=node_style.get("shape", "dot"),
            size=node_style.get("size", 8),
            borderWidth=node_style.get("borderWidth", 2),
            color=color,
            # Only when set: a null value stops vis-network drawing the nodes.
            **{k: node_style[k] for k in ("value", "font") if k in node_style},
            title=(
                f"{node_id}<br>"
                f"layer={level}<br>"
                f"in_degree={graph.in_degree(node)}<br>"
                f"out_degree={graph.out_degree(node)}"
            ),
        )

    for source, target in graph.edges():
        style = {"arrows": "to", "width": 1, "color": "#90a4ae"}
        if edge_styles:
            style.update(edge_styles.get((str(source), str(target)), {}))
        net.add_edge(str(source), str(target), **style)

    net.set_options(
        """
        {
          "configure": {
            "enabled": false
          },
          "interaction": {
            "hover": true,
            "dragNodes": false,
            "hideEdgesOnDrag": true,
            "zoomView": true,
            "dragView": true,
            "navigationButtons": true
          },
          "physics": {
            "enabled": false
          },
          "nodes": {
            "font": {
              "size": 9,
              "face": "Verdana"
            }
          },
          "edges": {
                        "smooth": false,
            "color": {
              "inherit": false
            }
          }
        }
        """
    )

    output_html.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(output_html), open_browser=False, notebook=False)
    print(f"Graph rendered to {output_html}")


def _layout(names: list[str], B_true: np.ndarray) -> ReferenceVisualSpec:
    """Node placement for a graph over ``names``: the CausalMan reference layout
    if it places every node, else the topological layers of the true DAG
    ``B_true`` (an estimate drawn on top may have cycles, so it can't be layered
    itself)."""
    try:
        ref = reference_spec()
    except FileNotFoundError:
        ref = None
    if ref is not None and set(names) <= set(ref["positions"]):
        return {**ref, "positions": {n: ref["positions"][n] for n in names},
                "node_styles": {n: ref["node_styles"][n] for n in names if n in ref["node_styles"]}}
    truth = nx.DiGraph()
    truth.add_nodes_from(names)
    truth.add_edges_from((names[i], names[j]) for i, j in zip(*np.nonzero(B_true)))
    positions, _ = compact_layout_positions(truth)
    return {"positions": positions, "node_styles": {}, "network_width": "100%",
            "network_height": "1200px"}


# ---- Learned graph vs ground truth --------------------------------------------

def render_learned_graph(
    names: list[str],
    B_true: np.ndarray,
    B_est: np.ndarray,
    output_html: Path,
    reference_spec: ReferenceVisualSpec | None = None,
) -> tuple[int, int, int]:
    """Render an estimated graph against the ground truth (both (d, d), indexed
    by ``names``). Returns (n_tp, n_fp, n_fn).

    - True positive edges (in both learned and ground truth): default grey
    - False positive edges (in learned, not ground truth): red solid
    - False negative edges (in ground truth, not learned): orange dashed
    """
    truth, est = np.asarray(B_true) != 0, np.asarray(B_est) != 0
    comparison_graph = nx.DiGraph()
    comparison_graph.add_nodes_from(names)
    edge_styles: dict[tuple[str, str], dict] = {}
    for i, j in zip(*np.nonzero(truth | est)):
        src, dst = names[i], names[j]
        comparison_graph.add_edge(src, dst)
        if not truth[i, j]:  # FP
            edge_styles[(src, dst)] = {"color": "#e53935"}
        elif not est[i, j]:  # FN
            edge_styles[(src, dst)] = {"color": "#ffA500", "dashes": True}

    render_pyvis_graph(
        comparison_graph, output_html, reference_spec=reference_spec, edge_styles=edge_styles
    )
    return int((truth & est).sum()), int((~truth & est).sum()), int((truth & ~est).sum())


def best_shd_threshold(exp_dir: Path, rid: str) -> float:
    """The run's w_threshold with the lowest shd (ties: the smallest threshold)."""
    metrics = pd.read_csv(store.run_dir(exp_dir, rid) / "metrics.csv")
    return float(metrics.sort_values(["shd", "w_threshold"]).iloc[0]["w_threshold"])


def render_run(exp_dir: Path, rid: str, threshold: float | None = None,
               output_html: Path | None = None) -> Path:
    """Render one stored run's estimate, thresholded at ``threshold`` (default:
    its best-shd threshold), against the ground truth. Writes
    ``<run dir>/graph_t<threshold>.html`` unless ``output_html`` is given."""
    z = store.load_graph(exp_dir, rid)
    if z is None:
        raise SystemExit(f"run {rid} in {exp_dir} has no saved graph (failed run?)")
    if threshold is None:
        threshold = best_shd_threshold(exp_dir, rid)
    names = [str(n) for n in z["variables"]]
    W = z["W"]
    B_est = W >= threshold if threshold > 0 else W != 0
    out = output_html or store.run_dir(exp_dir, rid) / f"graph_t{threshold:g}.html"
    n_tp, n_fp, n_fn = render_learned_graph(names, z["B_true"], B_est, out,
                                            _layout(names, z["B_true"]))
    print(f"threshold {threshold:g}: TP={n_tp}, FP={n_fp}, FN={n_fn}")
    return out


# ---- A Dataset's graph ----------------------------------------------------------

def render_dataset(ds: Dataset, output_html: str | Path) -> Path:
    """Render ``ds``'s ground-truth graph (placed as in `_layout`; bidirected
    edges dashed purple)."""
    names = ds.variables
    B = nx.to_numpy_array(ds.directed_subgraph(), nodelist=names)
    edge_styles = {(str(u), str(v)): {"color": "#8e24aa", "dashes": True, "arrows": "to;from"}
                   for u, v in ds.bidirected_edges()}
    render_pyvis_graph(ds.graph, Path(output_html), reference_spec=_layout(names, B),
                       edge_styles=edge_styles or None)
    return Path(output_html)


def render_benchmark(name: str, output_html: str | Path | None = None) -> Path:
    """Render a built benchmark's true graph (``python -m benchmarks`` first),
    by default to ``data/datasets/<name>_graph.html``."""
    return render_dataset(benchmarks.load(name),
                          output_html or benchmarks.DATASETS_DIR / f"{name}_graph.html")


def inline(path: str | Path, height: int = 1230):
    """Show a rendered graph page inside a Jupyter notebook (pages are 1200px
    tall; zoom and pan with the mouse or the corner buttons)."""
    from IPython.display import HTML

    page = html.escape(Path(path).read_text(encoding="utf-8"))
    return HTML(f'<iframe srcdoc="{page}" style="width:100%;height:{height}px;border:0"></iframe>')


# ---- CLI ------------------------------------------------------------------------

def render_ground_truth(dataset: str, observability: str, reference_html: str | None) -> None:
    """Render ``data/<dataset>/``'s ground-truth graph(s) next to the data."""
    ds_dir = dataset_dir(dataset)
    modes = OBSERVABILITY_CHOICES if observability == "both" else (observability,)
    for mode in modes:
        paths = observability_paths(ds_dir, mode)
        if not paths["gml"].exists():
            if observability == "both":
                print(f"[{mode}] Skipping: {paths['gml']} not found")
                continue
            raise FileNotFoundError(f"Missing graph file: {paths['gml']}")
        graph = nx.read_graphml(paths["gml"])
        graph = graph if isinstance(graph, nx.DiGraph) else nx.DiGraph(graph)
        spec = None
        if reference_html is not None or paths["default_reference"].exists():
            spec = reference_spec(dataset, mode, reference_html)
        out = paths["gml"].with_name(paths["gml"].stem + "_pyvis_structured.html")
        render_pyvis_graph(graph, out, reference_spec=spec)
        print(f"[{mode}] {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    bench = sub.add_parser("benchmark", help="a built benchmark's true graph")
    bench.add_argument("name", help="benchmark name, as in `python -m benchmarks`")
    bench.add_argument("--out", type=Path, default=None)
    truth = sub.add_parser("truth", help="a dataset directory's ground-truth graph")
    truth.add_argument("--dataset", default="dataset0", help="dataset folder under data/")
    truth.add_argument("-r", "--reference-html", default=None,
                       help="vis-network HTML giving node coordinates/styles "
                            "(default: the dataset's own export, if present)")
    truth.add_argument("--observability", choices=(*OBSERVABILITY_CHOICES, "both"), default="both")
    one = sub.add_parser("run", help="one trained run's estimate vs the ground truth")
    one.add_argument("exp_dir", type=Path)
    one.add_argument("run_id")
    one.add_argument("--threshold", type=float, default=None,
                     help="w_threshold to draw the estimate at (default: the run's best-shd one)")
    one.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.command == "benchmark":
        render_benchmark(args.name, args.out)
    elif args.command == "truth":
        render_ground_truth(args.dataset, args.observability, args.reference_html)
    else:
        render_run(args.exp_dir, args.run_id, args.threshold, args.out)


if __name__ == "__main__":
    main()
