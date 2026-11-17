"""The CausalMan reference layout: node coordinates and styles read from the
vis-network HTML export shipped with the Zenodo dataset.

Training uses it for the ``"reference"`` tier mask (a node's tier is its height
in the drawing); ``visualize.graph`` uses it to place nodes.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, TypedDict

from benchmarks import REPO_ROOT

OBSERVABILITY_CHOICES = ("full", "partial")


class ReferenceVisualSpec(TypedDict):
    positions: dict[str, tuple[float, float]]
    node_styles: dict[str, dict[str, Any]]
    network_width: str
    network_height: str


def dataset_dir(dataset_name: str) -> Path:
    path = REPO_ROOT / "data" / dataset_name
    if not path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {path}")
    return path


def observability_paths(dataset_path: Path, observability: str) -> dict[str, Path]:
    """The ground-truth graph, data CSV and reference layout HTML of a dataset
    directory, for one observability mode."""
    name = dataset_path.name
    if observability == "full":
        return {
            "gml": dataset_path / "ground_truth.gml",
            "csv": dataset_path / f"{name}_fully_observable.csv",
            "default_reference": dataset_path / f"ground_truth_{name}.gml_complete.html",
        }
    if observability == "partial":
        return {
            "gml": dataset_path / "ground_truth_partially_observable.gml",
            "csv": dataset_path / f"{name}_partially_observable.csv",
            "default_reference": (
                dataset_path / f"ground_truth_partially_observable_{name}.gml_complete.html"
            ),
        }
    raise ValueError(f"observability must be one of {OBSERVABILITY_CHOICES}, got {observability!r}")


def load_reference_visual_spec(reference_html_path: Path) -> ReferenceVisualSpec:
    """Extract node positions and visual style from a vis-network HTML export."""
    if not reference_html_path.exists():
        raise FileNotFoundError(f"Reference HTML not found: {reference_html_path}")

    html = reference_html_path.read_text(encoding="utf-8")
    match = re.search(r"nodes\s*=\s*new\s+vis\.DataSet\((\[.*?\])\);", html, re.DOTALL)
    if match is None:
        raise ValueError(
            "Could not parse node DataSet from reference HTML. "
            "Expected a vis-network export containing 'nodes = new vis.DataSet([...]);'."
        )

    try:
        nodes_data = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise ValueError("Failed to parse node JSON from reference HTML.") from exc

    positions: dict[str, tuple[float, float]] = {}
    node_styles: dict[str, dict] = {}
    copied_style_keys = {
        "color",
        "size",
        "shape",
        "borderWidth",
        "borderWidthSelected",
        "value",
        "font",
        "physics",
    }
    for node in nodes_data:
        node_id = str(node.get("id"))
        if "x" not in node or "y" not in node:
            raise ValueError(f"Reference node '{node_id}' does not contain x/y coordinates.")
        positions[node_id] = (float(node["x"]), float(node["y"]))
        node_styles[node_id] = {key: node[key] for key in copied_style_keys if key in node}

    network_width = "100%"
    network_height = "1200px"
    container_block = re.search(r"#mynetwork\s*\{(.*?)\}", html, re.DOTALL)
    if container_block is not None:
        css_body = container_block.group(1)
        width_match = re.search(r"width\s*:\s*([^;]+);", css_body)
        height_match = re.search(r"height\s*:\s*([^;]+);", css_body)
        if width_match:
            network_width = width_match.group(1).strip()
        if height_match:
            network_height = height_match.group(1).strip()

    return {
        "positions": positions,
        "node_styles": node_styles,
        "network_width": network_width,
        "network_height": network_height,
    }


def resolve_reference_html_path(raw_reference_path: str, ds_dir: Path) -> Path:
    """Resolve a reference path relative to dataset dir first, then repo root."""
    ref_path = Path(raw_reference_path)
    if ref_path.is_absolute():
        return ref_path

    candidate_in_dataset = ds_dir / ref_path
    if candidate_in_dataset.exists():
        return candidate_in_dataset

    candidate_in_repo = REPO_ROOT / ref_path
    if candidate_in_repo.exists():
        return candidate_in_repo

    return ref_path


def reference_spec(
    dataset: str = "dataset0",
    observability: str = "full",
    reference_html: str | Path | None = None,
) -> ReferenceVisualSpec:
    """The reference layout of ``data/<dataset>/`` (its default export for the
    observability mode unless ``reference_html`` names another one)."""
    ds_dir = dataset_dir(dataset)
    if reference_html is None:
        reference_html = observability_paths(ds_dir, observability)["default_reference"]
    return load_reference_visual_spec(resolve_reference_html_path(str(reference_html), ds_dir))
