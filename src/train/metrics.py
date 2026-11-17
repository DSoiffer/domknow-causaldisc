"""Graph-recovery metrics, used by the experiment worker and the figure scripts.

When some edges are *given* as domain knowledge (``E_given``), recovery is
scored on what was left to discover: ``E_unknown = E* \\ E_given``. The plain
metric names (``fdr``, ``tpr``, ..., ``skel_*``) always refer to E_unknown;
``full_*`` versions score the whole graph. With nothing given the two coincide.

``cpdag_*`` columns (opt-in) score the *Markov equivalence class*: both the
true DAG and the (thresholded) estimate are converted to their CPDAGs and
compared per unordered pair. The metric definitions are exactly
``count_accuracy`` generalized to CPDAGs, so they reduce to ``full_*`` when both
graphs are fully directed and give ``cpdag_shd == 0`` iff the estimate is
Markov-equivalent to the truth. They always score the whole graph.
"""

from __future__ import annotations

import numpy as np

from dagrad import count_accuracy, is_dag

from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.utils.DAG2CPDAG import dag2cpdag

_METRIC_KEYS = ("fdr", "tpr", "fpr", "shd", "nnz")
_CPDAG_KEYS = ("cpdag_fdr", "cpdag_tpr", "cpdag_fpr", "cpdag_shd", "cpdag_nnz")


def dag_to_cpdag_marks(B: np.ndarray) -> np.ndarray:
    """CPDAG of a 0/1 DAG adjacency, in causallearn endpoint-mark encoding.

    ``out[i, j] == -1 and out[j, i] == 1`` marks a directed ``i -> j``;
    ``out[i, j] == out[j, i] == -1`` marks an undirected ``i -- j``.
    """
    B = np.asarray(B) != 0
    nodes = [GraphNode(str(i)) for i in range(B.shape[0])]
    g = GeneralGraph(nodes)
    for i, j in zip(*np.nonzero(B)):
        g.add_directed_edge(nodes[i], nodes[j])
    return dag2cpdag(g).graph


def _pair_marks(marks: np.ndarray) -> dict[tuple[int, int], str]:
    """{(i, j) with i<j: 'F' (i->j) | 'R' (j->i) | 'U' (i--j)} from CPDAG marks."""
    out: dict[tuple[int, int], str] = {}
    d = marks.shape[0]
    for i in range(d):
        for j in range(i + 1, d):
            a, b = marks[i, j], marks[j, i]
            if a == -1 and b == 1:
                out[(i, j)] = "F"
            elif a == 1 and b == -1:
                out[(i, j)] = "R"
            elif a == -1 and b == -1:
                out[(i, j)] = "U"
    return out


def cpdag_accuracy(true_marks: np.ndarray, B_est: np.ndarray) -> dict:
    """``count_accuracy`` generalized to a CPDAG-vs-CPDAG comparison.

    ``true_marks`` is the true CPDAG (from :func:`dag_to_cpdag_marks`); ``B_est``
    is a 0/1 directed estimate that is converted to its own CPDAG. Each unordered
    pair gets a mark in {none, ->, <-, --}; an estimated edge counts only when its
    mark matches the truth exactly. A cyclic estimate has no CPDAG, so every
    metric is NaN (nnz still reports the edge count), matching
    :func:`safe_count_accuracy`.
    """
    Be = np.asarray(B_est) != 0
    if not is_dag(Be):
        return {k: float("nan") for k in _CPDAG_KEYS} | {"cpdag_nnz": int(Be.sum())}
    T = _pair_marks(true_marks)
    E = _pair_marks(dag_to_cpdag_marks(Be))
    hit = mism = extra = missing = 0
    for k in set(T) | set(E):
        t, e = T.get(k), E.get(k)
        if e is None:
            missing += 1
        elif t is None:
            extra += 1
        elif t == e:
            hit += 1
        else:
            mism += 1
    pred = hit + mism + extra
    cond = hit + mism + missing
    d = true_marks.shape[0]
    cond_neg = 0.5 * d * (d - 1) - cond
    return {
        "cpdag_fdr": (mism + extra) / max(pred, 1),
        "cpdag_tpr": hit / max(cond, 1),
        "cpdag_fpr": (mism + extra) / max(cond_neg, 1),
        "cpdag_shd": mism + extra + missing,
        "cpdag_nnz": pred,
    }


def skeleton_accuracy(B_true: np.ndarray, B_est: np.ndarray) -> dict:
    """Direction-agnostic counterpart of dagrad's `count_accuracy`.

    Scores the undirected *skeleton*: an edge between i and j counts as
    present if it exists in either direction. Same metric definitions as
    `count_accuracy`, but with no "reverse" term (direction is ignored), so
    `shd` is just the undirected symmetric difference and a cyclic estimate
    is fine (unlike `count_accuracy`, which requires a DAG estimate).
    """
    Bt = np.asarray(B_true) != 0
    Be = np.asarray(B_est) != 0
    d = Bt.shape[0]
    # Unique undirected edges (upper triangle of the symmetrized adjacency).
    skel_true = np.triu(Bt | Bt.T, k=1)
    skel_est = np.triu(Be | Be.T, k=1)

    true_pos = int(np.sum(skel_est & skel_true))
    false_pos = int(np.sum(skel_est & ~skel_true))
    missing = int(np.sum(skel_true & ~skel_est))
    pred_size = int(np.sum(skel_est))
    cond_size = int(np.sum(skel_true))
    cond_neg = 0.5 * d * (d - 1) - cond_size

    return {
        "fdr": false_pos / max(pred_size, 1),
        "tpr": true_pos / max(cond_size, 1),
        "fpr": false_pos / max(cond_neg, 1),
        "shd": false_pos + missing,
        "nnz": pred_size,
    }


def safe_count_accuracy(B_true: np.ndarray, B_est: np.ndarray) -> dict:
    """count_accuracy raises if the estimate has a cycle; record NaN/Nones."""
    try:
        return count_accuracy(B_true, B_est)
    except Exception:
        return {k: float("nan") for k in _METRIC_KEYS} | {
            "nnz": int(np.sum(B_est)),
        }


def _directed_and_skel(B_true: np.ndarray, B_est: np.ndarray) -> dict:
    directed = safe_count_accuracy(B_true, B_est)
    skel = skeleton_accuracy(B_true, B_est)
    return {**directed, **{f"skel_{k}": v for k, v in skel.items()}}


def edge_magnitudes(W: np.ndarray) -> np.ndarray:
    """|W| with the diagonal zeroed: the learned graph as stored and scored."""
    A = np.abs(np.asarray(W, dtype=float))
    np.fill_diagonal(A, 0.0)
    return A


def metrics_at_threshold(
    W_est: np.ndarray,
    B_true: np.ndarray,
    given_mask: np.ndarray | None = None,
    true_cpdag: np.ndarray | None = None,
) -> dict:
    """Directed + skeleton metrics for one thresholded estimate (nonzero = edge).

    ``given_mask`` is a (d, d) boolean matrix of given edges. Given positions
    are removed from both the ground truth and the estimate before scoring
    (pairs given in either direction are excluded from the skeleton), so the
    plain metrics describe recovery of E_unknown only. ``full_*`` metrics over
    the entire graph are reported alongside whenever a mask is supplied.

    If ``true_cpdag`` (the true CPDAG marks, from :func:`dag_to_cpdag_marks`) is
    given, ``cpdag_*`` whole-graph Markov-equivalence metrics are appended.
    """
    Bt = np.asarray(B_true) != 0
    Be = np.asarray(W_est) != 0
    cpdag = cpdag_accuracy(true_cpdag, Be) if true_cpdag is not None else {}
    if given_mask is None:
        return {**_directed_and_skel(Bt, Be), **cpdag}
    g = np.asarray(given_mask, dtype=bool)
    g_sym = g | g.T
    unknown = _directed_and_skel(Bt & ~g, Be & ~g)
    # skeleton parts must exclude the given *pairs*, not just the direction
    skel = skeleton_accuracy(Bt & ~g_sym, Be & ~g_sym)
    unknown |= {f"skel_{k}": v for k, v in skel.items()}
    full = _directed_and_skel(Bt, Be)
    return {**unknown, **{f"full_{k}": v for k, v in full.items()}, **cpdag}


def smallest_dag_metrics(
    W: np.ndarray, B_true: np.ndarray, given_mask: np.ndarray | None = None
) -> dict:
    """The smallest cutoff on the edge magnitudes ``W`` (|W|, zero diagonal)
    whose thresholded graph is acyclic, as ``dag_threshold``, and the metrics
    there as ``dag_<metric>`` (no ``cpdag_*``). An estimate that is cyclic at
    a plotted threshold can be scored by these instead ("fill"). All NaN when
    no cutoff gives a DAG, or W is empty.

    Acyclicity is monotone in the cutoff (raising it only removes edges), so
    the crossover is found by binary search over the distinct magnitudes."""
    W = np.asarray(W, dtype=float)
    keys = metrics_at_threshold(np.zeros_like(W), B_true, given_mask)
    mags = np.unique(W[W > 0])
    if mags.size == 0 or not is_dag((W >= mags[-1]).astype(float)):
        return {"dag_threshold": np.nan, **{f"dag_{k}": np.nan for k in keys}}
    lo, hi = 0, len(mags) - 1  # smallest index whose cutoff is acyclic
    while lo < hi:
        mid = (lo + hi) // 2
        if is_dag((W >= mags[mid]).astype(float)):
            hi = mid
        else:
            lo = mid + 1
    met = metrics_at_threshold(np.where(W < mags[lo], 0.0, W), B_true, given_mask)
    return {"dag_threshold": float(mags[lo]), **{f"dag_{k}": v for k, v in met.items()}}


def threshold_metric_rows(
    W_raw: np.ndarray,
    thresholds: list[float],
    B_true: np.ndarray,
    given_mask: np.ndarray | None = None,
    include_cpdag: bool = False,
) -> list[dict]:
    """Evaluate a raw (unthresholded) weight matrix at every threshold.

    Returns one dict per threshold: ``{"w_threshold": wt, <metrics...>}``. With
    ``include_cpdag``, each row also carries the ``cpdag_*`` columns (the true
    CPDAG is computed once and reused across thresholds). Every row also
    carries the run's ``smallest_dag_metrics`` (``dag_*``).
    """
    true_cpdag = dag_to_cpdag_marks(B_true) if include_cpdag else None
    dag = smallest_dag_metrics(edge_magnitudes(W_raw), B_true, given_mask)
    rows = []
    for wt in thresholds:
        W_th = edge_magnitudes(np.where(np.abs(W_raw) < wt, 0.0, W_raw))
        rows.append(
            {
                "w_threshold": wt,
                **metrics_at_threshold(W_th, B_true, given_mask, true_cpdag),
                **dag,
            }
        )
    return rows
