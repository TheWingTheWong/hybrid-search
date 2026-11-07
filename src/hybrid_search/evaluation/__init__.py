"""Offline evaluation: retrieval quality (MRR, nDCG, recall) and ANN recall."""

from .harness import (
    DEFAULT_RUNS,
    EvalReport,
    LabeledQuery,
    RunSpec,
    ann_recall,
    evaluate,
    load_queries,
    synthetic_ann_benchmark,
)
from .metrics import ndcg_at_k, overlap_at_k, recall_at_k, reciprocal_rank

__all__ = [
    "DEFAULT_RUNS",
    "EvalReport",
    "LabeledQuery",
    "RunSpec",
    "ann_recall",
    "evaluate",
    "load_queries",
    "ndcg_at_k",
    "overlap_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "synthetic_ann_benchmark",
]
