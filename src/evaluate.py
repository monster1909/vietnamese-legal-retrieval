"""Competition metrics: Recall (primary) and Precision (tie-break), plus MAP/MRR/Hit@1."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

MAX_DOCS_PER_QUESTION = 5  # a question with more predictions scores 0 on every metric


@dataclass
class Metrics:
    recall: float
    precision: float
    num_questions: int
    num_violations: int
    mrr: float = 0.0
    map: float = 0.0
    hit_at_1: float = 0.0

    def __str__(self) -> str:
        return (
            f"Recall={self.recall:.4f}  Precision={self.precision:.4f}  "
            f"MAP={self.map:.4f}  MRR={self.mrr:.4f}  Hit@1={self.hit_at_1:.4f}  "
            f"n={self.num_questions}  violations={self.num_violations}"
        )


def evaluate(
    predictions: dict[str, Sequence[str | int]],
    gold: dict[str, Sequence[str | int]],
) -> Metrics:
    """Score predictions against gold, both keyed by question id."""
    rows: list[tuple[float, float, float, float, float]] = []
    violations = 0
    for qid, gold_ids in gold.items():
        g = {str(x) for x in gold_ids}
        pred = list(dict.fromkeys(str(x) for x in predictions.get(qid, [])))
        if len(pred) > MAX_DOCS_PER_QUESTION:
            violations += 1
            rows.append((0.0, 0.0, 0.0, 0.0, 0.0))
            continue
        hits = [rank for rank, d in enumerate(pred, 1) if d in g]
        ap = sum(i / rank for i, rank in enumerate(hits, 1)) / len(g) if g else 0.0
        rows.append((
            len(hits) / len(g) if g else 0.0,
            len(hits) / len(pred) if pred else 0.0,
            1.0 / hits[0] if hits else 0.0,
            ap,
            float(bool(hits) and hits[0] == 1),
        ))
    n = len(rows) or 1
    recall, precision, mrr, map_, hit1 = (sum(col) / n for col in zip(*rows)) if rows else (0.0,) * 5
    return Metrics(recall, precision, len(rows), violations, mrr, map_, hit1)
