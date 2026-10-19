"""Pure-numpy fusion and scoring shared by inference and Optuna tuning."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields

import numpy as np


@dataclass
class ScoringParams:
    w_ft1: float = 0.0
    w_bm25: float = 0.0
    rrf_k: int = 60
    dense_top_k: int = 200
    bm25_top_k: int = 200
    pool_lambda: float = 0.0  # weight of a document's second-best chunk
    alpha: float = 0.0  # reranker share of the final score
    rerank_top_n: int = 30
    chunks_per_doc: int = 2
    final_k: int = 5

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> ScoringParams:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


def rrf_chunk_scores(
    rank_lists: Sequence[Sequence[int]], weights: Sequence[float], rrf_k: int
) -> dict[int, float]:
    """Weighted reciprocal rank fusion: score(c) = sum_i w_i / (k + rank_i(c))."""
    scores: dict[int, float] = defaultdict(float)
    for w, ids in zip(weights, rank_lists):
        if w <= 0.0:
            continue
        for rank, cid in enumerate(ids, start=1):
            scores[int(cid)] += w / (rrf_k + rank)
    return dict(scores)


def pool_to_docs(
    chunk_scores: dict[int, float],
    chunk_doc: np.ndarray,
    pool_lambda: float,
    chunks_per_doc: int,
) -> tuple[dict[int, float], dict[int, list[int]]]:
    """Document score = best chunk + pool_lambda * second-best chunk; also return top chunks."""
    per_doc: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for cid, s in chunk_scores.items():
        per_doc[int(chunk_doc[cid])].append((s, cid))

    doc_scores: dict[int, float] = {}
    best_chunks: dict[int, list[int]] = {}
    for did, items in per_doc.items():
        items.sort(reverse=True)
        second = items[1][0] if len(items) > 1 else 0.0
        doc_scores[did] = items[0][0] + pool_lambda * second
        best_chunks[did] = [c for _, c in items[:chunks_per_doc]]
    return doc_scores, best_chunks


def _minmax(d: dict[int, float]) -> dict[int, float]:
    if not d:
        return {}
    vals = np.fromiter(d.values(), dtype=np.float64, count=len(d))
    lo, span = float(vals.min()), float(vals.max() - vals.min())
    if span <= 0:
        return {k: 0.0 for k in d}
    return {k: (v - lo) / span for k, v in d.items()}


def blend_rerank(
    doc_scores: dict[int, float],
    best_chunks: dict[int, list[int]],
    rerank_lookup: dict[int, float],
    p: ScoringParams,
) -> dict[int, float]:
    """Linearly blend min-max normalised retrieval and reranker scores."""
    normed = _minmax(doc_scores)
    if p.alpha <= 0.0 or not rerank_lookup or not normed:
        return normed

    # Blending beats replacing: the fine-tuned bi-encoder is stronger than the zero-shot reranker.
    logits: dict[int, float] = {}
    for d in sorted(doc_scores, key=doc_scores.get, reverse=True)[: p.rerank_top_n]:
        vals = [rerank_lookup[c] for c in best_chunks.get(d, [])[: p.chunks_per_doc] if c in rerank_lookup]
        if vals:
            logits[d] = max(vals)
    if not logits:
        return normed
    nl = _minmax(logits)
    return {d: p.alpha * nl.get(d, 0.0) + (1.0 - p.alpha) * s for d, s in normed.items()}


def rank_with_chunks(
    rank_lists: Sequence[Sequence[int]],
    weights: Sequence[float],
    rerank_lookup: dict[int, float],
    chunk_doc: np.ndarray,
    p: ScoringParams,
) -> list[tuple[int, int]]:
    """Full scoring for one query; returns (doc_id, best chunk_id) for the top final_k documents."""
    chunk_scores = rrf_chunk_scores(rank_lists, weights, p.rrf_k)
    if not chunk_scores:
        return []
    doc_scores, best_chunks = pool_to_docs(chunk_scores, chunk_doc, p.pool_lambda, p.chunks_per_doc)
    scores = blend_rerank(doc_scores, best_chunks, rerank_lookup, p)
    top = sorted(scores, key=scores.get, reverse=True)[: p.final_k]
    return [(d, max(best_chunks[d], key=lambda c: rerank_lookup.get(c, -1e9))) for d in top]


def rank_one(
    rank_lists: Sequence[Sequence[int]],
    weights: Sequence[float],
    rerank_lookup: dict[int, float],
    chunk_doc: np.ndarray,
    p: ScoringParams,
) -> list[int]:
    """Top final_k document ids for one query."""
    return [d for d, _ in rank_with_chunks(rank_lists, weights, rerank_lookup, chunk_doc, p)]
