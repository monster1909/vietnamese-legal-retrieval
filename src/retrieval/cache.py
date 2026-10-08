"""Cache raw retriever rankings and reranker logits so tuning never touches the GPU."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .scoring import ScoringParams, rrf_chunk_scores

# (w_ft1, w_bm25) fusions whose top documents also enter the rerank pool.
FUSION_GRID = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.5), (1.0, 1.5), (0.5, 0.75))


def rerank_pool(
    rows: Sequence[np.ndarray],
    chunk_doc: np.ndarray,
    docs_per_retriever: int = 40,
    chunks_per_doc: int = 3,
    rrf_k: int = 60,
) -> list[int]:
    """Chunks to rerank for one query: top docs of each retriever and of each grid fusion."""
    rows = [[int(c) for c in np.asarray(r).tolist() if int(c) >= 0] for r in rows]
    picked: list[int] = []
    seen: set[int] = set()

    def take(cid: int) -> None:
        if cid not in seen:
            seen.add(cid)
            picked.append(cid)

    for row in rows:
        per_doc: dict[int, int] = {}
        for cid in row:
            did = int(chunk_doc[cid])
            if did not in per_doc:
                if len(per_doc) >= docs_per_retriever:
                    continue
                per_doc[did] = 0
            if per_doc[did] < chunks_per_doc:
                per_doc[did] += 1
                take(cid)

    for w_ft1, w_bm25 in FUSION_GRID:
        scores = rrf_chunk_scores(rows, [1.0, w_ft1, w_bm25], rrf_k)
        per_doc_fused: dict[int, list[tuple[float, int]]] = {}
        for cid, s in scores.items():
            per_doc_fused.setdefault(int(chunk_doc[cid]), []).append((s, cid))
        for items in per_doc_fused.values():
            items.sort(reverse=True)
        top = sorted(per_doc_fused, key=lambda d: per_doc_fused[d][0][0], reverse=True)
        for did in top[:docs_per_retriever]:
            for _, cid in per_doc_fused[did][:chunks_per_doc]:
                take(cid)
    return picked


@dataclass
class CandidateCache:
    queries: list[str]
    dense_ids: np.ndarray  # (n_q, K) round-2 encoder
    dense_ids_ft1: np.ndarray  # (n_q, K) round-1 encoder
    bm25_ids: np.ndarray  # (n_q, K), -1 pads short result lists
    rerank_pairs: np.ndarray  # (n_pairs, 2) of (query_idx, chunk_id)
    rerank_logits: np.ndarray  # (n_pairs,)
    has_reranker: bool = False

    def __post_init__(self) -> None:
        self._lookup: list[dict[int, float]] | None = None

    def rerank_lookup(self, i: int) -> dict[int, float]:
        """chunk_id -> reranker logit for query i."""
        if self._lookup is None:
            self._lookup = [{} for _ in self.queries]
            for (qi, cid), logit in zip(self.rerank_pairs.tolist(), self.rerank_logits.tolist()):
                self._lookup[qi][cid] = float(logit)
        return self._lookup[i]

    def rank_lists(self, i: int, p: ScoringParams) -> tuple[list[list[int]], list[float]]:
        """Per-retriever rankings truncated by p, with their RRF weights."""
        lists = [
            [x for x in self.dense_ids[i][: p.dense_top_k].tolist() if x >= 0],
            [x for x in self.dense_ids_ft1[i][: p.dense_top_k].tolist() if x >= 0],
            [x for x in self.bm25_ids[i][: p.bm25_top_k].tolist() if x >= 0],
        ]
        return lists, [1.0, p.w_ft1, p.w_bm25]

    def save(self, path) -> None:
        np.savez_compressed(
            path,
            queries=np.array(self.queries, dtype=object),
            dense_ids=self.dense_ids,
            dense_ids_ft1=self.dense_ids_ft1,
            bm25_ids=self.bm25_ids,
            rerank_pairs=self.rerank_pairs,
            rerank_logits=self.rerank_logits,
            has_reranker=np.array([self.has_reranker]),
        )

    @classmethod
    def load(cls, path) -> CandidateCache:
        d = np.load(path, allow_pickle=True)
        return cls(
            queries=[str(q) for q in d["queries"]],
            dense_ids=d["dense_ids"],
            dense_ids_ft1=d["dense_ids_ft1"],
            bm25_ids=d["bm25_ids"],
            rerank_pairs=d["rerank_pairs"],
            rerank_logits=d["rerank_logits"],
            has_reranker=bool(d["has_reranker"][0]),
        )


def build_candidate_pool(
    query_emb: np.ndarray,
    query_emb_ft1: np.ndarray,
    queries: Sequence[str],
    chunk_doc: np.ndarray,
    dense_index,
    dense_index_ft1,
    bm25_index,
    top_k: int = 500,
    docs_per_retriever: int = 40,
    chunks_per_doc: int = 3,
    rrf_k: int = 60,
) -> CandidateCache:
    """Retrieve top_k chunks per retriever and collect the rerank pool (not yet scored)."""
    from tqdm.auto import tqdm

    n_q = len(queries)
    dense_ids, _ = dense_index.search_batched(query_emb, top_k)
    ids1, _ = dense_index_ft1.search_batched(query_emb_ft1, top_k)
    bm_ids = np.full((n_q, top_k), -1, dtype=np.int32)
    for i, q in enumerate(tqdm(queries, desc="bm25")):
        ids, _ = bm25_index.search(q, top_k)
        bm_ids[i, : len(ids)] = ids

    pairs = [
        (i, cid)
        for i in tqdm(range(n_q), desc="rerank pool")
        for cid in rerank_pool(
            (dense_ids[i], ids1[i], bm_ids[i]), chunk_doc, docs_per_retriever, chunks_per_doc, rrf_k
        )
    ]
    return CandidateCache(
        queries=list(queries),
        dense_ids=dense_ids.astype(np.int32),
        dense_ids_ft1=ids1.astype(np.int32),
        bm25_ids=bm_ids,
        rerank_pairs=np.array(pairs, dtype=np.int32).reshape(-1, 2),
        rerank_logits=np.zeros(len(pairs), dtype=np.float32),
    )


def score_candidate_pool(cache: CandidateCache, chunk_text: Sequence[str], reranker) -> CandidateCache:
    """Run the cross-encoder over the pool and attach its logits."""
    pairs = [(cache.queries[qi], chunk_text[cid]) for qi, cid in cache.rerank_pairs.tolist()]
    cache.rerank_logits = np.asarray(reranker.score(pairs, show_progress=True), dtype=np.float32)
    cache.has_reranker = True
    cache._lookup = None
    return cache
