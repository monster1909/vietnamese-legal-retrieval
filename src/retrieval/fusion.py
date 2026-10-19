"""Query-time hybrid retrieval: two dense encoders + BM25 -> RRF -> reranker blend."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from tqdm.auto import tqdm

from ..config import CFG, ModelConfig
from .bm25 import BM25Index
from .dense import DenseIndex, encode_texts
from .rerank import Reranker
from .scoring import ScoringParams, pool_to_docs, rank_with_chunks, rrf_chunk_scores


class HybridRetriever:
    def __init__(
        self,
        chunks: pd.DataFrame,
        encoder,
        dense_index: DenseIndex,
        encoder_ft1,
        dense_index_ft1: DenseIndex,
        bm25_index: BM25Index,
        params: ScoringParams,
        reranker: Reranker | None = None,
        model_cfg: ModelConfig = CFG.model,
    ):
        self.encoder, self.dense_index = encoder, dense_index
        self.encoder_ft1, self.dense_index_ft1 = encoder_ft1, dense_index_ft1
        self.bm25_index = bm25_index
        self.params = params
        self.reranker = reranker
        self.model_cfg = model_cfg
        chunks = chunks.reset_index(drop=True)
        self.chunk_doc = chunks["doc_id"].to_numpy()
        self.chunk_text = chunks["text"].tolist()

    def _rerank_lookup(self, query: str, rank_lists, weights, p: ScoringParams) -> dict[int, float]:
        """Reranker logits for the best chunks of the top rerank_top_n documents."""
        if p.alpha <= 0.0 or self.reranker is None:
            return {}
        doc_scores, best_chunks = pool_to_docs(
            rrf_chunk_scores(rank_lists, weights, p.rrf_k), self.chunk_doc, p.pool_lambda, p.chunks_per_doc
        )
        top_docs = sorted(doc_scores, key=doc_scores.get, reverse=True)[: p.rerank_top_n]
        cids = [c for d in top_docs for c in best_chunks.get(d, [])]
        if not cids:
            return {}
        logits = self.reranker.score([(query, self.chunk_text[c]) for c in cids])
        return {c: float(s) for c, s in zip(cids, logits)}

    def retrieve(
        self, queries: Sequence[str], params: ScoringParams | None = None, show_progress: bool = True
    ) -> list[list[tuple[int, int]]]:
        """(doc_id, best chunk_id) pairs, best first, for each query; params overrides the tuned ones."""
        p = params or self.params
        queries = list(queries)
        q = encode_texts(self.encoder, queries, self.model_cfg, show_progress=show_progress)
        q1 = encode_texts(self.encoder_ft1, queries, self.model_cfg, show_progress=show_progress)
        dense_ids, _ = self.dense_index.search_batched(q, p.dense_top_k, show_progress=show_progress)
        ids1, _ = self.dense_index_ft1.search_batched(q1, p.dense_top_k, show_progress=show_progress)

        weights = [1.0, p.w_ft1, p.w_bm25]
        out: list[list[tuple[int, int]]] = []
        for i, query in enumerate(tqdm(queries, desc="retrieve", disable=not show_progress)):
            rank_lists = [
                dense_ids[i].tolist(),
                ids1[i].tolist(),
                self.bm25_index.search(query, p.bm25_top_k)[0].tolist() if p.w_bm25 > 0 else [],
            ]
            lookup = self._rerank_lookup(query, rank_lists, weights, p)
            out.append(rank_with_chunks(rank_lists, weights, lookup, self.chunk_doc, p))
        return out
