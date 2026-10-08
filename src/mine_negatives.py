"""Weakly supervised training data: pick positive chunks inside gold documents and mine hard negatives."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .config import CFG, ModelConfig, TrainConfig
from .data import Question


def select_records(
    query: str,
    qv: np.ndarray,
    gold_docs: set[int],
    cand_ids: np.ndarray,
    chunk_doc: np.ndarray,
    texts: Sequence[str],
    corpus_emb: np.ndarray,
    emb_row_of: dict[int, int] | None,
    cfg: TrainConfig,
) -> list[dict]:
    """Records for one query; emb_row_of maps chunk_id to a row of a partial corpus_emb (None = identity)."""

    def row_of(cid: int) -> int | None:
        if emb_row_of is None:
            return cid if cid < len(corpus_emb) else None
        return emb_row_of.get(cid)

    records: list[dict] = []
    for gd in sorted(gold_docs):
        rows = np.flatnonzero(chunk_doc == gd)
        if rows.size == 0:
            continue
        emb_rows = [row_of(int(c)) for c in rows]
        usable = [(int(c), r) for c, r in zip(rows, emb_rows) if r is not None]
        if not usable:
            continue

        idx = np.array([r for _, r in usable], dtype=np.int64)
        sims = corpus_emb[idx].astype(np.float32) @ qv
        order = np.argsort(-sims)[: cfg.positives_per_gold]

        for pos_rank in order:
            cid = usable[int(pos_rank)][0]
            pos_score = float(sims[int(pos_rank)])

            negs: list[str] = []
            for c in cand_ids.tolist():
                c = int(c)
                if int(chunk_doc[c]) in gold_docs:
                    continue
                r = row_of(c)
                if r is not None:
                    score = float(corpus_emb[r].astype(np.float32) @ qv)
                    # Near-positive scores are likely unlabeled relevant documents.
                    if score >= cfg.fn_threshold * pos_score:
                        continue
                negs.append(texts[c])
                if len(negs) >= cfg.num_hard_negatives:
                    break

            if negs:
                records.append(
                    {"query": query, "positive": texts[cid], "negatives": negs}
                )
    return records


def mine(
    questions: Sequence[Question],
    chunks: pd.DataFrame,
    corpus_emb: np.ndarray,
    encoder,
    dense_index=None,
    bm25_index=None,
    candidate_source: str = "dense",
    emb_row_of: dict[int, int] | None = None,
    train_cfg: TrainConfig = CFG.train,
    model_cfg: ModelConfig = CFG.model,
    show_progress: bool = True,
) -> list[dict]:
    """Build {query, positive, negatives} records with negatives from "dense" or "bm25" candidates."""
    from .retrieval.dense import encode_texts

    if candidate_source == "dense" and dense_index is None:
        raise ValueError("candidate_source='dense' needs dense_index")
    if candidate_source == "bm25" and bm25_index is None:
        raise ValueError("candidate_source='bm25' needs bm25_index")

    texts = chunks["text"].tolist()
    chunk_doc = chunks["doc_id"].to_numpy()

    queries = [q.question for q in questions]
    q_emb = encode_texts(encoder, queries, model_cfg, show_progress=show_progress)

    if candidate_source == "dense":
        cand_ids, _ = dense_index.search_batched(
            q_emb, train_cfg.mine_top_k, show_progress=show_progress
        )
    else:
        cand_ids = np.full((len(queries), train_cfg.mine_top_k), -1, dtype=np.int64)
        for i, q in enumerate(tqdm(queries, desc="bm25 mine", disable=not show_progress)):
            ids, _ = bm25_index.search(q, train_cfg.mine_top_k)
            cand_ids[i, : len(ids)] = ids

    records: list[dict] = []
    skipped = 0
    for i, q in enumerate(tqdm(questions, desc="mine", disable=not show_progress)):
        if not q.answer:
            continue
        row = cand_ids[i]
        recs = select_records(
            query=q.question,
            qv=q_emb[i].astype(np.float32),
            gold_docs={int(a) for a in q.answer},
            cand_ids=row[row >= 0],
            chunk_doc=chunk_doc,
            texts=texts,
            corpus_emb=corpus_emb,
            emb_row_of=emb_row_of,
            cfg=train_cfg,
        )
        if not recs:
            skipped += 1
        records.extend(recs)

    if skipped:
        print(f"[mine] skipped {skipped} questions with no usable positive/negative")
    print(f"[mine] {len(records):,} records from {len(questions):,} questions ({candidate_source})")
    return records


def gold_chunk_subset(
    chunks: pd.DataFrame, questions: Sequence[Question]
) -> tuple[list[int], dict[int, int]]:
    """Chunk ids of gold documents (~36% of the corpus) and their row in the partial embedding matrix."""
    gold_docs = {int(a) for q in questions if q.answer for a in q.answer}
    mask = chunks["doc_id"].isin(gold_docs).to_numpy()
    cids = np.flatnonzero(mask).tolist()
    return cids, {int(c): i for i, c in enumerate(cids)}
