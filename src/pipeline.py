"""End-to-end LegalIR pipeline: train, tune, predict and interactive search over one artifacts dir."""

from __future__ import annotations

import argparse
import gc
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from .config import CFG, Paths

ENCODER = "models/encoder"
ENCODER_FT1 = "models/encoder_ft1"
EMB = "emb_encoder.npy"
EMB_FT1 = "emb_encoder_ft1.npy"
CHUNKS = "chunks.parquet"
BM25 = "bm25_index.npz"
BEST_CONFIG = "best_config.json"

# Candidate pool used to collect reranker scores before tuning.
POOL_TOP_K = 1000
POOL_DOCS_PER_RETRIEVER = 150
POOL_CHUNKS_PER_DOC = 4
POOL_RRF_K = 60


def _free_gpu() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_chunks(art: Path):
    import pandas as pd

    return pd.read_parquet(art / CHUNKS).reset_index(drop=True)


def _load_encoders(art: Path):
    """Return (encoder, index, encoder_ft1, index_ft1)."""
    import numpy as np

    from .retrieval.dense import DenseIndex, load_encoder

    out = []
    for model, emb in ((ENCODER, EMB), (ENCODER_FT1, EMB_FT1)):
        out.append(load_encoder(art / model, cfg=CFG.model))
        out.append(DenseIndex(np.load(art / emb, mmap_mode="r")))
    return tuple(out)



def train(data_dir: Path, art: Path) -> None:
    """Chunk the corpus, build BM25 and fine-tune the bi-encoder in two hard-negative rounds."""
    from .chunking import build_chunks
    from .data import iter_documents, load_train, split_train_val
    from .doc_header import extract_header
    from .mine_negatives import gold_chunk_subset, mine
    from .retrieval.bm25 import BM25Index, build_bm25
    from .retrieval.dense import DenseIndex, encode_corpus, encode_texts, load_encoder
    from .train import make_document_ir_evaluator
    from .train import train as fit_encoder

    paths, tcfg, mcfg = Paths(root=data_dir), CFG.train, CFG.model
    art.mkdir(parents=True, exist_ok=True)

    if (art / CHUNKS).exists() and (art / BM25).exists():
        chunks, bm25 = _load_chunks(art), BM25Index.load(art / BM25)
    else:
        docs = list(iter_documents(paths))
        chunks = build_chunks(docs, cfg=CFG.chunk).reset_index(drop=True)
        chunks.to_parquet(art / CHUNKS, index=False, compression="zstd")
        doc_numbers = {d.id: extract_header(d.passage).number for d in docs}
        bm25 = build_bm25(
            chunks["raw_text"].tolist(),
            doc_numbers=[doc_numbers.get(int(d), "") for d in chunks["doc_id"]],
            cfg=CFG.bm25,
        )
        bm25.save(art / BM25)
    texts = chunks["text"].tolist()
    print(f"[train] {len(chunks):,} chunks / {chunks['doc_id'].nunique():,} documents")

    tr, val = split_train_val(load_train(paths), tcfg.val_size, tcfg.seed)
    evaluator = None
    if tcfg.early_stopping:
        evaluator = make_document_ir_evaluator(
            val[: tcfg.eval_questions], chunks,
            sample_size=tcfg.eval_corpus_sample_size, seed=tcfg.seed,
            encode_batch_size=mcfg.encode_batch_size,
            precision_weight=tcfg.selection_precision_weight,
            ndcg_weight=tcfg.selection_ndcg_weight,
        )
    scratch = art / "models" / "_training"

    def fit(records, dst: Path):
        model = fit_encoder(records, scratch, train_cfg=tcfg, model_cfg=mcfg, evaluator=evaluator)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(scratch / "final"), str(dst))
        shutil.rmtree(scratch, ignore_errors=True)
        return model

    # Round 1 mines negatives from BM25 so the base model only encodes gold-document chunks.
    base = load_encoder(cfg=mcfg)
    gold_cids, row_of = gold_chunk_subset(chunks, tr)
    emb_gold = encode_texts(base, [texts[c] for c in gold_cids], cfg=mcfg)
    records1 = mine(
        tr, chunks, emb_gold, base, bm25_index=bm25, candidate_source="bm25",
        emb_row_of=row_of, train_cfg=tcfg, model_cfg=mcfg,
    )
    del base, emb_gold
    _free_gpu()

    if (art / ENCODER_FT1).exists():
        enc1 = load_encoder(art / ENCODER_FT1, cfg=mcfg)
    else:
        enc1 = fit(records1, art / ENCODER_FT1)
    emb1 = encode_corpus(enc1, texts, art / EMB_FT1, cfg=mcfg)

    # Round 2 re-mines negatives with the round-1 encoder and trains on both rounds.
    records2 = mine(
        tr, chunks, emb1, enc1, dense_index=DenseIndex(emb1), candidate_source="dense",
        train_cfg=tcfg, model_cfg=mcfg,
    )
    del enc1, emb1
    _free_gpu()
    enc2 = fit(records1 + records2, art / ENCODER)
    encode_corpus(enc2, texts, art / EMB, cfg=mcfg, force=True)
    print(f"[train] done -> {art}")



def tune(data_dir: Path, art: Path, n_trials: int = 1500) -> dict:
    """Rerank a wide validation candidate pool once (cached), then search fusion params with Optuna."""
    from .data import load_train, split_train_val
    from .retrieval.bm25 import BM25Index
    from .retrieval.cache import (
        CandidateCache,
        build_candidate_pool,
        score_candidate_pool,
    )
    from .retrieval.dense import encode_texts
    from .retrieval.rerank import Reranker
    from .tune import run_study, save_best_config

    tcfg, mcfg = CFG.train, CFG.model
    _, val = split_train_val(load_train(Paths(root=data_dir)), tcfg.val_size, tcfg.seed)
    queries = [q.question for q in val]

    chunks = _load_chunks(art)
    chunk_doc = chunks["doc_id"].to_numpy()
    cache_path = art / "cache_val.npz"
    if cache_path.exists():
        pool = CandidateCache.load(cache_path)
    else:
        enc, index, enc1, index1 = _load_encoders(art)
        pool = build_candidate_pool(
            encode_texts(enc, queries, cfg=mcfg), encode_texts(enc1, queries, cfg=mcfg),
            queries, chunk_doc, index, index1, BM25Index.load(art / BM25),
            top_k=POOL_TOP_K, docs_per_retriever=POOL_DOCS_PER_RETRIEVER,
            chunks_per_doc=POOL_CHUNKS_PER_DOC, rrf_k=POOL_RRF_K,
        )
        del enc, index, enc1, index1
        _free_gpu()
        score_candidate_pool(pool, chunks["text"].tolist(), Reranker(cfg=mcfg))
        pool.save(cache_path)

    result = run_study(
        pool, chunk_doc,
        gold={q.qid: q.answer for q in val}, qids=[q.qid for q in val],
        tune_size=tcfg.tune_size, n_trials=n_trials,
        storage=f"sqlite:///{art / 'optuna.db'}", study_name="legalir",
        top_n=75, seed=tcfg.seed, max_rerank_top_n=160,
    )
    save_best_config(result, art / BEST_CONFIG)
    return result



@dataclass
class Hit:
    doc_id: int
    text: str  # best-matching chunk of the document


class Searcher:
    """Loads every artifact once; ``search`` answers one query, ``retrieve`` a batch."""

    def __init__(self, art: Path, rerank: bool = True):
        from .retrieval.bm25 import BM25Index
        from .retrieval.fusion import HybridRetriever
        from .retrieval.rerank import Reranker
        from .tune import load_best_config

        art = Path(art)
        params = load_best_config(art / BEST_CONFIG)
        if not rerank:
            params.alpha = 0.0
        self.chunks = _load_chunks(art)
        enc, index, enc1, index1 = _load_encoders(art)
        self.retriever = HybridRetriever(
            self.chunks, enc, index, enc1, index1, BM25Index.load(art / BM25), params,
            reranker=Reranker(cfg=CFG.model) if params.alpha > 0 else None,
        )
        first = self.chunks.drop_duplicates("doc_id")
        self._doc_text = dict(zip(first["doc_id"].astype(int), first["text"]))

    def retrieve(self, queries: list[str], k: int = 5) -> list[list[int]]:
        return self.retriever.retrieve(queries, top_k=k, show_progress=len(queries) > 1)

    def search(self, query: str, k: int = 5) -> list[Hit]:
        return [Hit(int(d), self._doc_text[int(d)]) for d in self.retrieve([query], k)[0]]


def predict(questions_json: Path, art: Path, out_dir: Path, k: int = 5) -> Path:
    """Write predictions.json ({qid: [doc_id, ...]}) for a question file."""
    from .data import load_questions

    questions = load_questions(Path(questions_json))
    preds = Searcher(art).retrieve([q.question for q in questions], k)
    out = Path(out_dir) / "predictions.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({q.qid: [str(d) for d in p] for q, p in zip(questions, preds)}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"[predict] {len(questions)} questions -> {out}")
    return out


def evaluate_val(data_dir: Path, art: Path, limit: int | None = None, rerank: bool = True):
    """Competition metrics of the full searcher on the held-out validation split."""
    from .data import load_train, split_train_val
    from .evaluate import evaluate

    _, val = split_train_val(load_train(Paths(root=data_dir)), CFG.train.val_size, CFG.train.seed)
    val = val[:limit] if limit else val
    preds = Searcher(art, rerank=rerank).retrieve([q.question for q in val])
    metrics = evaluate({q.qid: p for q, p in zip(val, preds)}, {q.qid: q.answer for q in val})
    print(metrics)
    return metrics


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m src.pipeline", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("train", "tune", "evaluate", "predict", "search"):
        p = sub.add_parser(name)
        p.add_argument("--artifacts", type=Path, default=Path("artifacts"))
        if name in ("train", "tune", "evaluate"):
            p.add_argument("--data", type=Path, default=Path("data"),
                           help="folder with selected-contexts.zip and train.json")
    sub.choices["tune"].add_argument("--trials", type=int, default=1500)
    sub.choices["evaluate"].add_argument("--limit", type=int, help="first N validation questions")
    sub.choices["evaluate"].add_argument("--no-rerank", action="store_true")
    sub.choices["predict"].add_argument("--questions", type=Path, required=True)
    sub.choices["predict"].add_argument("--out", type=Path, default=Path("output"))
    sub.choices["predict"].add_argument("-k", type=int, default=5)
    sub.choices["search"].add_argument("query")
    sub.choices["search"].add_argument("-k", type=int, default=5)
    a = ap.parse_args(argv)

    if a.cmd == "train":
        train(a.data, a.artifacts)
    elif a.cmd == "tune":
        tune(a.data, a.artifacts, a.trials)
    elif a.cmd == "evaluate":
        evaluate_val(a.data, a.artifacts, a.limit, rerank=not a.no_rerank)
    elif a.cmd == "predict":
        predict(a.questions, a.artifacts, a.out, a.k)
    else:
        for i, hit in enumerate(Searcher(a.artifacts).search(a.query, a.k), 1):
            print(f"{i}. doc {hit.doc_id}: {' '.join(hit.text.split())[:200]}")


if __name__ == "__main__":
    main()
