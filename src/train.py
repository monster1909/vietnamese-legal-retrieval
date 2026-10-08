"""Fine-tune the bi-encoder with CachedMultipleNegativesRankingLoss (GradCache) for large effective batches."""

from __future__ import annotations

import math
import shutil
from collections.abc import Sequence
from pathlib import Path

from .config import CFG, ModelConfig, TrainConfig


def make_loss_progress_callback():
    """Trainer callback printing loss with a 20-step moving average."""
    from transformers import TrainerCallback

    class LossProgress(TrainerCallback):
        def __init__(self):
            self.history: list[float] = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or "loss" not in logs:
                return
            self.history.append(logs["loss"])
            recent = self.history[-20:]
            avg = sum(recent) / len(recent)
            pct = 100 * state.global_step / max(state.max_steps or 1, 1)
            lr = logs.get("learning_rate")
            lr_s = f"  lr {lr:.2e}" if lr else ""
            print(
                f"  step {state.global_step:>5}/{state.max_steps} ({pct:5.1f}%)  "
                f"loss {logs['loss']:.4f}  ma20 {avg:.4f}{lr_s}",
                flush=True,
            )

    return LossProgress()


def build_dataset(records: Sequence[dict], num_negatives: int):
    """Dataset with anchor/positive/negative_1..n columns; negatives are truncated or padded to n."""
    from datasets import Dataset

    usable = [r for r in records if r.get("negatives")]
    data = {"anchor": [], "positive": []}
    for i in range(num_negatives):
        data[f"negative_{i + 1}"] = []

    for r in usable:
        negs: list[str] = list(r["negatives"])[:num_negatives]
        while len(negs) < num_negatives:
            negs.append(negs[-1])
        data["anchor"].append(r["query"])
        data["positive"].append(r["positive"])
        for i, n in enumerate(negs):
            data[f"negative_{i + 1}"].append(n)

    dropped = len(records) - len(usable)
    if dropped:
        print(f"[train] dropped {dropped} records without negatives")
    return Dataset.from_dict(data)


def save_encoder(model, path: Path, fp16: bool = True) -> Path:
    """Save the encoder, in fp16 by default (half the size; inference runs fp16 anyway)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fp16:
        import torch

        model = model.to(torch.float16)
    model.save_pretrained(str(path))
    print(f"[train] saved {'fp16' if fp16 else 'fp32'} checkpoint to {path}")
    return path


def checkpoint_selection_score(
    recall_at_5: float,
    precision_at_5: float,
    ndcg_at_10: float,
    precision_weight: float = 1e-6,
    ndcg_weight: float = 1e-10,
) -> float:
    """Checkpoint score: Recall first, Precision then NDCG as tie-breaks."""
    return float(
        recall_at_5
        + precision_weight * precision_at_5
        + ndcg_weight * ndcg_at_10
    )


def document_retrieval_metrics(
    query_embeddings,
    corpus_embeddings,
    corpus_doc_ids,
    gold_docs: Sequence[Sequence[int | str]],
    recall_ks: Sequence[int] = (5, 100),
    precision_k: int = 5,
    ndcg_k: int = 10,
) -> dict[str, float]:
    """Document-level Recall/Precision/NDCG, ranking each document by its best chunk."""
    import numpy as np

    q = np.asarray(query_embeddings, dtype=np.float32)
    c = np.asarray(corpus_embeddings, dtype=np.float32)
    doc_ids = np.asarray(corpus_doc_ids)
    ks = sorted({int(k) for k in recall_ks if int(k) > 0})
    recalls = {k: [] for k in ks}
    precisions: list[float] = []
    ndcgs: list[float] = []

    for lo in range(0, len(q), 32):
        scores = q[lo : lo + 32] @ c.T
        for offset, row in enumerate(scores):
            qi = lo + offset
            ranked_chunks = np.argsort(-row, kind="stable")
            ranked_docs: list[int] = []
            seen: set[int] = set()
            for cid in ranked_chunks.tolist():
                did = int(doc_ids[cid])
                if did not in seen:
                    seen.add(did)
                    ranked_docs.append(did)

            gold = {int(x) for x in gold_docs[qi]}
            for k in ks:
                recalls[k].append(len(gold & set(ranked_docs[:k])) / len(gold) if gold else 0.0)

            predicted = set(ranked_docs[:precision_k])
            precisions.append(
                len(gold & predicted) / len(predicted) if predicted else 0.0
            )

            dcg = sum(
                1.0 / math.log2(rank + 2)
                for rank, did in enumerate(ranked_docs[:ndcg_k])
                if did in gold
            )
            ideal_hits = min(len(gold), ndcg_k)
            idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
            ndcgs.append(dcg / idcg if idcg else 0.0)

    metrics = {f"doc_recall_at_{k}": float(np.mean(v)) for k, v in recalls.items()}
    metrics[f"doc_precision_at_{precision_k}"] = (
        float(np.mean(precisions)) if precisions else 0.0
    )
    metrics[f"doc_ndcg_at_{ndcg_k}"] = float(np.mean(ndcgs)) if ndcgs else 0.0
    return metrics


def make_document_ir_evaluator(
    questions,
    chunks,
    sample_size: int = 10_000,
    seed: int = 42,
    name: str = "val-doc",
    encode_batch_size: int = 256,
    precision_weight: float = 1e-6,
    ndcg_weight: float = 1e-10,
):
    """Document-level evaluator on gold chunks plus a fixed random sample, used for early stopping."""
    import numpy as np
    from sentence_transformers.evaluation import SentenceEvaluator

    rng = np.random.default_rng(seed)
    gold_docs = {int(a) for q in questions if q.answer for a in q.answer}
    chunk_doc = chunks["doc_id"].to_numpy()
    gold_chunks = np.flatnonzero(np.isin(chunk_doc, list(gold_docs))).astype(np.int64)
    others = np.setdiff1d(np.arange(len(chunks)), np.array(gold_chunks, dtype=np.int64))
    extra = rng.choice(others, size=min(sample_size, len(others)), replace=False)
    corpus_ids = np.unique(np.concatenate([gold_chunks, extra])).astype(np.int64)
    usable = [q for q in questions if q.answer]
    query_texts = [q.question for q in usable]
    gold_by_query = [[int(a) for a in q.answer] for q in usable]
    corpus_texts = chunks.iloc[corpus_ids]["text"].tolist()
    corpus_docs = chunk_doc[corpus_ids]

    if not query_texts or not corpus_texts:
        raise ValueError("empty query set or corpus for the evaluator")
    print(
        f"[eval] {len(query_texts)} questions | {len(corpus_texts):,} chunks "
        f"| {len(np.unique(corpus_docs)):,} documents"
    )

    class DocumentIREvaluator(SentenceEvaluator):
        def __init__(self):
            super().__init__()
            self.name = name
            self.primary_metric = f"{name}_doc_selection_score"
            self.greater_is_better = True

        def __call__(self, model, output_path=None, epoch=-1, steps=-1):
            q_emb = model.encode(
                query_texts,
                batch_size=encode_batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            c_emb = model.encode(
                corpus_texts,
                batch_size=encode_batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            metrics = document_retrieval_metrics(
                q_emb, c_emb, corpus_docs, gold_by_query,
                recall_ks=(5, 100), precision_k=5, ndcg_k=10,
            )
            metrics["doc_selection_score"] = checkpoint_selection_score(
                metrics["doc_recall_at_5"],
                metrics["doc_precision_at_5"],
                metrics["doc_ndcg_at_10"],
                precision_weight=precision_weight,
                ndcg_weight=ndcg_weight,
            )
            metrics = self.prefix_name_to_metrics(metrics, self.name)
            print(
                f"[eval] step={steps} "
                f"R@5={metrics[self.name + '_doc_recall_at_5']:.4f} "
                f"P@5={metrics[self.name + '_doc_precision_at_5']:.4f} "
                f"R@100={metrics[self.name + '_doc_recall_at_100']:.4f} "
                f"NDCG@10={metrics[self.name + '_doc_ndcg_at_10']:.4f} "
                f"select={metrics[self.primary_metric]:.8f}"
            )
            return metrics

    return DocumentIREvaluator()


def train(
    records: Sequence[dict],
    output_dir: Path,
    train_cfg: TrainConfig = CFG.train,
    model_cfg: ModelConfig = CFG.model,
    evaluator=None,
):
    """Fine-tune the base encoder, save it to output_dir/final and return it."""
    from packaging.version import parse as parse_version
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
    from transformers import EarlyStoppingCallback
    from transformers import __version__ as transformers_version

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import torch

    # fp32 master weights (fp16 checkpoints otherwise train straight to NaN); MPS SDPA lacks dropout.
    kwargs = {"dtype": torch.float32}
    if torch.backends.mps.is_available():
        kwargs["attn_implementation"] = "eager"
    model = SentenceTransformer(model_cfg.base_model, model_kwargs=kwargs)
    model.max_seq_length = model_cfg.max_seq_length

    train_ds = build_dataset(records, train_cfg.num_hard_negatives)

    loss = CachedMultipleNegativesRankingLoss(
        model, mini_batch_size=train_cfg.mini_batch_size
    )

    checkpoint_dir = output_dir / "checkpoints"
    has_eval = evaluator is not None
    use_early_stop = bool(train_cfg.early_stopping and has_eval)
    patience_calls = math.ceil(train_cfg.early_stopping_patience_steps / train_cfg.eval_steps)

    warmup_kwargs = (
        {"warmup_steps": train_cfg.warmup_ratio}
        if parse_version(transformers_version).major >= 5
        else {"warmup_ratio": train_cfg.warmup_ratio}
    )
    args = SentenceTransformerTrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=train_cfg.epochs,
        per_device_train_batch_size=train_cfg.train_batch_size,
        per_device_eval_batch_size=train_cfg.train_batch_size,
        learning_rate=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
        max_grad_norm=train_cfg.max_grad_norm,
        lr_scheduler_type=train_cfg.lr_scheduler_type,
        optim=train_cfg.optim,
        bf16=train_cfg.bf16,
        tf32=train_cfg.tf32,
        gradient_checkpointing=False,  # conflicts with GradCache
        logging_steps=train_cfg.logging_steps,
        eval_strategy="steps" if has_eval else "no",
        eval_steps=train_cfg.eval_steps if has_eval else None,
        save_strategy="steps" if use_early_stop else "no",
        save_steps=train_cfg.eval_steps if use_early_stop else 500,
        save_total_limit=train_cfg.save_total_limit,
        load_best_model_at_end=use_early_stop,
        metric_for_best_model=(f"eval_{evaluator.primary_metric}" if use_early_stop else None),
        greater_is_better=(evaluator.greater_is_better if use_early_stop else None),
        seed=train_cfg.seed,
        report_to=[],
        dataloader_num_workers=train_cfg.dataloader_num_workers,
        disable_tqdm=False,
        logging_first_step=True,
        **warmup_kwargs,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        loss=loss,
        evaluator=evaluator,
    )
    trainer.add_callback(make_loss_progress_callback())
    if use_early_stop:
        trainer.add_callback(
            EarlyStoppingCallback(
                early_stopping_patience=patience_calls,
                early_stopping_threshold=train_cfg.early_stopping_threshold,
            )
        )

    steps_per_epoch = max(1, math.ceil(len(train_ds) / train_cfg.train_batch_size))
    print(
        f"[train] {len(train_ds):,} examples | batch {train_cfg.train_batch_size} "
        f"(mini {train_cfg.mini_batch_size}) | {steps_per_epoch} step/epoch "
        f"× {train_cfg.epochs} epoch"
    )
    if use_early_stop:
        print(
            f"[train] eval every {train_cfg.eval_steps} steps | patience "
            f"{train_cfg.early_stopping_patience_steps} steps ({patience_calls} evals) "
            f"| metric={evaluator.primary_metric}"
        )
    trainer.train()
    save_encoder(model, output_dir / "final", fp16=train_cfg.save_fp16)
    if use_early_stop:
        print(
            f"[train] best step={trainer.state.best_global_step} "
            f"metric={trainer.state.best_metric}"
        )
        if train_cfg.cleanup_checkpoints and checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
            print(f"[train] removed intermediate checkpoints in {checkpoint_dir}")
    return model
