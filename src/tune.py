"""Optuna search over fusion parameters on a precomputed CandidateCache (CPU only)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np

from .evaluate import Metrics, evaluate
from .retrieval.cache import CandidateCache
from .retrieval.scoring import ScoringParams, rank_one


def score_params(
    cache: CandidateCache,
    chunk_doc: np.ndarray,
    p: ScoringParams,
    gold: dict[str, Sequence[str]],
    qids: Sequence[str],
    lo: int,
    hi: int,
    weights: Sequence[float] | None = None,
) -> Metrics:
    """Evaluate one parameter set on queries [lo, hi) of the cache; weights override the RRF weights."""
    preds: dict[str, list[int]] = {}
    for i in range(lo, hi):
        rank_lists, w = cache.rank_lists(i, p)
        preds[qids[i]] = rank_one(rank_lists, weights or w, cache.rerank_lookup(i), chunk_doc, p)
    return evaluate(preds, {qids[i]: gold[qids[i]] for i in range(lo, hi)})


def ablate(
    cache: CandidateCache,
    chunk_doc: np.ndarray,
    best: ScoringParams,
    gold: dict[str, Sequence[str]],
    qids: Sequence[str],
) -> dict[str, Metrics]:
    """Metrics of each retriever alone, their fusion, and fusion + reranker, under the tuned params."""
    fused = [1.0, best.w_ft1, best.w_bm25]
    variants = {
        "BM25": ([0.0, 0.0, 1.0], 0.0),
        "Bi-encoder, round 1": ([0.0, 1.0, 0.0], 0.0),
        "Bi-encoder, round 2": ([1.0, 0.0, 0.0], 0.0),
        "RRF fusion of the three": (fused, 0.0),
        "Fusion + reranker": (fused, best.alpha),
    }
    return {
        name: score_params(cache, chunk_doc, replace(best, alpha=alpha), gold, qids, 0, len(qids), weights)
        for name, (weights, alpha) in variants.items()
    }


def _suggest_zeroable(trial, name: str, low: float, high: float) -> float:
    """Sample exactly 0.0 half the time so "layer off" is reachable by the sampler."""
    if not trial.suggest_categorical(f"{name}_on", [False, True]):
        return 0.0
    return trial.suggest_float(name, low, high)


def suggest_params(
    trial, max_top_k: int = 500, max_rerank_top_n: int = 40, enable_rerank: bool = True
) -> ScoringParams:
    """Search space; every layer weight can be sampled as exactly zero."""
    top_k_choices = [k for k in (100, 200, 300, 500, 800, 1000) if k <= max_top_k] or [max_top_k]
    return ScoringParams(
        w_ft1=_suggest_zeroable(trial, "w_ft1", 0.0, 1.5),
        w_bm25=_suggest_zeroable(trial, "w_bm25", 0.0, 1.5),
        rrf_k=trial.suggest_int("rrf_k", 10, 120),
        dense_top_k=trial.suggest_categorical("dense_top_k", top_k_choices),
        bm25_top_k=trial.suggest_categorical("bm25_top_k", top_k_choices),
        pool_lambda=_suggest_zeroable(trial, "pool_lambda", 0.0, 0.8),
        alpha=_suggest_zeroable(trial, "alpha", 0.0, 1.0) if enable_rerank else 0.0,
        rerank_top_n=trial.suggest_int("rerank_top_n", 10, max_rerank_top_n),
        chunks_per_doc=trial.suggest_categorical("chunks_per_doc", [1, 2, 3, 4]),
        final_k=5,
    )


def select_best(
    trial_rows: Sequence[dict],
    cache: CandidateCache,
    chunk_doc: np.ndarray,
    gold: dict[str, Sequence[str]],
    qids: Sequence[str],
    tune_size: int,
    top_n: int = 20,
    recall_tolerance: float = 0.0015,
) -> dict:
    """Pick among the top trials on the tune split, then score the winner once on verify."""
    ranked = sorted(trial_rows, key=lambda r: r["value"], reverse=True)[:top_n]
    if not ranked:
        raise ValueError("no completed trials")

    scored = []
    for row in ranked:
        p = ScoringParams.from_json(row["params"])
        scored.append((row["number"], p, score_params(cache, chunk_doc, p, gold, qids, 0, tune_size)))

    # Recall gaps under ~one query are noise, so rank quality decides inside that band.
    max_recall = max(m.recall for _, _, m in scored)
    band = [s for s in scored if s[2].recall >= max_recall - recall_tolerance]
    number, p, m_tune = max(
        band, key=lambda s: (s[2].map, s[2].mrr, s[2].hit_at_1, s[2].recall, s[2].precision)
    )
    m_ver = score_params(cache, chunk_doc, p, gold, qids, tune_size, len(qids))
    n_t, n_v = m_tune.num_questions, m_ver.num_questions
    best = {
        "number": number,
        "params": p.to_json(),
        "recall_tune": m_tune.recall,
        "recall_verify": m_ver.recall,
        "recall_all": (m_tune.recall * n_t + m_ver.recall * n_v) / max(n_t + n_v, 1),
        "precision_all": (m_tune.precision * n_t + m_ver.precision * n_v) / max(n_t + n_v, 1),
    }
    print(
        f"[tune] trial #{number}  tune {best['recall_tune']:.4f} | "
        f"verify {best['recall_verify']:.4f} | all {best['recall_all']:.4f}"
    )
    return best


def run_study(
    cache: CandidateCache,
    chunk_doc: np.ndarray,
    gold: dict[str, Sequence[str]],
    qids: Sequence[str],
    tune_size: int,
    n_trials: int = 800,
    storage: str | None = None,
    study_name: str = "legalir",
    top_n: int = 20,
    show_progress: bool = True,
    seed: int = 42,
    max_rerank_top_n: int = 40,
    recall_tolerance: float = 0.0015,
    precision_weight: float = 1e-4,
) -> dict:
    """Run (or resume) an Optuna study and return the selected configuration."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    max_top_k = int(cache.dense_ids.shape[1])

    def objective(trial):
        p = suggest_params(trial, max_top_k, max_rerank_top_n, enable_rerank=cache.has_reranker)
        m = score_params(cache, chunk_doc, p, gold, qids, 0, tune_size)
        return m.recall + precision_weight * m.precision

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress)
    rows = [
        {
            "number": t.number,
            "value": t.value,
            "params": ScoringParams.from_json({**t.params, "final_k": 5}).to_json(),
        }
        for t in study.trials
        if t.value is not None
    ]
    return select_best(
        rows, cache, chunk_doc, gold, qids, tune_size, top_n=top_n, recall_tolerance=recall_tolerance
    )


def save_best_config(result: dict, path) -> None:
    Path(path).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[tune] wrote {path}")


def load_best_config(path) -> ScoringParams:
    return ScoringParams.from_json(json.loads(Path(path).read_text(encoding="utf-8"))["params"])
