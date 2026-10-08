"""Central configuration; defaults are the settings of the final competition run."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Paths:
    root: Path = Path("data")

    @property
    def contexts_dir(self) -> Path:
        return self.root / "selected-contexts"

    @property
    def contexts_zip(self) -> Path:
        return self.root / "selected-contexts.zip"

    @property
    def train_json(self) -> Path:
        return self.root / "train.json"


@dataclass
class ChunkConfig:
    max_chars: int = 3200  # ~1000 tokens of Vietnamese text
    min_chars: int = 400  # shorter articles are merged with the next one
    overlap_chars: int = 320
    # Prefix every chunk with the document number/issuer/title so mid-document articles stay attributable.
    prepend_title: bool = True
    prefix_max_chars: int = 320
    add_breadcrumb: bool = True  # e.g. "Chương III > Mục 1"


@dataclass
class BM25Config:
    k1: float = 0.9
    b: float = 0.75
    min_df: int = 2
    use_bigrams: bool = True  # syllable bigrams approximate Vietnamese word segmentation
    add_docnum_token: bool = True  # lets queries citing a decree number match its body chunks
    bigram_min_ram_gb: float = 0.0  # fall back to unigrams below this much free RAM
    strip_accents: bool = True


@dataclass
class ModelConfig:
    base_model: str = "AITeamVN/Vietnamese_Embedding"
    reranker_model: str = "AITeamVN/Vietnamese_Reranker"
    max_seq_length: int = 1024
    normalize: bool = True
    encode_batch_size: int = 256
    encode_dtype: str = "float16"
    rerank_batch_size: int = 32
    rerank_max_length: int = 2304


@dataclass
class TrainConfig:
    epochs: int = 4
    lr: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "linear"
    optim: str = "adamw_torch_fused"
    bf16: bool = True
    tf32: bool = True
    train_batch_size: int = 128  # effective batch via GradCache
    mini_batch_size: int = 8
    num_hard_negatives: int = 8
    positives_per_gold: int = 2
    fn_threshold: float = 0.95  # drop negatives scoring >= this fraction of the positive
    mine_top_k: int = 120
    val_size: int = 1000
    tune_size: int = 700  # first part of val tunes Optuna, the rest verifies
    seed: int = 42
    logging_steps: int = 10
    dataloader_num_workers: int = 2
    early_stopping: bool = True
    eval_steps: int = 100
    early_stopping_patience_steps: int = 400
    early_stopping_threshold: float = 0.0
    selection_precision_weight: float = 1e-6
    selection_ndcg_weight: float = 1e-10
    eval_questions: int = 500
    eval_corpus_sample_size: int = 40_000
    save_total_limit: int = 1
    cleanup_checkpoints: bool = True
    save_fp16: bool = True


@dataclass
class Config:
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    bm25: BM25Config = field(default_factory=BM25Config)


CFG = Config()
