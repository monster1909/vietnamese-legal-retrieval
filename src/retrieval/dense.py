"""Dense retrieval with a BGE-M3 bi-encoder and exact torch-matmul search (no FAISS needed)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from ..config import CFG, ModelConfig

_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": None}


def enable_tf32() -> None:
    """Allow TF32 tensor cores for leftover fp32 matmuls."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def encoder_model_kwargs(device: str, cfg: ModelConfig = CFG.model) -> dict:
    """SentenceTransformer model_kwargs; half precision on GPU (CUDA or MPS)."""
    if device == "cpu":
        return {}
    if cfg.encode_dtype not in _DTYPES:
        raise ValueError(
            f"Unknown encode_dtype='{cfg.encode_dtype}'. "
            f"Valid options: {', '.join(sorted(_DTYPES.keys()))}"
        )
    dtype = _DTYPES[cfg.encode_dtype]
    kw: dict = {"attn_implementation": "sdpa"} if device == "cuda" else {}
    if dtype is not None:
        kw["torch_dtype"] = dtype
    return kw


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_encoder(model_path: str | Path | None = None, cfg: ModelConfig = CFG.model):
    """Load the base encoder, or a fine-tuned checkpoint when model_path is given."""
    from sentence_transformers import SentenceTransformer

    enable_tf32()
    device = get_device()
    name = str(model_path) if model_path else cfg.base_model
    model = SentenceTransformer(
        name, device=device, model_kwargs=encoder_model_kwargs(device, cfg)
    )
    model.max_seq_length = cfg.max_seq_length
    return model


def encode_texts(
    model,
    texts: Sequence[str],
    cfg: ModelConfig = CFG.model,
    batch_size: int | None = None,
    show_progress: bool = True,
) -> np.ndarray:
    """Encode to an L2-normalised fp16 matrix (n, dim), cast on device to save host memory."""
    out = model.encode(
        list(texts),
        batch_size=batch_size or cfg.encode_batch_size,
        normalize_embeddings=cfg.normalize,
        convert_to_tensor=True,
        show_progress_bar=show_progress,
    )
    return out.to(torch.float16).cpu().numpy()


def encode_corpus(
    model,
    texts: Sequence[str],
    out_path: Path,
    cfg: ModelConfig = CFG.model,
    show_progress: bool = True,
    force: bool = False,
) -> np.ndarray:
    """Encode and save the corpus; reuse an existing file with matching row count unless force."""
    out_path = Path(out_path)
    if out_path.exists() and not force:
        emb = np.load(out_path)
        if len(emb) == len(texts):
            return emb
        print(f"[dense] {out_path.name} has {len(emb)} rows, expected {len(texts)}; re-encoding")
    emb = encode_texts(model, texts, cfg, show_progress=show_progress)
    np.save(out_path, emb)
    return emb


class DenseIndex:
    """Exact inner-product search, sharded to bound memory."""

    def __init__(self, embeddings: np.ndarray, device: str | None = None, shard_size: int = 200_000):
        self.device = device or get_device()
        self.shard_size = shard_size
        self.n = len(embeddings)
        # Embeddings are saved as fp16; upcasting on CPU only doubles RAM.
        self.shards = [
            torch.from_numpy(np.ascontiguousarray(embeddings[i : i + shard_size]))
            .to(self.device, dtype=torch.float16)
            for i in range(0, self.n, shard_size)
        ]

    @torch.no_grad()
    def search(self, query_emb: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (indices, scores), each of shape (n_queries, top_k)."""
        dtype = self.shards[0].dtype
        q = torch.from_numpy(np.ascontiguousarray(query_emb)).to(self.device, dtype=dtype)

        all_scores: list[torch.Tensor] = []
        all_idx: list[torch.Tensor] = []
        offset = 0
        for shard in self.shards:
            sims = q @ shard.T
            k = min(top_k, sims.shape[1])
            s, i = torch.topk(sims, k, dim=1)
            all_scores.append(s.float())
            all_idx.append(i + offset)
            offset += shard.shape[0]

        scores = torch.cat(all_scores, dim=1)
        idx = torch.cat(all_idx, dim=1)
        k = min(top_k, scores.shape[1])
        s, pos = torch.topk(scores, k, dim=1)
        i = torch.gather(idx, 1, pos)
        return i.cpu().numpy(), s.cpu().numpy()

    def search_batched(
        self, query_emb: np.ndarray, top_k: int, batch_size: int = 256, show_progress: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        idxs, scores = [], []
        rng = range(0, len(query_emb), batch_size)
        for i in tqdm(rng, desc="dense search", disable=not show_progress):
            a, b = self.search(query_emb[i : i + batch_size], top_k)
            idxs.append(a)
            scores.append(b)
        return np.concatenate(idxs), np.concatenate(scores)
