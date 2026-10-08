"""Cross-encoder reranker (bge-reranker-v2-m3 based)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

from ..config import CFG, ModelConfig
from .dense import get_device


class Reranker:
    def __init__(self, model_name: str | None = None, cfg: ModelConfig = CFG.model):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.cfg = cfg
        self.device = get_device()
        name = model_name or cfg.reranker_model
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            name, dtype=torch.float32 if self.device == "cpu" else torch.float16
        )
        self.model.to(self.device).eval()

    @torch.no_grad()
    def score(
        self,
        pairs: Sequence[tuple[str, str]],
        batch_size: int | None = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Raw relevance logits for (query, passage) pairs."""
        if not pairs:
            return np.empty(0, dtype=np.float32)

        bs = batch_size or self.cfg.rerank_batch_size
        # Length-sorted batches minimise padding; results are scattered back in order.
        order = np.argsort([len(p[1]) for p in pairs])
        out = np.empty(len(pairs), dtype=np.float32)

        rng = range(0, len(order), bs)
        for i in tqdm(rng, desc="rerank", disable=not show_progress):
            sel = order[i : i + bs]
            batch = [list(pairs[j]) for j in sel]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.cfg.rerank_max_length,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc, return_dict=True).logits.view(-1).float()
            out[sel] = logits.cpu().numpy()
        return out
