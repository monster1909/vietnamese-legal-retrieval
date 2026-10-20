"""REST API: POST /search returns the most relevant Vietnamese legal documents for a question."""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.config import CFG
from src.pipeline import Searcher

# Same CPU serving budget as the Gradio demo.
RERANK_TOP_N = int(os.environ.get("RERANK_TOP_N", "10"))
CFG.model.rerank_max_length = int(os.environ.get("RERANK_MAX_LENGTH", "512"))

searcher = Searcher(Path(os.environ.get("ARTIFACTS", "artifacts")))
app = FastAPI(title="Vietnamese Legal Retrieval")


class SearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    k: int = Field(5, ge=1, le=10)
    rerank: bool = False


class Document(BaseModel):
    doc_id: int
    title: str
    link: str
    passage: str


class SearchResponse(BaseModel):
    results: list[Document]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/search")
def search(req: SearchRequest) -> SearchResponse:
    hits = searcher.search(req.question, req.k, rerank=req.rerank, rerank_top_n=RERANK_TOP_N)
    return SearchResponse(results=[Document(**asdict(h)) for h in hits])
