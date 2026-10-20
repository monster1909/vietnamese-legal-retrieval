"""Gradio demo: ask a question, get the most relevant Vietnamese legal documents."""

from __future__ import annotations

import os
import re
import time
from functools import lru_cache
from pathlib import Path

import gradio as gr

from src.config import CFG
from src.pipeline import BEST_CONFIG, Searcher

ARTIFACTS = Path(os.environ.get("ARTIFACTS", "artifacts"))
ARTIFACTS_REPO = os.environ.get("ARTIFACTS_REPO")  # private HF repo with the artifacts, for Spaces
# Reranking budget for CPU serving; the competition setup reranked ~100 docs at 2304 tokens.
RERANK_TOP_N = int(os.environ.get("RERANK_TOP_N", "10"))
CFG.model.rerank_max_length = int(os.environ.get("RERANK_MAX_LENGTH", "512"))

if ARTIFACTS_REPO and not (ARTIFACTS / BEST_CONFIG).exists():
    from huggingface_hub import snapshot_download

    snapshot_download(
        ARTIFACTS_REPO, local_dir=ARTIFACTS, token=os.environ.get("HF_TOKEN"),
        ignore_patterns=["cache_val.npz", "optuna.db"],
    )
searcher = Searcher(ARTIFACTS)

EXAMPLES = [
    "Thủ tục đăng ký kết hôn với người nước ngoài gồm những giấy tờ gì?",
    "Người lao động được nghỉ phép năm bao nhiêu ngày?",
    "Không đội mũ bảo hiểm khi đi xe máy bị phạt bao nhiêu tiền?",
    "Điều kiện thành lập doanh nghiệp tư nhân là gì?",
    "Thời hạn cấp lại căn cước công dân là bao lâu?",
]


# Separator lines and the national motto that open every legal document.
NOISE = re.compile(
    r"[-_=*.]{3,}|CỘNG\s*H(?:OÀ|ÒA)\s*XÃ\s*HỘI\s*CHỦ\s*NGHĨA\s*VIỆT\s*NAM|Độc\s*lập\s*-\s*Tự\s*do\s*-\s*Hạnh\s*phúc",
    re.IGNORECASE,
)


def render_hit(i: int, hit) -> str:
    """Markdown card: number as heading, issuer and subject below, explicit source link, clean excerpt."""
    head, *rest = [part.strip() for part in hit.title.split(" — ")]
    lines = [f"### {i}. {head}"]
    if rest:
        lines.append(" · ".join(rest))
    if hit.link:
        lines.append(f"🔗 [View original document]({hit.link})")
    passage = " ".join(NOISE.sub(" ", hit.passage).split())
    lines.append(f"> {passage[:600]}{'…' if len(passage) > 600 else ''}")
    return "\n\n".join(lines)


@lru_cache(maxsize=512)
def _search(query: str, k: int, rerank: bool):
    return tuple(searcher.search(query, k, rerank=rerank, rerank_top_n=RERANK_TOP_N))


def search(query: str, k: float, mode: str) -> str:
    """Markdown list of the top-k documents for a question."""
    query = " ".join(query.split())
    if not query:
        return "Enter a question."
    start = time.perf_counter()
    hits = _search(query, int(k), mode == "Rerank")
    elapsed = time.perf_counter() - start
    cards = [render_hit(i, h) for i, h in enumerate(hits, 1)]
    return f"*{len(hits)} documents in {elapsed:.2f}s*\n\n" + "\n\n---\n\n".join(cards)


with gr.Blocks(title="Vietnamese Legal Retrieval") as demo:
    gr.Markdown(
        "# Vietnamese Legal Retrieval\n"
        "Ask a question in Vietnamese and get the most relevant legal documents from a corpus of 8.5k "
        "laws, decrees and circulars. **Fast** fuses two fine-tuned encoders with BM25; **Rerank** adds a "
        "cross-encoder (slower on CPU). [Source code](https://github.com/monster1909/vietnamese-legal-retrieval)"
    )
    with gr.Row():
        query = gr.Textbox(label="Question", placeholder=EXAMPLES[0], scale=4)
        button = gr.Button("Search", variant="primary", scale=1)
    with gr.Row():
        k = gr.Slider(1, 10, value=5, step=1, label="Number of documents")
        mode = gr.Radio(["Fast", "Rerank"], value="Fast", label="Mode")
    output = gr.Markdown()
    gr.Examples(EXAMPLES, inputs=query)
    button.click(search, [query, k, mode], output, api_name="search")
    query.submit(search, [query, k, mode], output)

if __name__ == "__main__":
    demo.launch()
