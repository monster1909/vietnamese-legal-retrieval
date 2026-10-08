"""Load the document corpus and question sets."""

from __future__ import annotations

import json
import re
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from tqdm.auto import tqdm

from .config import Paths


@dataclass
class Document:
    id: int
    passage: str
    link: str
    name: str  # derived from the link when missing (1,125 of 8,532 docs)


@dataclass
class Question:
    qid: str
    question: str
    answer: list[str] | None  # None for test sets


_SLUG_TAIL = re.compile(r"-\d+\.aspx$")


def name_from_link(link: str) -> str:
    """Derive a slug title from a thuvienphapluat URL."""
    if not link:
        return ""
    slug = link.rstrip("/").split("/")[-1]
    slug = _SLUG_TAIL.sub("", slug)
    return slug.replace(".aspx", "")


def ensure_extracted(paths: Paths = Paths()) -> Path:
    """Extract selected-contexts.zip unless already extracted."""
    if paths.contexts_dir.exists() and any(paths.contexts_dir.iterdir()):
        return paths.contexts_dir
    if not paths.contexts_zip.exists():
        raise FileNotFoundError(
            f"Neither {paths.contexts_dir} nor {paths.contexts_zip} exists"
        )
    with zipfile.ZipFile(paths.contexts_zip) as z:
        z.extractall(paths.root)
    if not paths.contexts_dir.exists():
        raise RuntimeError(f"{paths.contexts_dir} missing after extraction")
    return paths.contexts_dir


def iter_documents(paths: Paths = Paths(), show_progress: bool = True) -> Iterator[Document]:
    d = ensure_extracted(paths)
    files = sorted(d.glob("context_*.json"))
    it = tqdm(files, desc="load corpus", disable=not show_progress)
    for f in it:
        with open(f, encoding="utf-8") as fh:
            j = json.load(fh)
        link = j.get("link", "")
        yield Document(
            id=int(j["id"]),
            passage=j.get("passage", ""),
            link=link,
            name=j.get("name") or name_from_link(link),
        )


def load_questions(path: Path) -> list[Question]:
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    out: list[Question] = []
    for qid, v in raw.items():
        ans = v.get("answer")
        out.append(
            Question(
                qid=str(qid),
                question=v["question"],
                answer=[str(a) for a in ans] if ans else None,
            )
        )
    return out


def load_train(paths: Paths = Paths()) -> list[Question]:
    return load_questions(paths.train_json)


def split_train_val(
    questions: list[Question], val_size: int, seed: int
) -> tuple[list[Question], list[Question]]:
    """Deterministic shuffle split so every pipeline step sees the same validation set."""
    import random

    qs = sorted(questions, key=lambda q: q.qid)
    rng = random.Random(seed)
    rng.shuffle(qs)
    return qs[val_size:], qs[:val_size]
