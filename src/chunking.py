"""Split legal documents into chunks along Điều > khoản > điểm > paragraph > sentence boundaries."""

from __future__ import annotations

import re

import pandas as pd
from tqdm.auto import tqdm

from .config import CFG, ChunkConfig
from .data import Document
from .doc_header import build_prefix

# A heading is "Điều 5." / "Điều 5:" / "Điều 5-" or a bare "Điều 5" line; references such as
# "Điều 11 Luật Đặc xá" that a line break pushed to line start are followed by words and do not split.
ARTICLE_RE = re.compile(r"^[ \t]*(?=Điều\s+\d+[a-zđ]?\s*(?:[.:\-–]|$))", re.MULTILINE)
SECTION_RE = re.compile(r"^[ \t]*(?=(?:Chương|Mục|Phần)\s+[IVXLCĐ0-9]+\b)", re.MULTILINE)

SECTION_HEADING_RE = re.compile(
    r"^[ \t]*((?:Phần|Chương|Mục)\s+[IVXLCĐ0-9]+[^\n]{0,80})", re.MULTILINE
)
SECTION_LEVEL = {"Phần": 0, "Chương": 1, "Mục": 2}

# Boundaries inside an article, strongest first.
KHOAN_RE = re.compile(r"^[ \t]*(?=\d{1,2}[\.\)]\s)", re.MULTILINE)  # "1." "2)"
DIEM_RE = re.compile(r"^[ \t]*(?=[a-zđ][\.\)]\s)", re.MULTILINE)  # "a)" "b."
PARA_RE = re.compile(r"\n\s*\n")
LINE_RE = re.compile(r"\n")
SENT_RE = re.compile(r"(?<=[.;:])\s+(?=[A-ZĐÀ-Ỹ0-9])")

INNER_SEPARATORS = [KHOAN_RE, DIEM_RE, PARA_RE, LINE_RE, SENT_RE]

_WS = re.compile(r"[ \t]*\r?\n[ \t]*")
_MULTI_NL = re.compile(r"\n{3,}")
_SPACES = re.compile(r"[ \t]{2,}")


def normalize(text: str) -> str:
    """Collapse the stray whitespace and line breaks of raw passages."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS.sub("\n", text)
    text = _MULTI_NL.sub("\n\n", text)
    text = _SPACES.sub(" ", text)
    return text.strip()


def build_breadcrumbs(text: str) -> list[tuple[int, str]]:
    """(char offset, "Chương III > Mục 1") markers in document order."""
    stack: list[str | None] = [None, None, None]
    out: list[tuple[int, str]] = []
    for m in SECTION_HEADING_RE.finditer(text):
        heading = re.sub(r"\s+", " ", m.group(1)).strip()
        level = SECTION_LEVEL[heading.split()[0]]
        stack[level] = heading
        for i in range(level + 1, 3):
            stack[i] = None
        out.append((m.start(), " > ".join(x for x in stack if x)))
    return out


def breadcrumb_at(markers: list[tuple[int, str]], pos: int) -> str:
    """Breadcrumb in effect at offset pos."""
    found = ""
    for start, crumb in markers:
        if start > pos:
            break
        found = crumb
    return found


def _split_keep_pos(pattern: re.Pattern, text: str) -> list[tuple[int, str]]:
    """Like _split_keep but also returns each piece's start offset."""
    spans = [(m.start(), m.end()) for m in pattern.finditer(text)]
    if not spans:
        return [(0, text)] if text.strip() else []

    parts: list[tuple[int, str]] = []
    cursor = 0
    for start, end in spans:
        if start > cursor:
            seg = text[cursor:start]
            if seg.strip():
                parts.append((cursor, seg.strip()))
        cursor = end
    if text[cursor:].strip():
        parts.append((cursor, text[cursor:].strip()))
    return parts


def _split_keep(pattern: re.Pattern, text: str) -> list[str]:
    """Split at every match; lookahead patterns keep the heading at the start of the next piece."""
    spans = [(m.start(), m.end()) for m in pattern.finditer(text)]
    if not spans:
        return [text] if text.strip() else []

    parts: list[str] = []
    cursor = 0
    for start, end in spans:
        if start > cursor:
            seg = text[cursor:start].strip()
            if seg:
                parts.append(seg)
        cursor = end
    tail = text[cursor:].strip()
    if tail:
        parts.append(tail)
    return parts


def _hard_window(text: str, max_chars: int, overlap: int) -> list[str]:
    """Last resort: fixed-size overlapping windows."""
    out: list[str] = []
    start = 0
    step = max(1, max_chars - overlap)
    while start < len(text):
        end = min(start + max_chars, len(text))
        piece = text[start:end].strip()
        if piece:
            out.append(piece)
        if end >= len(text):
            break
        start += step
    return out


def _atomize(text: str, max_chars: int, overlap: int) -> list[str]:
    """Split text into pieces of at most max_chars, preferring the strongest boundary."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    for sep in INNER_SEPARATORS:
        parts = _split_keep(sep, text)
        if len(parts) > 1:
            out: list[str] = []
            for p in parts:
                out.extend(_atomize(p, max_chars, overlap))
            return out

    return _hard_window(text, max_chars, overlap)


def _pack(
    atoms: list[tuple[str, str]], max_chars: int, min_chars: int
) -> list[tuple[str, str]]:
    """Merge short consecutive atoms with the same breadcrumb until min_chars is reached."""
    packed: list[tuple[str, str]] = []
    buf, buf_crumb = "", ""
    for a, crumb in atoms:
        if not buf:
            buf, buf_crumb = a, crumb
            continue
        if (
            crumb == buf_crumb
            and len(buf) < min_chars
            and len(buf) + len(a) + 2 <= max_chars
        ):
            buf = f"{buf}\n\n{a}"
        else:
            packed.append((buf, buf_crumb))
            buf, buf_crumb = a, crumb
    if buf:
        packed.append((buf, buf_crumb))
    return packed


def article_heading(part: str, max_chars: int = 200) -> str:
    """Heading of an article part ("Điều 4. ..."), up to its first clause; "" if part is not an article."""
    if not ARTICLE_RE.match(part):
        return ""
    head = KHOAN_RE.split(part, maxsplit=1)[0]
    return " ".join(head.split())[:max_chars]


def chunk_document(
    doc: Document, cfg: ChunkConfig = CFG.chunk
) -> tuple[list[tuple[str, str]], str]:
    """Return ([(body, breadcrumb), ...], document prefix)."""
    text = normalize(doc.passage)
    if not text:
        # 20 documents have empty passages (6 are gold answers); index their title slug instead.
        title = doc.name.replace("-", " ").strip()
        return ([(title, "")], "") if title else ([], "")

    prefix = ""
    if cfg.prepend_title:
        prefix = build_prefix(doc.passage, doc.name, cfg.prefix_max_chars)
    body_budget = max(cfg.max_chars - len(prefix) - 2, cfg.max_chars // 2)

    parts = _split_keep_pos(ARTICLE_RE, text)
    if len(parts) <= 1:
        parts = _split_keep_pos(SECTION_RE, text)
    if not parts:
        parts = [(0, text)]

    markers = build_breadcrumbs(text) if cfg.add_breadcrumb else []

    atoms: list[tuple[str, str]] = []
    for pos, p in parts:
        crumb = breadcrumb_at(markers, pos) if markers else ""
        pieces = _atomize(p, body_budget, cfg.overlap_chars)
        heading = article_heading(p) if cfg.repeat_article_heading and len(pieces) > 1 else ""
        # Pieces after the one holding the heading name their article in the breadcrumb.
        first = 2 if heading and " ".join(pieces[0].split()) == heading else 1
        cont = " > ".join(x for x in (crumb, heading) if x)
        for i, a in enumerate(pieces):
            atoms.append((a, crumb if i < first or not heading else cont))

    bodies = _pack(atoms, body_budget, cfg.min_chars)
    return [(b, c) for b, c in bodies if b.strip()], prefix


def build_chunks(documents, cfg: ChunkConfig = CFG.chunk, show_progress: bool = True) -> pd.DataFrame:
    """Chunk the corpus; `text` carries the prefix for neural models, `raw_text` is the bare body for BM25."""
    rows = []
    for doc in tqdm(documents, desc="chunking", disable=not show_progress):
        bodies, prefix = chunk_document(doc, cfg)
        for i, (body, crumb) in enumerate(bodies):
            head = " | ".join(x for x in (prefix, crumb) if x)
            rows.append((len(rows), doc.id, i, f"{head}\n\n{body}" if head else body, body))
    return pd.DataFrame(rows, columns=["chunk_id", "doc_id", "chunk_index", "text", "raw_text"])
