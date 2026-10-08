"""Extract issuer, number, type and subject from the standard header of a Vietnamese legal document."""

from __future__ import annotations

import re
from dataclasses import dataclass

HEADER_SCAN_CHARS = 4000

NUMBER_RE = re.compile(r"Số:\s*([0-9A-Za-zÀ-ỹ\-\./]+)")

DOC_TYPES = [
    "NGHỊ QUYẾT LIÊN TỊCH", "THÔNG TƯ LIÊN TỊCH", "QUYẾT ĐỊNH", "NGHỊ ĐỊNH",
    "THÔNG TƯ", "NGHỊ QUYẾT", "CHỈ THỊ", "CÔNG VĂN", "LUẬT", "PHÁP LỆNH",
    "HIẾN PHÁP", "SẮC LỆNH", "KẾ HOẠCH", "THÔNG BÁO", "HƯỚNG DẪN", "CÔNG ĐIỆN",
    "BÁO CÁO", "ĐIỀU LỆ", "QUY CHẾ", "QUY ĐỊNH", "TIÊU CHUẨN", "QUY CHUẨN",
]
DOC_TYPE_RE = re.compile(
    r"^[ \t]*(" + "|".join(DOC_TYPES) + r")[ \t]*$", re.MULTILINE
)

# The number suffix is more reliable than scanning lines, since appendix titles can precede the real type line.
NUMBER_SUFFIX_TO_TYPE = [
    ("TTLT", "THÔNG TƯ LIÊN TỊCH"),
    ("NQLT", "NGHỊ QUYẾT LIÊN TỊCH"),
    ("UBTVQH", "PHÁP LỆNH"),
    ("QH", "LUẬT"),
    ("NĐ-CP", "NGHỊ ĐỊNH"),
    ("ND-CP", "NGHỊ ĐỊNH"),
    ("TT", "THÔNG TƯ"),
    ("QĐ", "QUYẾT ĐỊNH"),
    ("QD", "QUYẾT ĐỊNH"),
    ("NQ", "NGHỊ QUYẾT"),
    ("CT", "CHỈ THỊ"),
    ("PL", "PHÁP LỆNH"),
    ("HD", "HƯỚNG DẪN"),
    ("KH", "KẾ HOẠCH"),
    ("TB", "THÔNG BÁO"),
    ("BC", "BÁO CÁO"),
]


def type_from_number(number: str) -> str:
    """Infer the document type from its number suffix, e.g. '15/2022/TT-BLĐTBXH' -> 'THÔNG TƯ'."""
    if not number:
        return ""
    tail = number.split("/")[-1].upper()
    for code, doc_type in NUMBER_SUFFIX_TO_TYPE:
        if tail == code or tail.startswith(code + "-") or re.fullmatch(rf"{code}\d*", tail):
            return doc_type
    return ""


# National motto lines; matched loosely because the source breaks them across lines arbitrarily.
BOILERPLATE = re.compile(
    r"CỘNG\s*HÒA|CHỦ\s*NGHĨA\s*VIỆT|^\s*VIỆT\s*NAM\s*$|ĐỘC\s*LẬP|TỰ\s*DO|HẠNH\s*PHÚC"
    r"|^-+$|^_+$",
    re.IGNORECASE,
)

DATELINE_RE = re.compile(r"^.{0,40},?\s*ngày\s+\d", re.MULTILINE)

ISSUE_DATE_RE = re.compile(
    r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})", re.IGNORECASE
)


@dataclass
class DocHeader:
    issuer: str = ""       # "BỘ Y TẾ"
    number: str = ""       # "5868/QĐ-BYT"
    doc_type: str = ""     # "QUYẾT ĐỊNH"
    subject: str = ""      # "QUY ĐỊNH CHỨC NĂNG, NHIỆM VỤ..."
    issue_date: str = ""   # "28/9/2018"

    def as_prefix(self, max_chars: int = 320) -> str:
        """One-line summary prepended to every chunk."""
        parts = []
        head = " ".join(
            x for x in (sentence_case(self.doc_type), self.number) if x
        ).strip()
        if head:
            parts.append(head)
        if self.issuer:
            parts.append(sentence_case(self.issuer))
        if self.subject:
            parts.append(sentence_case(self.subject))
        text = " — ".join(parts) if parts else ""
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > max_chars:
            text = text[:max_chars].rsplit(" ", 1)[0] + "…"
        return text


def sentence_case(text: str) -> str:
    """Turn an ALL CAPS line into sentence case (not title case, which the tokenizer treats differently)."""
    if not text:
        return ""
    letters = [c for c in text if c.isalpha()]
    if letters and all(c.isupper() for c in letters):
        text = text.lower()
        for i, c in enumerate(text):
            if c.isalpha():
                return text[:i] + c.upper() + text[i + 1:]
    return text


def _is_all_caps_line(ln: str) -> bool:
    letters = [c for c in ln if c.isalpha()]
    return len(letters) >= 2 and all(c.isupper() for c in letters)


def _clean_lines(text: str) -> list[str]:
    out = []
    for ln in text.split("\n"):
        ln = ln.strip()
        if not ln or BOILERPLATE.search(ln):
            continue
        if set(ln) <= set("-_= "):
            continue
        out.append(ln)
    return out


def extract_header(passage: str) -> DocHeader:
    head = passage.replace("\r\n", "\n").replace("\r", "\n")[:HEADER_SCAN_CHARS]
    h = DocHeader()

    m = NUMBER_RE.search(head)
    if m:
        h.number = m.group(1).strip().rstrip(".")

    m = ISSUE_DATE_RE.search(head)
    if m:
        h.issue_date = f"{int(m.group(1))}/{int(m.group(2))}/{m.group(3)}"

    lines = _clean_lines(head)

    # Issuer: the first run of ALL CAPS lines before "Số:"; names often wrap over several lines.
    issuer_lines: list[str] = []
    for ln in lines[:8]:
        if ln.startswith("Số:") or DATELINE_RE.match(ln):
            break
        if _is_all_caps_line(ln):
            if ln in DOC_TYPES:
                break
            issuer_lines.append(ln)
        elif issuer_lines:
            break
    # The motto column can bleed into the issuer as "... CỘNG" after text extraction.
    issuer = " ".join(issuer_lines).strip()
    h.issuer = re.split(r"\bCỘNG\b", issuer)[0].strip(" -")

    h.doc_type = type_from_number(h.number)

    m = DOC_TYPE_RE.search(head)
    if m:
        if not h.doc_type:
            h.doc_type = m.group(1)
        # Subject: ALL CAPS lines right after the type line.
        tail = head[m.end():]
        subject_lines: list[str] = []
        for ln in _clean_lines(tail)[:6]:
            if not any(c.isalpha() for c in ln):
                continue
            if _is_all_caps_line(ln):
                subject_lines.append(ln)
            elif subject_lines:
                break
            else:
                break
        h.subject = " ".join(subject_lines).strip()

    return h


def build_prefix(passage: str, fallback_slug: str = "", max_chars: int = 320) -> str:
    """Context prefix for chunks, falling back to the URL slug when no header is found."""
    prefix = extract_header(passage).as_prefix(max_chars)
    if prefix:
        return prefix
    if fallback_slug:
        return re.sub(r"\s+", " ", fallback_slug.replace("-", " ")).strip()[:max_chars]
    return ""
