"""Sparse BM25 over chunks with a tokenizer that keeps document numbers (e.g. 75/2017/NĐ-CP) whole."""

from __future__ import annotations

import re
import unicodedata
from array import array
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from tqdm.auto import tqdm

from ..config import CFG, BM25Config

DOCNUM_RE = re.compile(r"\d{1,4}/(?:\d{4}/)?[A-Za-zÀ-ỹ][A-Za-zÀ-ỹ\-]{1,25}")
WORD_RE = re.compile(r"[0-9a-zà-ỹđ]+", re.IGNORECASE)

VIETNAMESE_STOPWORDS = {
    "và", "của", "có", "được", "cho", "các", "trong", "là", "với", "này",
    "khi", "theo", "về", "đã", "những", "một", "tại", "từ", "để", "đến",
    "hoặc", "như", "thì", "sẽ", "bị", "phải", "còn", "cũng", "nếu", "mà",
}


def _strip_accents(s: str) -> str:
    """Remove diacritics, folding đ/Đ explicitly since NFD does not decompose it."""
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.replace("Đ", "D").replace("đ", "d")
    return s


def docnum_token(number: str) -> str:
    """Normalise a document number into one accent-free, upper-case token."""
    s = number.strip()
    if not s:
        return ""
    s = re.sub(r"\s+", "", s)
    s = _strip_accents(s).upper().rstrip(".")
    return "§" + s


def tokenize(
    text: str,
    remove_stopwords: bool = True,
    use_bigrams: bool = False,
    strip_accents: bool = True,
) -> list[str]:
    """Tokenize with whole document numbers, optional accent stripping and syllable bigrams."""
    tokens: list[str] = []
    lowered = text.lower()

    spans = []
    for m in DOCNUM_RE.finditer(text):
        tokens.append(docnum_token(m.group(0)))
        spans.append(m.span())

    # Blank out matched numbers with "\n" so they are not re-tokenized and bigrams do not span them.
    if spans:
        buf = list(lowered)
        for s, e in spans:
            for i in range(s, e):
                buf[i] = "\n"
        lowered = "".join(buf)

    words: list[str] = []
    word_positions: list[tuple[int, int]] = []
    for m in WORD_RE.finditer(lowered):
        w = m.group(0)
        keep = not (remove_stopwords and w in VIETNAMESE_STOPWORDS)
        normalized = _strip_accents(w) if strip_accents else w
        if keep:
            tokens.append(normalized)
        if use_bigrams:
            words.append(normalized)
            word_positions.append((m.start(), m.end()))

    if use_bigrams:
        for i in range(len(words) - 1):
            gap_start = word_positions[i][1]
            gap_end = word_positions[i + 1][0]
            if "\n" not in lowered[gap_start:gap_end]:
                tokens.append(f"{words[i]}_{words[i + 1]}")

    return tokens


@dataclass
class BM25Index:
    """BM25 Okapi that keeps raw term frequencies so k1/b can change without re-tokenizing."""

    vocab: dict[str, int]
    tf: sp.csr_matrix          # (num_docs, vocab_size) raw counts
    idf: np.ndarray            # (vocab_size,)
    doc_len: np.ndarray        # (num_docs,)
    k1: float = 1.2
    b: float = 0.75
    use_bigrams: bool = False
    strip_accents: bool = True
    weights: sp.csc_matrix | None = None   # derived by reweight()

    def __post_init__(self) -> None:
        if self.weights is None:
            self.reweight(self.k1, self.b)

    @property
    def num_docs(self) -> int:
        return self.tf.shape[0]

    def reweight(self, k1: float | None = None, b: float | None = None) -> None:
        """Recompute BM25 weights from tf in place."""
        self.k1 = self.k1 if k1 is None else float(k1)
        self.b = self.b if b is None else float(b)

        avgdl = float(self.doc_len.mean()) or 1.0
        denom_per_doc = (
            self.k1 * (1 - self.b + self.b * self.doc_len / avgdl)
        ).astype(np.float32)

        vals = self.tf.data.astype(np.float32, copy=True)
        indptr = self.tf.indptr
        for i in range(self.tf.shape[0]):
            s, e = indptr[i], indptr[i + 1]
            if s != e:
                vals[s:e] = vals[s:e] * (self.k1 + 1) / (vals[s:e] + denom_per_doc[i])
        vals *= self.idf[self.tf.indices]

        csr = sp.csr_matrix((vals, self.tf.indices, self.tf.indptr), shape=self.tf.shape, copy=False)
        # Queries read columns (terms), so CSC is the right layout.
        self.weights = csr.tocsc()

    def search(self, query: str, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (indices, scores) sorted by descending score."""
        toks = tokenize(
            query,
            use_bigrams=self.use_bigrams,
            strip_accents=self.strip_accents,
        )
        cols = [self.vocab[t] for t in toks if t in self.vocab]
        if not cols:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

        scores = np.zeros(self.num_docs, dtype=np.float32)
        for c in cols:
            start, end = self.weights.indptr[c], self.weights.indptr[c + 1]
            np.add.at(scores, self.weights.indices[start:end], self.weights.data[start:end])

        k = min(top_k, int((scores > 0).sum()))
        if k == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return idx, scores[idx]

    def save(self, path) -> None:
        """Save as compressed npz (smaller and more portable than pickle)."""
        terms = np.array(list(self.vocab.keys()), dtype=object)
        cols = np.fromiter(self.vocab.values(), dtype=np.int64, count=len(self.vocab))
        np.savez_compressed(
            path,
            terms=terms,
            cols=cols,
            tf_data=self.tf.data,
            tf_indices=self.tf.indices,
            tf_indptr=self.tf.indptr,
            tf_shape=np.array(self.tf.shape, dtype=np.int64),
            idf=self.idf,
            doc_len=self.doc_len,
            params=np.array([self.k1, self.b], dtype=np.float64),
            use_bigrams=np.array([self.use_bigrams]),
            strip_accents=np.array([self.strip_accents]),
        )

    @classmethod
    def load(cls, path) -> BM25Index:
        d = np.load(path, allow_pickle=True)
        vocab = {str(t): int(c) for t, c in zip(d["terms"], d["cols"])}
        tf = sp.csr_matrix(
            (d["tf_data"], d["tf_indices"], d["tf_indptr"]), shape=tuple(d["tf_shape"])
        )
        k1, b = d["params"].tolist()
        return cls(
            vocab=vocab,
            tf=tf,
            idf=d["idf"],
            doc_len=d["doc_len"],
            k1=k1,
            b=b,
            use_bigrams=bool(d["use_bigrams"][0]),
            strip_accents=bool(d["strip_accents"][0]) if "strip_accents" in d.files else True,
        )


def _bigrams_fit_in_ram(min_gb: float) -> bool:
    """Bigrams inflate the index ~2.5x, so check free RAM first."""
    try:
        import psutil

        return psutil.virtual_memory().available / 1e9 >= min_gb
    except ImportError:
        import os

        try:
            avail = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1e9
        except (ValueError, OSError):
            return True
        return avail >= min_gb


def build_bm25(
    texts: Sequence[str],
    doc_numbers: Sequence[str] | None = None,
    cfg: BM25Config | None = None,
    show_progress: bool = True,
) -> BM25Index:
    """Build the index; doc_numbers[i] is the number of the document that owns chunk i."""
    cfg = cfg if cfg is not None else CFG.bm25

    use_bigrams = cfg.use_bigrams
    if use_bigrams and not _bigrams_fit_in_ram(cfg.bigram_min_ram_gb):
        print(
            f"[bm25] less than {cfg.bigram_min_ram_gb}GB RAM free, falling back to unigrams"
        )
        use_bigrams = False

    # array.array keeps ~86M non-zeros at 4 bytes each instead of GBs of Python ints.
    vocab: dict[str, int] = {}
    indices = array("i")
    data = array("f")
    indptr = array("q", [0])
    doc_len = np.zeros(len(texts), dtype=np.float32)

    for i, t in enumerate(tqdm(texts, desc="bm25 tokenize", disable=not show_progress)):
        toks = tokenize(
            t,
            use_bigrams=use_bigrams,
            strip_accents=cfg.strip_accents,
        )
        if cfg.add_docnum_token and doc_numbers is not None:
            num = docnum_token(doc_numbers[i])
            if num:
                toks.append(num)
        doc_len[i] = len(toks)
        counts: dict[int, int] = {}
        for tok in toks:
            j = vocab.get(tok)
            if j is None:
                j = len(vocab)
                vocab[tok] = j
            counts[j] = counts.get(j, 0) + 1
        indices.extend(counts.keys())
        data.extend(counts.values())
        indptr.append(len(indices))

    n_docs, n_vocab = len(texts), len(vocab)
    tf = sp.csr_matrix(
        (
            np.frombuffer(data, dtype=np.float32),
            np.frombuffer(indices, dtype=np.int32),
            np.frombuffer(indptr, dtype=np.int64),
        ),
        shape=(n_docs, n_vocab),
        copy=True,
    )
    del indices, data, indptr

    df = np.bincount(tf.indices, minlength=n_vocab)
    keep = df >= cfg.min_df
    if not keep.all():
        keep_idx = np.flatnonzero(keep)
        remap = -np.ones(n_vocab, dtype=np.int64)
        remap[keep_idx] = np.arange(len(keep_idx))
        vocab = {t: int(remap[j]) for t, j in vocab.items() if remap[j] >= 0}
        tf = tf[:, keep_idx].tocsr()
        df = df[keep_idx]

    idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
    return BM25Index(
        vocab=vocab, tf=tf, idf=idf, doc_len=doc_len,
        k1=cfg.k1, b=cfg.b, use_bigrams=use_bigrams,
        strip_accents=cfg.strip_accents,
    )
