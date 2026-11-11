"""Hybrid retrieval over document blocks: BM25 (exact tokens) + RapidFuzz (OCR-typo tolerant).

OCR on government forms makes character errors (खसरा read as खसस), which pure BM25
misses entirely; the character-level fuzzy score catches them.
"""
from __future__ import annotations

import re

from rapidfuzz import fuzz

from ..models import Block, ExtractedField
from ..pipeline import normalize

TOKEN = re.compile(r"[\wऀ-ॿ਀-੿]+")


def tokenize(text: str) -> list[str]:
    t = normalize.ascii_digits(normalize.nfc(text)).lower()
    return [w for w in TOKEN.findall(t) if len(w) > 1 or w.isdigit()]


def _minmax(xs: list[float]) -> list[float]:
    lo, hi = min(xs), max(xs)
    return [0.0 if hi == lo else (x - lo) / (hi - lo) for x in xs]


def _fuzzy(q_tokens: list[str], text: str) -> float:
    t = normalize.ascii_digits(normalize.nfc(text)).lower()
    toks = [q for q in q_tokens if len(q) >= 3]
    if not toks or not t:
        return 0.0
    return sum(fuzz.partial_ratio(q, t) for q in toks) / (100 * len(toks))


def search(queries: list[str], blocks: list[Block], fields: list[ExtractedField] | None = None,
           k: int = 20, w_bm25: float = 0.55) -> list[tuple[Block, float]]:
    """Score blocks against one or more phrasings of the question (original + translation)."""
    if not blocks:
        return []
    # field labels/values point back at their source blocks, which bridges English questions
    # ("what is the area?") to Indic-script blocks.
    field_text: dict[str, str] = {}
    for f in fields or []:
        for bid in f.source_block_ids:
            field_text[bid] = field_text.get(bid, "") + f" {f.label} {f.name.replace('_', ' ')} {f.value or ''}"
    docs = [b.text + field_text.get(b.id, "") for b in blocks]
    corpus = [tokenize(d) or ["_"] for d in docs]

    from rank_bm25 import BM25Okapi

    bm25 = BM25Okapi(corpus)
    combined = [0.0] * len(blocks)
    for q in queries:
        qt = tokenize(q)
        if not qt:
            continue
        b = _minmax(list(bm25.get_scores(qt)))
        f = [_fuzzy(qt, d) for d in docs]
        f = _minmax(f) if max(f) > 0 else f
        for i in range(len(blocks)):
            combined[i] = max(combined[i], w_bm25 * b[i] + (1 - w_bm25) * f[i])
    ranked = sorted(zip(blocks, combined), key=lambda x: -x[1])
    return ranked[:k]


def context_blocks(queries: list[str], blocks: list[Block], fields: list[ExtractedField],
                   budget_chars: int = 8000, k: int = 20) -> list[Block]:
    """Small documents go in whole; larger ones get the top-k hits, kept in reading order."""
    if sum(len(b.text) for b in blocks) <= budget_chars:
        return blocks
    hits = {b.id for b, _ in search(queries, blocks, fields, k=k)}
    for f in fields:
        hits.update(f.source_block_ids)
    return [b for b in blocks if b.id in hits]
