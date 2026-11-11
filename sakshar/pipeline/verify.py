"""Confidence gating for extracted fields.

Gate logic (any failure -> needs_review):
  * validator rejects the value
  * value isn't (fuzzily) present in the text of the blocks it cites
  * regex and LLM extractors disagree
  * the weakest OCR word confidence in the cited blocks is below threshold
Legal / financial / identity fields use the stricter threshold.
"""
from __future__ import annotations

from rapidfuzz import fuzz

from ..config import get_settings
from ..models import Block, DocType, ExtractedField, FieldCategory
from . import normalize
from .schemas import spec_by_name

STRICT = {FieldCategory.legal, FieldCategory.financial, FieldCategory.identity}


def presence(value: str, texts: list[str]) -> float:
    """0..1: how well `value` appears inside the cited source text."""
    if not value or not texts:
        return 0.0
    v = normalize.ascii_digits(normalize.nfc(value)).lower()
    hay = normalize.ascii_digits(normalize.nfc(" ".join(texts))).lower()
    if v in hay:
        return 1.0
    v2, h2 = v.replace(",", "").replace(" ", ""), hay.replace(",", "").replace(" ", "")
    if v2 and v2 in h2:
        return 1.0
    return fuzz.partial_ratio(v, hay) / 100


def verify_field(f: ExtractedField, blocks: dict[str, Block], doc_type: DocType) -> ExtractedField:
    s = get_settings()
    spec = spec_by_name(doc_type).get(f.name)
    reasons = list(f.review_reasons)  # e.g. extractor disagreement from merge()
    cited = [blocks[i] for i in f.source_block_ids if i in blocks]

    if f.source == "user":
        f.confidence, f.needs_review, f.review_reasons = 1.0, False, []
        return f
    if not f.value:
        f.confidence, f.needs_review = 0.0, True
        f.review_reasons = ["no value found"]
        return f

    # 1. validator
    validator_ok = True
    if spec is not None:
        norm, err = spec.validator(f.value)
        f.normalized = norm
        f.validation_error = err
        if err:
            validator_ok = False
            reasons.append(err)

    # 2. grounding in cited text
    pres = presence(f.value, [b.text for b in cited])
    if not cited:
        reasons.append("no source text cited")
    elif pres < s.fuzzy_presence_threshold:
        reasons.append(f"value not clearly found in source text ({pres:.0%})")

    # 3. OCR confidence of the cited words
    word_confs = [c for b in cited for c in (b.word_confs or [b.conf])]
    min_conf = min(word_confs) if word_confs else 0.0
    threshold = s.review_threshold_strict if f.category in STRICT else s.review_threshold
    if min_conf < threshold:
        reasons.append(f"low OCR confidence {min_conf:.0%}")

    agree = f.source == "both"
    # A single reported score for the UI; gating itself does not depend on it.
    f.confidence = round(min(min_conf, pres, 1.0 if validator_ok else 0.4) * (1.0 if agree else 0.95), 3)
    f.needs_review = bool(reasons)
    f.review_reasons = reasons
    return f


def verify_all(fields: list[ExtractedField], blocks: list[Block], doc_type: DocType) -> list[ExtractedField]:
    bmap = {b.id: b for b in blocks}
    return [verify_field(f, bmap, doc_type) for f in fields]
