"""Field extraction: deterministic regex/label heuristics + LLM structured extraction, merged."""
from __future__ import annotations

import logging
import re

from rapidfuzz import fuzz

from .. import pii
from ..models import BBox, Block, DocType, ExtractedField, FieldCategory
from . import normalize
from .schemas import FieldSpec, schema_for

log = logging.getLogger(__name__)

REF_RE = re.compile(r"[A-Za-z]{0,6}[/\-]?\d[\dA-Za-z/\-.]{0,30}")


def _alias_hit(block: Block, spec: FieldSpec) -> tuple[float, int]:
    """(score, matched alias length) for a block's key (or start of text) against a spec.

    An exact key beats containment; fuzzy matching is only used for long aliases and with a
    high bar, because short Indic labels differ by a single akshara (खाता संख्या vs आधार संख्या).
    """
    key = normalize.nfc((block.key or block.text[:40]).lower()).strip(" :")
    best = (0.0, 0)
    for a in spec.aliases:
        a = normalize.nfc(a.lower())
        if a == key:
            hit = (100.0, len(a))
        elif key.startswith(a):
            hit = (95.0 if block.key else 88.0, len(a))
        elif a in key:
            hit = ((95.0 if block.key else 85.0) - min(15.0, (len(key) - len(a)) / 3), len(a))
        elif len(a) >= 7 and fuzz.partial_ratio(a, key) >= 92:
            hit = (80.0, len(a))
        else:
            continue
        best = max(best, hit)
    return best


def _label_match(block: Block, spec: FieldSpec) -> float:
    return _alias_hit(block, spec)[0]


def _is_header(block: Block, spec: FieldSpec) -> bool:
    """A table header cell: label text only, no value on the same line."""
    if block.value or block.col is None:
        return False  # headers live inside a ruled table column
    t = normalize.nfc(normalize.ascii_digits(block.text).lower()).strip(" :")
    for a in spec.aliases:
        a = normalize.nfc(a.lower())
        i = t.find(a)
        # the label must lead the cell, allowing a little OCR junk before it
        # ("राम प्रसाद पुत्र श्याम लाल" merely contains "पुत्र"), and a header carries no value
        # of its own ("ਇੰਤਕਾਲ ਨੰਬਰ 4521" is label + value)
        if 0 <= i <= 6 and len(t) <= len(a) + 30 and not re.search(r"\d", t[i + len(a):]):
            return True
    return False


RELATION_RE = re.compile(
    r"(?:पुत्र|पुत्री|पत्नी|पिता|पति|s/o|d/o|w/o|son of|wife of|ਪੁੱਤਰ|ਪੁੱਤਰੀ|ਪਤਨੀ|ਪਿਤਾ)\s+"
    r"(?P<name>.+?)(?=\s+(?:निवासी|ਵਾਸੀ|resident|r/o)\b|$)",
    re.IGNORECASE,
)


def relation_names(blocks: list[Block]) -> list[ExtractedField]:
    """Father's/husband's name from 'X पुत्र Y' / 'X ਪਤਨੀ Y' patterns inside owner cells."""
    out = []
    for b in blocks:
        m = RELATION_RE.search(normalize.nfc(b.text))
        if m and len(m.group("name").strip()) >= 2:
            out.append(ExtractedField(name="father_name", label="Father's / husband's name",
                                      value=m.group("name").strip(), category=FieldCategory.identity,
                                      source_block_ids=[b.id], bbox=b.bbox, source="regex"))
    return out


def _column_cells(header: Block, blocks: list[Block]) -> list[list[Block]]:
    """Cells below a header in the same column, grouped by table row (multi-line cells stay together)."""
    below = [
        c for c in blocks
        if c is not header and c.page == header.page and c.bbox.y1 >= header.bbox.y2 - 0.3 * header.bbox.h
        and c.col == header.col
    ]
    below.sort(key=lambda c: c.bbox.y1)
    cells: list[list[Block]] = []
    prev_y2 = header.bbox.y2
    for c in below:
        if c.bbox.y1 - prev_y2 > 3.5 * max(header.bbox.h, 0.01):
            break  # left the table
        if header.row is not None and c.row is None:
            break  # below the last ruling line of the table
        if cells and c.row is not None and c.row == cells[-1][-1].row:
            cells[-1].append(c)
        elif cells and c.row is None and cells[-1][-1].row is None and c.bbox.y1 - prev_y2 < 0.5 * c.bbox.h:
            cells[-1].append(c)
        else:
            cells.append([c])
        prev_y2 = c.bbox.y2
    return cells


def _value_for(spec: FieldSpec, raw: str) -> str | None:
    raw = raw.strip(" :-–|")
    if not raw:
        return None
    cat = spec.category
    if spec.validator.__name__ == "v_date":
        found = normalize.find_dates(raw)
        return found[0][0] if found else None
    if cat == FieldCategory.financial:
        found = normalize.find_amounts(raw)
        if found:
            return found[0][0]
        m = re.search(r"\d[\d,]*(?:\.\d+)?", normalize.ascii_digits(raw))
        return m.group(0) if m else None
    if spec.validator.__name__ == "v_ref":
        t = normalize.ascii_digits(raw)
        for m in REF_RE.finditer(t):
            if not any(m.group(0) in d or d in m.group(0) for d, _ in normalize.find_dates(t)):
                return m.group(0)  # first reference-like token that is not a date
        return None
    return raw


def regex_extract(blocks: list[Block], doc_type: DocType) -> list[ExtractedField]:
    out: list[ExtractedField] = []
    specs = [s for s in schema_for(doc_type) if s.name not in ("aadhaar", "pan")]
    # Each label belongs to the field with the longest matching alias:
    # "भोगवटादार वर्ग" is the occupancy class, not the owner ("भोगवटादार").
    owner_len = {b.id: max((_alias_hit(b, s)[1] for s in specs), default=0) for b in blocks}
    for spec in specs:
        cands: list[tuple[float, list[Block], str]] = []
        for b in blocks:
            score, alen = _alias_hit(b, spec)
            if alen and alen < owner_len[b.id]:
                continue
            if _is_header(b, spec):
                for cell in _column_cells(b, blocks):
                    val = _value_for(spec, " ".join(c.text for c in cell))
                    if val:
                        cands.append((95.0, cell, val))
                continue
            if score < 80:
                continue
            raw = b.value if b.value else b.text
            if not b.value:
                # strip the matched alias from the line, keep the remainder as value
                for a in sorted(spec.aliases, key=len, reverse=True):
                    i = normalize.nfc(raw).lower().find(normalize.nfc(a).lower())
                    if i >= 0:
                        raw = raw[i + len(a):]
                        break
            val = _value_for(spec, raw)
            if val:
                cands.append((score, [b], val))
        cands.sort(key=lambda c: -c[0])  # stable: keeps reading order among equal scores
        seen: set[str] = set()
        for score, src, val in (cands if spec.multi else cands[:1]):
            if val in seen:
                continue
            seen.add(val)
            out.append(ExtractedField(
                name=spec.name, label=spec.label, value=val, category=spec.category,
                source_block_ids=[b.id for b in src], bbox=BBox.union([b.bbox for b in src]), source="regex",
            ))
    if any(s.name == "father_name" for s in specs) and not any(f.name == "father_name" for f in out):
        out.extend(relation_names(blocks))
    out.extend(_ids(blocks))
    return out


def _ids(blocks: list[Block]) -> list[ExtractedField]:
    """Aadhaar / PAN anywhere on the page (checksum-gated so random 12-digit numbers don't count)."""
    out = []
    for b in blocks:
        t = normalize.ascii_digits(b.text)
        for m in pii.AADHAAR_RE.finditer(t):
            num = "".join(m.groups())
            near_label = any(k in b.text.lower() for k in ("aadhaar", "आधार", "ਆਧਾਰ", "uid"))
            if pii.verhoeff_valid(num) or near_label:
                out.append(ExtractedField(name="aadhaar", label="Aadhaar number", value=num,
                                          category=FieldCategory.identity, source_block_ids=[b.id],
                                          bbox=b.bbox, source="regex"))
        for m in pii.PAN_RE.finditer(b.text.upper()):
            out.append(ExtractedField(name="pan", label="PAN", value=m.group(0),
                                      category=FieldCategory.identity, source_block_ids=[b.id],
                                      bbox=b.bbox, source="regex"))
    return out


def merge(regex_fields: list[ExtractedField], llm_fields: list[ExtractedField],
          blocks: list[Block], doc_type: DocType) -> list[ExtractedField]:
    """Combine both extractors. Agreement is recorded via source='both'."""
    bmap = {b.id: b for b in blocks}
    specs = {s.name: s for s in schema_for(doc_type)}
    merged: list[ExtractedField] = []
    used_llm: set[int] = set()
    for rf in regex_fields:
        match_idx = None
        for i, lf in enumerate(llm_fields):
            if i in used_llm or lf.name != rf.name:
                continue
            if _same(rf.value, lf.value):
                match_idx = i
                break
        if match_idx is not None:
            used_llm.add(match_idx)
            rf.source = "both"
        else:
            other = next((lf for i, lf in enumerate(llm_fields) if i not in used_llm and lf.name == rf.name), None)
            if other is not None:
                rf.review_reasons.append(f"extractors disagree (LLM read '{other.value}')")
        merged.append(rf)
    for i, lf in enumerate(llm_fields):
        if i in used_llm or lf.name not in specs:
            continue
        if not spec_multi(specs[lf.name]) and any(m.name == lf.name for m in merged):
            continue
        lf.source = "llm"
        lf.label = specs[lf.name].label
        lf.category = specs[lf.name].category
        cited = [bmap[c] for c in lf.source_block_ids if c in bmap]
        lf.source_block_ids = [c.id for c in cited]
        lf.bbox = BBox.union([c.bbox for c in cited]) if cited else None
        merged.append(lf)
    return merged


def spec_multi(spec: FieldSpec) -> bool:
    return spec.multi


def _same(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False
    na = normalize.ascii_digits(normalize.nfc(a)).lower().replace(",", "").strip()
    nb = normalize.ascii_digits(normalize.nfc(b)).lower().replace(",", "").strip()
    return na == nb or fuzz.ratio(na, nb) >= 90 or (len(na) > 3 and (na in nb or nb in na))
