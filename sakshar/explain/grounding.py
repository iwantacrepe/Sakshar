"""Post-hoc grounding check for LLM statements.

A statement passes when it cites at least one existing block and every "hard" value it
mentions (numbers with 3+ digits, dates, amounts, ids) occurs in the text it cites.
If the value exists elsewhere in the document, the citation is repaired; if it exists
nowhere, the statement is marked unverified.
"""
from __future__ import annotations

import re

from ..models import Block, Statement
from ..pipeline import normalize

HARD_VALUE = re.compile(r"\d[\d,./\-]*\d|\d{3,}")


def _norm(text: str) -> str:
    return normalize.ascii_digits(normalize.nfc(text)).lower()


def hard_values(text: str) -> list[str]:
    vals = []
    for m in HARD_VALUE.finditer(_norm(text)):
        v = m.group(0).strip(".,/-")
        if sum(c.isdigit() for c in v) >= 3 or re.search(r"[/\-.]", v):
            vals.append(v)
    return vals


def _contains(hay: str, val: str) -> bool:
    if val in hay:
        return True
    squash = lambda s: re.sub(r"[,\s]", "", s)
    return squash(val) in squash(hay)


def check_statement(st: Statement, blocks: dict[str, Block]) -> Statement | None:
    valid = [i for i in st.cited_block_ids if i in blocks]
    vals = set(hard_values(st.text_en)) | set(hard_values(st.text_local))
    notes = []
    if len(valid) != len(st.cited_block_ids):
        notes.append("removed citations to non-existent blocks")

    if not valid:
        if vals:
            return None  # numeric claim with nothing to back it: drop
        st.cited_block_ids, st.grounded = [], False
        st.note = "no source cited"
        return st

    cited_text = _norm(" ".join(blocks[i].text for i in valid))
    missing = [v for v in vals if not _contains(cited_text, v)]
    for v in list(missing):
        elsewhere = [b.id for b in blocks.values() if _contains(_norm(b.text), v)]
        if elsewhere:
            valid.extend(i for i in elsewhere[:2] if i not in valid)
            missing.remove(v)
            notes.append(f"citation repaired for {v}")
    st.cited_block_ids = valid
    if missing:
        st.grounded = False
        notes.append("value(s) not found in document: " + ", ".join(missing))
    else:
        st.grounded = True
    st.note = "; ".join(notes) or None
    return st


def ground(statements: list[Statement], blocks: list[Block]) -> list[Statement]:
    bmap = {b.id: b for b in blocks}
    out = []
    for st in statements:
        checked = check_statement(st, bmap)
        if checked is not None:
            out.append(checked)
    return out
