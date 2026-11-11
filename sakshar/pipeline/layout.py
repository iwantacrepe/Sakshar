"""Layout analysis: recursive XY-cut into regions (columns / table cells), then line
clustering inside each region, producing citable blocks b0, b1, ...

Pure y-overlap clustering across a whole page merges neighbouring table columns
(e.g. an owner's name with the adjacent survey number on a 7/12 extract). XY-cut
first separates regions at whitespace gaps, so each cell stays its own block.
"""
from __future__ import annotations

import re
import statistics

from ..models import BBox, Block, OCRWord

# Colon is the real separator on forms; hyphens occur inside terms (दाखिल-खारिज), so a dash
# only counts when surrounded by spaces ("Name - Ram").
KV_COLON = re.compile(r"^\s*(?P<key>[^:：]{2,60}?)\s*[:：]\s*(?P<val>.+?)\s*$")
KV_DASH = re.compile(r"^\s*(?P<key>.{2,40}?)\s+[\-–]\s+(?P<val>.+?)\s*$")


def split_kv(text: str) -> tuple[str, str] | None:
    m = KV_COLON.match(text) or KV_DASH.match(text)
    if not m or re.fullmatch(r"[\d\s/.\-]+", text):
        return None
    return m.group("key").strip(), m.group("val").strip()


def _median_h(words: list[OCRWord]) -> float:
    hs = [w.bbox.h for w in words if w.bbox.h > 0]
    return statistics.median(hs) if hs else 0.02


def _split(idxs: list[int], words: list[OCRWord], axis: str, min_gap: float) -> list[list[int]]:
    lo = (lambda b: b.y1) if axis == "y" else (lambda b: b.x1)
    hi = (lambda b: b.y2) if axis == "y" else (lambda b: b.x2)
    order = sorted(idxs, key=lambda i: lo(words[i].bbox))
    groups: list[list[int]] = [[order[0]]]
    reach = hi(words[order[0]].bbox)
    for i in order[1:]:
        b = words[i].bbox
        if lo(b) - reach > min_gap:
            groups.append([i])
        else:
            groups[-1].append(i)
        reach = max(reach, hi(b))
    return groups


def xy_cut(words: list[OCRWord], gap_y: float | None = None, gap_x: float | None = None) -> list[list[int]]:
    """Return regions as lists of word indices in reading order."""
    if not words:
        return []
    mh = _median_h(words)
    gy = gap_y if gap_y is not None else 0.9 * mh
    gx = gap_x if gap_x is not None else 2.2 * mh

    def rec(idxs: list[int], depth: int) -> list[list[int]]:
        if len(idxs) <= 1 or depth > 40:
            return [idxs]
        parts = _split(idxs, words, "y", gy)
        if len(parts) > 1:
            return [r for p in parts for r in rec(p, depth + 1)]
        parts = _split(idxs, words, "x", gx)
        if len(parts) > 1:
            return [r for p in parts for r in rec(p, depth + 1)]
        return [idxs]

    return rec(list(range(len(words))), 0)


def _lines(idxs: list[int], words: list[OCRWord]) -> list[list[int]]:
    """Cluster words of one region into lines by vertical centre.

    Compares against the line's *median* centre/height rather than its union box: with tall
    Devanagari/Gurmukhi matras a growing union box would swallow the next line.
    """
    order = sorted(idxs, key=lambda i: words[i].bbox.cy)
    lines: list[list[int]] = []
    for i in order:
        b = words[i].bbox
        if lines:
            cys = [words[j].bbox.cy for j in lines[-1]]
            hs = [words[j].bbox.h for j in lines[-1]]
            if abs(b.cy - statistics.median(cys)) < 0.45 * min(statistics.median(hs), b.h * 1.5):
                lines[-1].append(i)
                continue
        lines.append([i])
    return [sorted(l, key=lambda i: words[i].bbox.x1) for l in lines]


def _split_at_gaps(line: list[int], words: list[OCRWord], gap: float) -> list[list[int]]:
    """Two 'label : value' pairs printed side by side on one row become separate blocks."""
    segs = [[line[0]]]
    for prev, cur in zip(line, line[1:]):
        if words[cur].bbox.x1 - words[prev].bbox.x2 > gap:
            segs.append([cur])
        else:
            segs[-1].append(cur)
    return segs


def _row_index(b: BBox, hlines: list[tuple[float, float, float]]) -> int | None:
    spanning = [y for y, x1, x2 in hlines if x1 - 0.01 <= b.cx <= x2 + 0.01]
    if len(spanning) < 2 or not (spanning[0] < b.cy < spanning[-1]):
        return None  # not inside a ruled table
    return sum(1 for y in spanning if y < b.cy)


JUNK = re.compile(r"^[|!'\"`>\[\]{}_~=]+$")  # keeps ':' and '-', which carry key:value meaning


def _clean_words(words: list[OCRWord]) -> list[OCRWord]:
    """Drop table-border artefacts ('|', '[') that Tesseract emits for ruling lines."""
    out = []
    for w in words:
        t = re.sub(r"_{2,}", "", w.text).strip().strip("|[]_")
        if not t or JUNK.match(t):
            continue
        out.append(w if t == w.text else w.model_copy(update={"text": t}))
    return out


def _split_at_vlines(line: list[int], words: list[OCRWord],
                     vlines: list[tuple[float, float, float]]) -> list[list[int]]:
    """Break a line into cells wherever a vertical table rule passes between two words."""
    if not vlines or len(line) < 2:
        return [line]
    segs = [[line[0]]]
    for prev, cur in zip(line, line[1:]):
        a, b = words[prev].bbox, words[cur].bbox
        cy = (a.cy + b.cy) / 2
        # centre-based: OCR junk read off the rule itself may straddle it
        crosses = any(a.cx < x < b.cx and y1 <= cy <= y2 for x, y1, y2 in vlines)
        if crosses:
            segs.append([cur])
        else:
            segs[-1].append(cur)
    return segs


def _col_index(b: BBox, vlines: list[tuple[float, float, float]]) -> int | None:
    xs = sorted(x for x, y1, y2 in vlines if y1 - 0.005 <= b.cy <= y2 + 0.005)
    if len(xs) < 2 or not (xs[0] < b.cx < xs[-1]):
        return None
    return sum(1 for x in xs if x < b.cx)


def build_blocks(words: list[OCRWord], start: int = 0,
                 hlines: list[tuple[float, float, float]] | None = None,
                 vlines: list[tuple[float, float, float]] | None = None) -> list[Block]:
    words = _clean_words(words)
    if not words:
        return []
    gap = 3.0 * _median_h(words)
    blocks: list[Block] = []
    for r, region in enumerate(xy_cut(words)):
        cells = [cell for line in _lines(region, words)
                 for seg in _split_at_vlines(line, words, vlines or [])
                 for cell in _split_at_gaps(seg, words, gap)]
        for line in cells:
            ws = [words[i] for i in line]
            text = " ".join(w.text for w in ws).strip()
            if not text:
                continue
            confs = [w.conf for w in ws]
            blk = Block(
                id=f"b{start + len(blocks)}",
                text=text,
                bbox=BBox.union([w.bbox for w in ws]),
                page=ws[0].page,
                conf=min(confs),
                word_confs=confs,
                region=r,
            )
            kv = split_kv(text)
            if kv:
                blk.key, blk.value = kv
            if hlines:
                blk.row = _row_index(blk.bbox, hlines)
            if vlines:
                blk.col = _col_index(blk.bbox, vlines)
            blocks.append(blk)
    _pair_labels(blocks)
    return blocks


def _pair_labels(blocks: list[Block]) -> None:
    """A label block ("नाम :") with its value in the next cell to the right."""
    for b in blocks:
        if b.value or not re.search(r"[:：]\s*$", b.text):
            continue
        cands = [
            o for o in blocks
            if o is not b and o.page == b.page and o.bbox.x1 >= b.bbox.x2 - 0.01
            and min(o.bbox.y2, b.bbox.y2) - max(o.bbox.y1, b.bbox.y1) > 0.4 * min(o.bbox.h, b.bbox.h)
        ]
        if cands:
            right = min(cands, key=lambda o: o.bbox.x1)
            b.key = b.text.rstrip(" :：")
            b.value = right.text
