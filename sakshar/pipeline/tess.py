"""Tesseract discovery + thin wrappers (shared by OCR and script detection)."""
from __future__ import annotations

import os
import shutil
import sys
from functools import lru_cache
from pathlib import Path

from ..config import get_settings


@lru_cache
def tesseract_cmd() -> str | None:
    s = get_settings()
    candidates = [
        s.tesseract_cmd,
        str(Path(sys.prefix) / "Library" / "bin" / "tesseract.exe"),  # conda-forge (Windows)
        str(Path(sys.prefix) / "bin" / "tesseract"),  # conda-forge (Linux/mac)
        shutil.which("tesseract"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    ]
    for c in candidates:
        if c and Path(c).exists():
            return c
    return None


@lru_cache
def tessdata_dir() -> str | None:
    if os.environ.get("TESSDATA_PREFIX"):
        return os.environ["TESSDATA_PREFIX"]
    cmd = tesseract_cmd()
    if not cmd:
        return None
    p = Path(cmd).resolve()
    for cand in (
        Path(sys.prefix) / "share" / "tessdata",  # conda-forge on Windows: <env>/share/tessdata
        p.parent.parent / "share" / "tessdata",  # conda on Linux/mac, Library/share on some builds
        p.parent / "tessdata",  # UB Mannheim installer
    ):
        if cand.exists():
            return str(cand)
    return None


@lru_cache
def pytesseract_mod():
    """Return a configured pytesseract module, or None if Tesseract is unavailable."""
    cmd = tesseract_cmd()
    if not cmd:
        return None
    try:
        import pytesseract
    except ImportError:
        return None
    pytesseract.pytesseract.tesseract_cmd = cmd
    td = tessdata_dir()
    if td:
        os.environ.setdefault("TESSDATA_PREFIX", td)
    return pytesseract


@lru_cache
def available_langs() -> frozenset[str]:
    pt = pytesseract_mod()
    if pt is None:
        return frozenset()
    try:
        return frozenset(pt.get_languages(config=""))
    except Exception:
        return frozenset()


def lang_string(wanted: list[str]) -> str | None:
    have = available_langs()
    langs = [l for l in wanted if l in have]
    return "+".join(langs) if langs else None
