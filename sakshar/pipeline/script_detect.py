"""Script and language detection for Devanagari (hi/mr), Gurmukhi (pa) and Latin (en)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .tess import lang_string, pytesseract_mod

RANGES = {
    "Deva": (0x0900, 0x097F),
    "Guru": (0x0A00, 0x0A7F),
}
OSD_MAP = {"Devanagari": "Deva", "Gurmukhi": "Guru", "Latin": "Latn"}

MARATHI_MARKERS = ["आहे", "च्या", "ळ", "आणि", "नाही", "तालुका", "जिल्हा", "सातबारा", "गाव ", "यांचे", "भोगवटादार"]
HINDI_MARKERS = ["है", "और", " के ", " की ", " में ", "जिला", "तहसील", "ग्राम", "खसरा", "खतौनी", "का ", "हैं"]


@dataclass
class ScriptGuess:
    script: str  # Deva | Guru | Latn
    lang: str  # hi | mr | pa | en
    conf: float
    counts: dict[str, int]


def count_scripts(text: str) -> dict[str, int]:
    counts = {"Deva": 0, "Guru": 0, "Latn": 0}
    for ch in text:
        o = ord(ch)
        if ("a" <= ch <= "z") or ("A" <= ch <= "Z"):
            counts["Latn"] += 1
            continue
        for name, (lo, hi) in RANGES.items():
            if lo <= o <= hi:
                counts[name] += 1
                break
    return counts


def dominant_script(counts: dict[str, int], hint: str | None = None) -> tuple[str, float]:
    total = sum(counts.values())
    if total == 0:
        return (hint or "Latn"), 0.2
    indic = {k: v for k, v in counts.items() if k != "Latn"}
    best = max(indic, key=indic.get)
    # Government documents mix English labels with Indic text; Indic wins if it is a real share.
    if indic[best] >= 0.15 * total:
        return best, min(1.0, indic[best] / max(1, total) + 0.3)
    return "Latn", counts["Latn"] / total


@lru_cache
def _lingua():
    try:
        from lingua import Language, LanguageDetectorBuilder

        langs = [Language.HINDI, Language.MARATHI, Language.PUNJABI, Language.ENGLISH]
        return LanguageDetectorBuilder.from_languages(*langs).build(), Language
    except Exception:
        return None


def devanagari_lang(text: str) -> tuple[str, float]:
    """Hindi vs Marathi for Devanagari text: marker vote + lingua."""
    padded = f" {text} "
    mr = sum(padded.count(m) for m in MARATHI_MARKERS) + 3 * text.count("ळ")
    hi = sum(padded.count(m) for m in HINDI_MARKERS)
    score_mr = mr / (mr + hi) if (mr + hi) else 0.5
    det = _lingua()
    if det is not None:
        detector, Language = det
        try:
            confs = {c.language: c.value for c in detector.compute_language_confidence_values(text[:2000])}
            lm, lh = confs.get(Language.MARATHI, 0.0), confs.get(Language.HINDI, 0.0)
            if lm + lh > 0:
                score_mr = 0.5 * score_mr + 0.5 * (lm / (lm + lh))
        except Exception:
            pass
    return ("mr", score_mr) if score_mr > 0.5 else ("hi", 1 - score_mr)


def osd_hint(binary: np.ndarray) -> str | None:
    pt = pytesseract_mod()
    if pt is None or "osd" not in (lang_string(["osd"]) or ""):
        return None
    try:
        osd = pt.image_to_osd(binary, config="--psm 0", output_type=pt.Output.DICT)
        return OSD_MAP.get(osd.get("script"))
    except Exception:
        return None


def probe_text(binary: np.ndarray) -> str:
    """Cheap multi-script Tesseract pass on a downscaled page."""
    pt = pytesseract_mod()
    langs = lang_string(["hin", "pan", "eng"])
    if pt is None or not langs:
        return ""
    import cv2

    h, w = binary.shape[:2]
    s = min(1.0, 1400 / max(h, w))
    small = cv2.resize(binary, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else binary
    try:
        return pt.image_to_string(small, lang=langs, config="--psm 3")
    except Exception:
        return ""


def detect(text: str, hint: str | None = None) -> ScriptGuess:
    text = re.sub(r"\s+", " ", text)
    counts = count_scripts(text)
    script, conf = dominant_script(counts, hint)
    if hint and hint == script:
        conf = min(1.0, conf + 0.15)
    if script == "Deva":
        lang, lconf = devanagari_lang(text)
        return ScriptGuess(script, lang, min(conf, 0.5 + lconf / 2), counts)
    if script == "Guru":
        return ScriptGuess(script, "pa", conf, counts)
    return ScriptGuess("Latn", "en", conf, counts)


def detect_page(binary: np.ndarray) -> ScriptGuess:
    """Pre-OCR detection used to route a page to the right engine/model."""
    hint = osd_hint(binary)
    return detect(probe_text(binary), hint)
