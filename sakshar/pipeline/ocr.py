"""OCR routing: PaddleOCR (primary for Devanagari/Latin) with Tesseract fallback,
and a Paddle-detect + Tesseract-recognize hybrid for Gurmukhi (Punjabi).
"""
from __future__ import annotations

import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np

from ..config import LANGUAGES, get_settings
from ..models import BBox, OCRResult, OCRWord
from . import script_detect
from .preprocess import PreparedPage
from .tess import lang_string, pytesseract_mod

log = logging.getLogger(__name__)

# Script -> PaddleOCR recognizer used as the primary engine.
# Devanagari is deliberately absent: on our printed land-record samples Paddle's devanagari
# PP-OCRv3 recognizer dropped spaces and conjuncts while still reporting ~0.95 confidence
# (so confidence-based fallback never fired); Tesseract hin/mar + digit repair read far better.
# Devanagari and Gurmukhi therefore use Tesseract recognition with Paddle *detection*.
PADDLE_SUPPORTED_SCRIPTS = {"Latn": "en"}
TESS_LANGS = {
    ("Deva", "hi"): ["hin", "eng"],
    ("Deva", "mr"): ["mar", "hin", "eng"],
    ("Guru", "pa"): ["pan", "eng"],
    ("Latn", "en"): ["eng"],
}

# Paddle's CPU predictor is neither thread-safe nor usable from a thread other than the one that
# created it ("RuntimeError: Unknown exception" when asyncio.to_thread hands a later document to a
# different pool thread). So every Paddle call (model creation AND inference) runs on this one
# dedicated thread, which both serializes access and keeps thread affinity.
_PADDLE_THREAD = ThreadPoolExecutor(max_workers=1, thread_name_prefix="paddle")
_PADDLE_LOCK = threading.Lock()  # guards the model cache
_paddle_cache: dict[str, object] = {}
_paddle_failed = False


def _on_paddle_thread(fn, *args):
    return _PADDLE_THREAD.submit(fn, *args).result()

# Test hook: replace the whole OCR step with a function(page) -> OCRResult
_override: Callable[[PreparedPage], OCRResult] | None = None


def set_override(fn: Callable[[PreparedPage], OCRResult] | None) -> None:
    global _override
    _override = fn


# ---------------------------------------------------------------- Paddle
def _torch_first() -> None:
    """Windows DLL clash: once Paddle has loaded its MKL/OpenMP DLLs, a later `import torch` fails
    with 'WinError 127 ... torch\\lib\\shm.dll'. Loading torch first makes both work in one process
    (OCR on Paddle, then voice models on torch)."""
    if get_settings().enable_voice:
        try:
            import torch  # noqa: F401
        except ImportError:
            pass


def _paddle(model: str):
    global _paddle_failed
    if _paddle_failed or not get_settings().enable_paddle:
        return None
    _torch_first()
    if model in _paddle_cache:
        return _paddle_cache[model]
    with _PADDLE_LOCK:
        if model in _paddle_cache:
            return _paddle_cache[model]
        try:
            from paddleocr import PaddleOCR

            if model == "det":
                inst = _detector()
            else:
                try:  # PaddleOCR 3.x; mobile models (server ones take ~50 s/page on CPU)
                    inst = PaddleOCR(lang=model, device="cpu", use_doc_orientation_classify=False,
                                     use_doc_unwarping=False, use_textline_orientation=False,
                                     text_detection_model_name="PP-OCRv5_mobile_det",
                                     text_recognition_model_name="en_PP-OCRv4_mobile_rec" if model == "en" else None)
                except TypeError:  # PaddleOCR 2.x
                    inst = PaddleOCR(use_angle_cls=False, lang=model, use_gpu=False, show_log=False)
            _paddle_cache[model] = inst
            return inst
        except Exception as e:  # import error, model download failure...
            log.warning("PaddleOCR unavailable (%s); using Tesseract", e)
            _paddle_failed = True
            return None


def _detector():
    try:  # PaddleOCR 3.x: standalone text-detection module
        from paddleocr import TextDetection

        return TextDetection(model_name="PP-OCRv5_mobile_det", device="cpu")
    except ImportError:  # 2.x: full pipeline, called with rec=False
        from paddleocr import PaddleOCR

        return PaddleOCR(use_angle_cls=False, lang="en", use_gpu=False, show_log=False)


def _paddle_lines(ocr_obj, img: np.ndarray) -> list[tuple[list, str, float]]:
    """(polygon, text, score) per detected line, for either PaddleOCR API generation."""
    if hasattr(ocr_obj, "predict"):
        res = ocr_obj.predict(img)
        if not res:
            return []
        r = res[0]
        polys = r.get("rec_polys") if r.get("rec_polys") is not None else r.get("dt_polys", [])
        return [(list(p), t, float(s)) for p, t, s in zip(polys, r.get("rec_texts", []), r.get("rec_scores", []))]
    res = ocr_obj.ocr(img, cls=False)
    return [(line[0], line[1][0], float(line[1][1])) for line in (res[0] if res and res[0] else [])]


def _paddle_boxes(det_obj, img: np.ndarray) -> list:
    if hasattr(det_obj, "predict"):
        res = det_obj.predict(img)
        return list(res[0].get("dt_polys", [])) if res else []
    res = det_obj.ocr(img, rec=False, cls=False)
    return res[0] if res and res[0] else []


def _quad_to_bbox(quad, w: int, h: int) -> BBox:
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    return BBox(
        x1=max(0.0, min(xs) / w), y1=max(0.0, min(ys) / h),
        x2=min(1.0, max(xs) / w), y2=min(1.0, max(ys) / h),
    )


def _recognize_job(model: str, img: np.ndarray):
    ocr = _paddle(model)
    return None if ocr is None else _paddle_lines(ocr, img)


def _detect_job(img: np.ndarray):
    det = _paddle("det")
    return None if det is None else _paddle_boxes(det, img)


def paddle_recognize(page: PreparedPage, model: str) -> list[OCRWord] | None:
    try:
        lines = _on_paddle_thread(_recognize_job, model, page.clahe)
    except Exception as e:  # never fail a document because of Paddle; Tesseract takes over
        log.warning("PaddleOCR recognition failed (%s); falling back to Tesseract", e)
        return None
    if lines is None:
        return None
    w, h = page.size
    words: list[OCRWord] = []
    for quad, text, conf in lines:
        text = (text or "").strip()
        if text:
            words.append(OCRWord(text=text, bbox=_quad_to_bbox(quad, w, h), conf=max(0.0, min(1.0, conf)),
                                 page=page.index, engine="paddle"))
    return words


def paddle_detect(page: PreparedPage) -> list[BBox] | None:
    """Script-agnostic text-line detection (used to find lines Tesseract missed)."""
    try:
        quads = _on_paddle_thread(_detect_job, page.clahe)
    except Exception as e:
        log.warning("PaddleOCR detection failed (%s); continuing with Tesseract only", e)
        return None
    if quads is None:
        return None
    w, h = page.size
    return [_quad_to_bbox(q, w, h) for q in quads]


# ---------------------------------------------------------------- Tesseract
def tesseract_recognize(img: np.ndarray, langs: list[str], psm: int = 6, page_index: int = 0,
                        offset: tuple[int, int] = (0, 0), page_size: tuple[int, int] | None = None) -> list[OCRWord]:
    pt = pytesseract_mod()
    lang = lang_string(langs) or lang_string(["eng"])
    if pt is None or lang is None:
        return []
    data = pt.image_to_data(img, lang=lang, config=f"--psm {psm} --oem 1", output_type=pt.Output.DICT)
    pw, ph = page_size or (img.shape[1], img.shape[0])
    ox, oy = offset
    words = []
    for i, text in enumerate(data["text"]):
        text = (text or "").strip()
        conf = float(data["conf"][i])
        if not text or conf < 0:
            continue
        x, y, bw, bh = data["left"][i] + ox, data["top"][i] + oy, data["width"][i], data["height"][i]
        words.append(OCRWord(
            text=text, conf=max(0.0, min(1.0, conf / 100)), page=page_index, engine="tesseract",
            bbox=BBox(x1=x / pw, y1=y / ph, x2=(x + bw) / pw, y2=(y + bh) / ph),
        ))
    return words


def _iou_cover(box: BBox, words: list[OCRWord], min_conf: float) -> float:
    """Fraction of a detected line's WIDTH covered by confident words centred in it.
    (Area ratios undercount: detector line boxes are taller than Tesseract's tight word boxes.)"""
    spans = sorted(
        (max(box.x1, wd.bbox.x1), min(box.x2, wd.bbox.x2)) for wd in words
        if wd.conf >= min_conf and box.y1 <= wd.bbox.cy <= box.y2 and wd.bbox.x2 > box.x1 and wd.bbox.x1 < box.x2
    )
    covered, reach = 0.0, box.x1
    for a, b in spans:
        a = max(a, reach)
        if b > a:
            covered += b - a
            reach = b
    return min(1.0, covered / max(1e-9, box.w))


def _dedupe(words: list[OCRWord]) -> list[OCRWord]:
    """Drop repeated reads of the same word (same text, overlapping boxes), keeping the most confident."""
    keep: list[OCRWord] = []
    for w in sorted(words, key=lambda x: -x.conf):
        if any(k.text == w.text and (_overlap(w.bbox, k.bbox) > 0.3 or _overlap(k.bbox, w.bbox) > 0.3) for k in keep):
            continue
        keep.append(w)
    return keep


def _inside(word: OCRWord, box: BBox) -> bool:
    return box.x1 - 0.002 <= word.bbox.cx <= box.x2 + 0.002 and box.y1 - 0.002 <= word.bbox.cy <= box.y2 + 0.002


NUMERIC = re.compile(r"^[₹(]?\d[\d.,/:\-]*\)?$")
INDIC_LETTER = re.compile(r"[ऀ-॥॰-ॿ਀-੥ੰ-੿]")


def _overlap(a: BBox, b: BBox) -> float:
    """Intersection area as a fraction of a's area."""
    ix = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    iy = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    return ix * iy / max(1e-9, a.w * a.h)


def digit_repair(words: list[OCRWord], eng_words: list[OCRWord], min_conf: float = 0.6) -> list[OCRWord]:
    """Indic Tesseract models misread Latin digits (00127 -> 0027, 214 -> dropped).
    Numbers read by an English-only pass replace overlapping non-letter tokens, or are added
    where the Indic pass found nothing. Indic words are never replaced."""
    out = list(words)
    for e in eng_words:
        if e.conf < min_conf or not NUMERIC.match(e.text):
            continue
        hits = [w for w in out if _overlap(e.bbox, w.bbox) > 0.25 or _overlap(w.bbox, e.bbox) > 0.5]
        if any(INDIC_LETTER.search(w.text) and _overlap(e.bbox, w.bbox) > 0.5 for w in hits):
            continue  # the number sits on real Indic text: trust the Indic model there
        out = [w for w in out if w not in hits]
        out.append(e)
    return out


def tesseract_hybrid(page: PreparedPage, langs: list[str], psm: int = 6) -> list[OCRWord]:
    """Indic pages: full-page Tesseract pass + English digit repair, then re-read the
    Paddle-detected lines the page pass missed with --psm 7 (crops run 8-wide in threads)."""
    thr = get_settings().ocr_fallback_conf
    w, h = page.size
    indic = langs[0] != "eng"
    words = tesseract_recognize(page.binary, langs, psm=psm, page_index=page.index)
    if indic:
        words = digit_repair(words, tesseract_recognize(page.binary, ["eng"], psm=psm, page_index=page.index))
    boxes = paddle_detect(page)
    weak = [b for b in boxes or [] if _iou_cover(b, words, thr) < 0.5]
    if not weak:
        return _drop_noise(words) if indic else words

    def reread(b: BBox) -> tuple[BBox, list[OCRWord]]:
        pad = 4
        x1, y1 = max(0, int(b.x1 * w) - pad), max(0, int(b.y1 * h) - pad)
        x2, y2 = min(w, int(b.x2 * w) + pad), min(h, int(b.y2 * h) + pad)
        crop = page.binary[y1:y2, x1:x2]
        if crop.size == 0:
            return b, []
        kw = dict(psm=7, page_index=page.index, offset=(x1, y1), page_size=(w, h))
        got = tesseract_recognize(crop, langs, **kw)
        if indic:
            got = digit_repair(got, tesseract_recognize(crop, ["eng"], **kw))
        return b, got

    # tesseract.exe is a subprocess -> threads parallelize fine (no GIL contention)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(reread, weak))
    for box, new in results:
        if not new:
            continue
        old = [wd for wd in words if _overlap(wd.bbox, box) > 0.4]  # partial overlaps too -> no duplicates
        old_conf = np.mean([o.conf for o in old]) if old else 0.0
        new_conf = float(np.mean([n.conf for n in new]))
        if new_conf > old_conf:
            words = [wd for wd in words if wd not in old] + new
    words = _dedupe(words)
    return _drop_noise(words) if indic else words


LATIN_WORD = re.compile(r"^[A-Za-z][A-Za-z'.]*$")


def _drop_noise(words: list[OCRWord]) -> list[OCRWord]:
    """On Indic pages, low-confidence Latin 'words' are almost always stamp/seal or border noise."""
    return [w for w in words if not (LATIN_WORD.match(w.text) and w.conf < 0.5)]


# ---------------------------------------------------------------- routing
def _score(words: list[OCRWord]) -> float:
    if not words:
        return 0.0
    chars = sum(len(w.text) for w in words)
    return float(np.mean([w.conf for w in words])) * min(1.0, chars / 200)


def run_ocr(page: PreparedPage) -> OCRResult:
    if _override is not None:
        return _override(page)

    guess = script_detect.detect_page(page.binary)
    script, lang = guess.script, guess.lang
    langs = TESS_LANGS.get((script, lang), [LANGUAGES[lang]["tess"], "eng"])
    thr = get_settings().ocr_fallback_conf
    engine = "tesseract"

    if script not in PADDLE_SUPPORTED_SCRIPTS:
        words = tesseract_hybrid(page, langs)
        engine = "tesseract+paddle-det" if "det" in _paddle_cache else "tesseract"
    else:
        words = paddle_recognize(page, PADDLE_SUPPORTED_SCRIPTS[script])
        if words:
            engine = "paddle"
        mean = float(np.mean([w.conf for w in words])) if words else 0.0
        if not words or mean < thr:
            tw = tesseract_recognize(page.binary, langs, psm=3, page_index=page.index)
            if _score(tw) > _score(words or []):
                words, engine = tw, "tesseract"

    # confirm script on the real output (the probe may have been noisy)
    final = script_detect.detect(" ".join(w.text for w in words), hint=script)
    mean_conf = float(np.mean([w.conf for w in words])) if words else 0.0
    return OCRResult(words=words, script=final.script, lang=final.lang, script_conf=final.conf,
                     engine=engine, mean_conf=mean_conf)
