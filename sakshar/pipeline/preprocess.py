"""Image preprocessing.

Flow per page:
    load (image or PDF page) -> estimate skew on a binarized copy -> rotate the ORIGINAL RGB
    -> save page_deskewed.png (served to the viewer; ink stamps keep their colour)
    -> fork A: denoise + CLAHE (for PaddleOCR, which wants natural grey/colour images)
    -> fork B: adaptive threshold + morphology (for Tesseract only)

Every fork has the same size as page_deskewed.png, so OCR boxes normalized to [0,1]
line up 1:1 in the UI without any inverse transform.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MAX_SIDE = 2400  # cap resolution; enough for 300-dpi A4 crops
MIN_SIDE = 1400  # upscale small phone photos so small text survives


@dataclass
class PreparedPage:
    index: int
    rgb: np.ndarray  # deskewed colour image (H, W, 3) uint8, RGB order
    clahe: np.ndarray  # fork A, (H, W, 3) uint8 grey replicated to 3 channels
    binary: np.ndarray  # fork B, (H, W) uint8 {0,255}
    skew_deg: float
    hlines: list[tuple[float, float, float]] = None  # table ruling lines: (y, x1, x2), normalized
    vlines: list[tuple[float, float, float]] = None  # vertical rules: (x, y1, y2), normalized

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.rgb.shape[:2]
        return w, h


def load_pages(path: Path, dpi: int = 250) -> list[np.ndarray]:
    """Return RGB uint8 arrays for an image file or each page of a PDF."""
    if path.suffix.lower() == ".pdf":
        try:
            import pymupdf as fitz  # PyMuPDF >= 1.24
        except ImportError:
            import fitz

        pages = []
        with fitz.open(path) as pdf:
            for page in pdf:
                pix = page.get_pixmap(dpi=dpi, alpha=False)
                arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                pages.append(arr[:, :, :3].copy())
        return pages
    data = np.fromfile(str(path), dtype=np.uint8)  # handles non-ASCII Windows paths
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Unsupported or corrupt image: {path.name}")
    return [cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)]


def _resize(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    side = max(h, w)
    if side > MAX_SIDE:
        s = MAX_SIDE / side
    elif side < MIN_SIDE:
        s = MIN_SIDE / side
    else:
        return rgb
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    return cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=interp)


def estimate_skew_from_rules(gray: np.ndarray) -> float | None:
    """Skew from long straight lines (table rules, underlines) via probabilistic Hough.
    Far more precise than text blobs when a form has ruling lines; None if it has none."""
    h, w = gray.shape
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 1800, threshold=200, minLineLength=int(0.3 * w), maxLineGap=8)
    if lines is None:
        return None
    angs, lens = [], []
    for x1, y1, x2, y2 in lines[:, 0]:
        a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(a) < 20:
            angs.append(a)
            lens.append(np.hypot(x2 - x1, y2 - y1))
    if len(angs) < 3:
        return None
    order = np.argsort(angs)
    a, l = np.array(angs)[order], np.array(lens)[order]
    cum = np.cumsum(l)
    return -float(a[np.searchsorted(cum, cum[-1] / 2)])


def estimate_skew(gray: np.ndarray) -> float:
    """Skew angle in degrees (positive = text rotated counter-clockwise).

    Prefers ruling lines (Hough); otherwise uses text-line blobs: binarize, dilate
    horizontally so words merge into lines, minAreaRect each wide contour, median angle.
    """
    ruled = estimate_skew_from_rules(gray)
    if ruled is not None:
        return ruled
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = bw.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, w // 60), 3))
    lines = cv2.dilate(bw, kernel, iterations=2)
    contours, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    angles: list[float] = []
    weights: list[float] = []
    for c in contours:
        (cx, cy), (rw, rh), ang = cv2.minAreaRect(c)
        if rw < rh:
            rw, rh = rh, rw
            ang = ang - 90
        if rw < w * 0.08 or rh < 4 or rw / max(rh, 1) < 4:
            continue
        # normalize to (-45, 45]
        while ang <= -45:
            ang += 90
        while ang > 45:
            ang -= 90
        angles.append(ang)
        weights.append(rw)
    if not angles:
        return 0.0
    order = np.argsort(angles)
    a = np.array(angles)[order]
    wts = np.array(weights)[order]
    cum = np.cumsum(wts)
    median = float(a[np.searchsorted(cum, cum[-1] / 2)])
    # OpenCV's minAreaRect angle is measured in image coords (y down); convert so that
    # a positive result means the content is rotated counter-clockwise.
    return -median


def rotate(img: np.ndarray, angle_deg: float, border=(255, 255, 255)) -> np.ndarray:
    """Rotate by angle_deg (counter-clockwise positive), keeping full content."""
    if abs(angle_deg) < 0.05:
        return img
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    bv = border if img.ndim == 3 else border[0]
    return cv2.warpAffine(img, m, (nw, nh), flags=cv2.INTER_CUBIC, borderValue=bv)


def deskew(rgb: np.ndarray) -> tuple[np.ndarray, float]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    skew = estimate_skew(gray)
    if abs(skew) < 0.3 or abs(skew) > 20:
        return rgb, 0.0
    # content is rotated by +skew (CCW) -> rotate back clockwise
    return rotate(rgb, -skew), skew


def make_clahe(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, h=7, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return cv2.cvtColor(clahe, cv2.COLOR_GRAY2RGB)


def make_binary(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # suppress coloured stamps: take the max over channels so purple/blue ink gets lighter
    maxc = rgb.max(axis=2)
    gray = cv2.addWeighted(gray, 0.5, maxc, 0.5, 0)
    gray = cv2.medianBlur(gray, 3)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
    # remove salt noise (small black specks) without eroding thin matras
    inv = 255 - bw
    inv = cv2.morphologyEx(inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    return 255 - inv


def find_hlines(binary: np.ndarray) -> list[tuple[float, float, float]]:
    """Horizontal ruling lines (table borders) as normalized (y, x1, x2)."""
    h, w = binary.shape[:2]
    inv = 255 - binary
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, w // 14), 1))
    lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel)
    lines = cv2.dilate(lines, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3)))
    contours, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw >= 0.12 * w and bh <= 0.015 * h:
            out.append(((y + bh / 2) / h, x / w, (x + bw) / w))
    out.sort()
    merged: list[tuple[float, float, float]] = []
    for ln in out:
        if merged and abs(ln[0] - merged[-1][0]) < 0.004:
            y0, a, b = merged[-1]
            merged[-1] = (y0, min(a, ln[1]), max(b, ln[2]))
        else:
            merged.append(ln)
    return merged


def find_vlines(binary: np.ndarray) -> list[tuple[float, float, float]]:
    """Vertical ruling lines (table column separators) as normalized (x, y1, y2)."""
    h, w = binary.shape[:2]
    inv = 255 - binary
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(30, h // 40)))
    lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel)
    lines = cv2.dilate(lines, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15)))
    contours, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bh >= 0.03 * h and bw <= 0.012 * w:
            out.append(((x + bw / 2) / w, y / h, (y + bh) / h))
    return sorted(out)


def prepare_page(rgb: np.ndarray, index: int = 0) -> PreparedPage:
    rgb = _resize(rgb)
    deskewed, skew = deskew(rgb)
    binary = make_binary(deskewed)
    return PreparedPage(
        index=index,
        rgb=deskewed,
        clahe=make_clahe(deskewed),
        binary=binary,
        skew_deg=skew,
        hlines=find_hlines(binary),
        vlines=find_vlines(binary),
    )


def save_rgb(rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise IOError("PNG encode failed")
    buf.tofile(str(path))
