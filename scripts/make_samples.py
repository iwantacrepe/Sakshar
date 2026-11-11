"""Generate synthetic demo documents with realistic scan artifacts.

    python scripts/make_samples.py            -> samples/*.png

Each sample gets: ~3.5 deg rotation, a purple ink stamp over a date, uneven lighting,
Gaussian noise, and one deliberately smudged digit in a financial/area value so that
the review flag (needs_review) fires during demos.

All text is fictional. Aadhaar numbers are generated to pass the Verhoeff checksum but
are not real.
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, features

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sakshar.pii import verhoeff_check_digit  # noqa: E402

OUT = ROOT / "samples"
W, H = 1700, 2200
FONT_PATHS = [r"C:\Windows\Fonts\Nirmala.ttc", r"C:\Windows\Fonts\Nirmala.ttf",
              "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"]

_LAYOUT = None


def _layout_engine():
    global _LAYOUT
    if _LAYOUT is None:
        if features.check("raqm"):
            _LAYOUT = ImageFont.Layout.RAQM
        else:
            print("WARNING: Pillow has no RAQM (libraqm/fribidi) support; Devanagari/Gurmukhi conjuncts and "
                  "matras may render incorrectly. Falling back to basic layout.")
            _LAYOUT = ImageFont.Layout.BASIC
    return _LAYOUT


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for p in FONT_PATHS:
        if Path(p).exists():
            idx = 1 if bold and p.endswith(".ttc") else 0
            try:
                return ImageFont.truetype(p, size, index=idx, layout_engine=_layout_engine())
            except OSError:
                return ImageFont.truetype(p, size, index=0, layout_engine=_layout_engine())
    raise SystemExit("Nirmala UI font not found (expected C:\\Windows\\Fonts\\Nirmala.ttc)")


def aadhaar(seed: str) -> str:
    base = ("2" + "".join(str((ord(c) * 7) % 10) for c in seed) + "0" * 11)[:11]
    n = base + verhoeff_check_digit(base)
    return f"{n[:4]} {n[4:8]} {n[8:]}"


class Page:
    def __init__(self):
        self.img = Image.new("RGB", (W, H), (252, 251, 246))
        self.d = ImageDraw.Draw(self.img)
        self.y = 90
        self.marks: dict[str, tuple[int, int, int, int]] = {}  # named boxes for artifacts

    def text(self, x, y, s, size=34, bold=False, fill=(25, 25, 30), mark: str | None = None):
        f = font(size, bold)
        self.d.text((x, y), s, font=f, fill=fill)
        if mark:
            self.marks[mark] = self.d.textbbox((x, y), s, font=f)

    def center(self, s, size=44, bold=True):
        f = font(size, bold)
        w = self.d.textlength(s, font=f)
        self.d.text(((W - w) / 2, self.y), s, font=f, fill=(20, 20, 25))
        self.y += int(size * 1.6)

    def kv(self, pairs: list[tuple[str, str]], size=34, col2=None, mark_keys=()):
        """Rows of 'label : value'. Two pairs per row when col2 is given."""
        for i in range(0, len(pairs), 2 if col2 else 1):
            row = pairs[i:i + (2 if col2 else 1)]
            for j, (k, v) in enumerate(row):
                x = 110 if j == 0 else col2
                self.text(x, self.y, f"{k} : {v}", size, mark=k if k in mark_keys else None)
            self.y += int(size * 1.75)

    def rule(self):
        self.d.line([(90, self.y), (W - 90, self.y)], fill=(60, 60, 60), width=2)
        self.y += 24

    def table(self, headers: list[str], rows: list[list[str]], widths: list[int], size=30, mark_cells=None):
        mark_cells = mark_cells or {}
        x0, line_h, pad = 90, int(size * 1.45), 16
        xs = [x0]
        for w in widths:
            xs.append(xs[-1] + w)
        all_rows = [headers] + rows
        # each row is as tall as its tallest (multi-line) cell
        heights = [max(len(c.split("\n")) for c in cells) * line_h + 2 * pad for cells in all_rows]
        ys = [self.y]
        for h in heights:
            ys.append(ys[-1] + h)
        for r, cells in enumerate(all_rows):
            for c, cell in enumerate(cells):
                for li, line in enumerate(cell.split("\n")):
                    name = mark_cells.get((r, c)) if li == 0 else None
                    self.text(xs[c] + 14, ys[r] + pad + li * line_h, line, size, bold=(r == 0), mark=name)
        for x in xs:
            self.d.line([(x, ys[0]), (x, ys[-1])], fill=(70, 70, 70), width=2)
        for yy in ys:
            self.d.line([(xs[0], yy), (xs[-1], yy)], fill=(70, 70, 70), width=2)
        self.y = ys[-1] + 40


# ---------------------------------------------------------------- artifacts
def stamp(img: Image.Image, center: tuple[int, int], ring: str, middle: str, color=(110, 40, 150)):
    r = 135
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    cx, cy = center
    ink = color + (150,)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=ink, width=7)
    d.ellipse([cx - r + 22, cy - r + 22, cx + r - 22, cy + r - 22], outline=ink, width=3)
    f = font(26, True)
    n = len(ring)
    for i, ch in enumerate(ring):
        ang = -math.pi / 2 + (i - n / 2) * (2 * math.pi * 0.8 / n)
        x, y = cx + (r - 40) * math.cos(ang), cy + (r - 40) * math.sin(ang)
        chl = Image.new("RGBA", (40, 40), (0, 0, 0, 0))
        ImageDraw.Draw(chl).text((8, 2), ch, font=f, fill=ink)
        chl = chl.rotate(-math.degrees(ang) - 90, resample=Image.BICUBIC)
        layer.alpha_composite(chl, (int(x - 20), int(y - 20)))
    fm = font(30, True)
    w = d.textlength(middle, font=fm)
    d.text((cx - w / 2, cy - 18), middle, font=fm, fill=ink)
    layer = layer.rotate(random.uniform(-25, 25), center=center, resample=Image.BICUBIC)
    base = img.convert("RGBA")
    base.alpha_composite(layer)
    return base.convert("RGB")


def smudge(img: Image.Image, box: tuple[int, int, int, int], frac: tuple[float, float] = (0.55, 0.75)):
    """Blur + ink blot over part of a value so one digit is hard to read."""
    x1, y1, x2, y2 = box
    sx1 = int(x1 + (x2 - x1) * frac[0])
    sx2 = int(x1 + (x2 - x1) * frac[1])
    region = (sx1 - 4, y1 - 4, sx2 + 4, y2 + 6)
    crop = img.crop(region).filter(ImageFilter.GaussianBlur(5))
    img.paste(crop, region[:2])
    d = ImageDraw.Draw(img, "RGBA")
    d.ellipse([sx1, y1 + 8, sx2, y2], fill=(40, 40, 60, 70))
    return img


def scanify(img: Image.Image, angle=3.5, seed=0) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = np.asarray(img).astype(np.float32)
    h, w = arr.shape[:2]
    # uneven lighting: radial vignette + linear gradient
    yy, xx = np.mgrid[0:h, 0:w]
    grad = 1.0 - 0.18 * (xx / w) - 0.10 * (yy / h)
    vign = 1.0 - 0.22 * (((xx - w * 0.4) / w) ** 2 + ((yy - h * 0.35) / h) ** 2)
    arr = arr * (grad * vign)[..., None]
    arr += rng.normal(0, 9, arr.shape)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    out = Image.fromarray(arr).rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(205, 203, 196))
    return out.filter(ImageFilter.GaussianBlur(0.6))


# ---------------------------------------------------------------- documents
def hi_khatauni() -> Image.Image:
    p = Page()
    p.center("उत्तर प्रदेश राजस्व परिषद", 40)
    p.center("खतौनी (अधिकार अभिलेख) - फसली वर्ष 1431", 46)
    p.rule()
    p.kv([("ग्राम", "रामपुर कलां"), ("परगना", "बिजनौर"), ("तहसील", "नगीना"), ("जिला", "बिजनौर")], col2=900)
    p.rule()
    p.table(
        ["खाता संख्या", "खातेदार का नाम व पता", "खसरा संख्या", "क्षेत्रफल", "लगान"],
        [["00127", "राम प्रसाद पुत्र श्याम लाल\nनिवासी रामपुर कलां", "214", "0.632 हेक्टेयर", "₹ 45.00"],
         ["00127", "सीता देवी पत्नी राम प्रसाद\nनिवासी रामपुर कलां", "215", "0.418 हेक्टेयर", "₹ 30.00"]],
        [230, 640, 230, 230, 190], size=30, mark_cells={(1, 3): "area"},
    )
    p.kv([("आधार संख्या", aadhaar("ramprasad"))])
    p.kv([("दाखिल-खारिज आदेश दिनांक", "12/03/2023")], mark_keys=("दाखिल-खारिज आदेश दिनांक",))
    p.kv([("टिप्पणी", "विरासत के आधार पर नाम दर्ज किया गया है।")])
    p.y += 120
    p.text(1150, p.y, "लेखपाल / राजस्व निरीक्षक", 32)
    img = smudge(p.img, p.marks["area"], (0.10, 0.22))
    bx = p.marks["दाखिल-खारिज आदेश दिनांक"]
    img = stamp(img, (bx[2] + 240, (bx[1] + bx[3]) // 2), "तहसील नगीना बिजनौर", "राजस्व")
    return scanify(img, 3.5, 1)


def mr_712() -> Image.Image:
    p = Page()
    p.center("महाराष्ट्र शासन", 38)
    p.center("गाव नमुना सात (अधिकार अभिलेख पत्रक) - सातबारा उतारा", 42)
    p.rule()
    p.kv([("गाव", "वडगाव शेरी"), ("तालुका", "हवेली"), ("जिल्हा", "पुणे"), ("भूमापन क्रमांक", "45/2")], col2=900)
    p.kv([("भोगवटादार वर्ग", "वर्ग-1"), ("खाते क्रमांक", "1187")], col2=900)
    p.rule()
    p.table(
        ["भोगवटादाराचे नाव", "एकूण क्षेत्र (हे.आर.चौ.मी.)", "आकारणी", "फेरफार क्रमांक"],
        [["सुनील गणपत पाटील", "1.20.00", "रु. 4.50", "3412"],
         ["अनिता सुनील पाटील", "0.40.00", "रु. 1.50", "3412"]],
        [560, 470, 260, 280], size=30, mark_cells={(1, 1): "area"},
    )
    p.kv([("इतर हक्क", "बँक ऑफ महाराष्ट्र कर्ज बोजा रु. 2,00,000")], mark_keys=("इतर हक्क",))
    p.kv([("फेरफार दिनांक", "15/06/2022")], mark_keys=("फेरफार दिनांक",))
    p.kv([("टीप", "हा उतारा माहितीसाठी आहे. मूळ अभिलेख तलाठी कार्यालयात उपलब्ध आहे.")], size=30)
    p.y += 120
    p.text(1200, p.y, "तलाठी, सजा वडगाव शेरी", 32)
    img = smudge(p.img, p.marks["इतर हक्क"], (0.80, 0.86))
    bx = p.marks["फेरफार दिनांक"]
    img = stamp(img, (bx[2] + 230, (bx[1] + bx[3]) // 2), "तलाठी कार्यालय हवेली", "पुणे")
    return scanify(img, -3.5, 2)


def pa_jamabandi() -> Image.Image:
    p = Page()
    p.center("ਪੰਜਾਬ ਸਰਕਾਰ - ਮਾਲ ਵਿਭਾਗ", 40)
    p.center("ਫ਼ਰਦ ਜਮ੍ਹਾਂਬੰਦੀ ਸਾਲ 2021-22", 46)
    p.rule()
    p.kv([("ਪਿੰਡ", "ਘੁੰਮਣ ਕਲਾਂ"), ("ਹੱਦਬਸਤ ਨੰਬਰ", "112"), ("ਤਹਿਸੀਲ", "ਬਟਾਲਾ"), ("ਜ਼ਿਲ੍ਹਾ", "ਗੁਰਦਾਸਪੁਰ")], col2=900)
    p.rule()
    p.table(
        ["ਖੇਵਟ ਨੰਬਰ", "ਖਤੌਨੀ ਨੰਬਰ", "ਮਾਲਕ ਦਾ ਨਾਮ", "ਕਾਸ਼ਤਕਾਰ", "ਖਸਰਾ ਨੰਬਰ", "ਰਕਬਾ"],
        [["58", "71", "ਗੁਰਮੀਤ ਸਿੰਘ ਪੁੱਤਰ\nਬਲਵੰਤ ਸਿੰਘ", "ਖੁਦ ਕਾਸ਼ਤ", "23//7", "8 ਕਨਾਲ 4 ਮਰਲੇ"],
         ["58", "72", "ਹਰਪ੍ਰੀਤ ਕੌਰ ਪਤਨੀ\nਗੁਰਮੀਤ ਸਿੰਘ", "ਖੁਦ ਕਾਸ਼ਤ", "23//8", "4 ਕਨਾਲ 0 ਮਰਲੇ"]],
        [190, 190, 400, 230, 220, 290], size=28, mark_cells={(1, 5): "area"},
    )
    p.kv([("ਇੰਤਕਾਲ ਨੰਬਰ", "4521"), ("ਇੰਤਕਾਲ ਮਿਤੀ", "09/11/2021")], col2=900, mark_keys=("ਇੰਤਕਾਲ ਮਿਤੀ",))
    p.kv([("ਮਾਮਲਾ", "ਰੁ. 120")])
    p.kv([("ਕੈਫ਼ੀਅਤ", "ਵਿਰਾਸਤ ਅਨੁਸਾਰ ਇੰਤਕਾਲ ਮਨਜ਼ੂਰ ਹੋਇਆ।")])
    p.y += 120
    p.text(1250, p.y, "ਪਟਵਾਰੀ ਹਲਕਾ", 32)
    img = smudge(p.img, p.marks["area"], (0.0, 0.08))
    bx = p.marks["ਇੰਤਕਾਲ ਮਿਤੀ"]
    img = stamp(img, ((bx[0] + bx[2]) // 2 + 80, bx[3] + 60), "ਤਹਿਸੀਲ ਬਟਾਲਾ", "ਮਾਲ", color=(40, 60, 160))
    return scanify(img, 3.5, 3)


def en_application() -> Image.Image:
    p = Page()
    p.center("Government of Uttar Pradesh", 40)
    p.center("e-District Service - Application Acknowledgement", 44)
    p.rule()
    p.kv([
        ("Application No", "INC/2024/0045123"),
        ("Applicant Name", "Meena Kumari"),
        ("Father's / Husband's Name", "Rajesh Kumar"),
        ("Service", "Income Certificate"),
        ("Date of Application", "05/08/2024"),
        ("Fee Paid", "Rs. 30/-"),
        ("Office", "Tehsil Office, Sadar, Lucknow"),
        ("Aadhaar", aadhaar("meena")),
        ("Last date to submit documents", "20 August 2024"),
    ], mark_keys=("Date of Application", "Fee Paid"))
    p.rule()
    p.text(110, p.y, "Please keep this acknowledgement. Visit the office with original documents", 30)
    p.y += 50
    p.text(110, p.y, "(ration card, salary slip or self declaration) before the last date.", 30)
    p.y += 160
    p.text(1150, p.y, "Authorised Signatory", 32)
    img = smudge(p.img, p.marks["Fee Paid"], (0.80, 0.88))
    bx = p.marks["Date of Application"]
    img = stamp(img, (bx[2] + 220, (bx[1] + bx[3]) // 2), "TEHSIL SADAR LUCKNOW", "e-District")
    return scanify(img, -3.5, 4)


def main() -> None:
    random.seed(7)
    OUT.mkdir(exist_ok=True)
    docs = {"hi_khatauni.png": hi_khatauni, "mr_satbara_712.png": mr_712,
            "pa_jamabandi.png": pa_jamabandi, "en_application.png": en_application}
    for name, fn in docs.items():
        img = fn()
        img.save(OUT / name)
        print(f"wrote {OUT / name}  ({img.width}x{img.height})")


if __name__ == "__main__":
    main()
