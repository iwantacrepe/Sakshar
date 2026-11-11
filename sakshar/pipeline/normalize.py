"""Text normalization for Indian documents: digits, amounts, dates, areas."""
from __future__ import annotations

import re
import unicodedata
from datetime import date

_DIGITS = {}
for base in (0x0966, 0x0A66):  # Devanagari ०-९, Gurmukhi ੦-੯
    for i in range(10):
        _DIGITS[chr(base + i)] = str(i)
_DIGIT_TABLE = str.maketrans(_DIGITS)


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def ascii_digits(text: str) -> str:
    return text.translate(_DIGIT_TABLE)


def clean(text: str) -> str:
    text = nfc(text).replace("‌", "").replace("‍", "")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


# ---------------------------------------------------------------- amounts
_CUR = r"(?:₹|(?<![a-z])rs\.?|(?<![a-z])inr|रु\.?|रू\.?|रुपये|रुपए|रुपया|ਰੁ\.?|ਰੁਪਏ)"
AMOUNT_RE = re.compile(
    rf"(?:{_CUR}\s*(?P<a>\d[\d,]*(?:\.\d{{1,2}})?)|(?P<b>\d[\d,]*(?:\.\d{{1,2}})?)\s*(?:/-)?\s*{_CUR})",
    re.IGNORECASE,
)


def parse_amount(text: str) -> float | None:
    """'₹ 1,25,000/-' -> 125000.0 ; '४५,००० रुपये' -> 45000.0"""
    m = AMOUNT_RE.search(ascii_digits(nfc(text)))
    if not m:
        return None
    raw = (m.group("a") or m.group("b")).replace(",", "")
    try:
        return float(raw)
    except ValueError:
        return None


def find_amounts(text: str) -> list[tuple[str, float]]:
    out = []
    t = ascii_digits(nfc(text))
    for m in AMOUNT_RE.finditer(t):
        raw = (m.group("a") or m.group("b")).replace(",", "")
        try:
            out.append((m.group(0).strip(), float(raw)))
        except ValueError:
            pass
    return out


def format_inr(value: float) -> str:
    """Indian digit grouping: 1234567 -> '12,34,567'."""
    neg = value < 0
    whole, _, frac = f"{abs(value):.2f}".partition(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        head = ",".join(re.findall(r"\d{1,2}(?=(?:\d{2})*$)", head))
        whole = f"{head},{tail}"
    s = whole if frac == "00" else f"{whole}.{frac}"
    return f"{'-' if neg else ''}₹{s}"


# ---------------------------------------------------------------- dates
_MONTHS_RAW = {
    1: ["january", "jan", "जनवरी", "जानेवारी", "ਜਨਵਰੀ"],
    2: ["february", "feb", "फ़रवरी", "फरवरी", "फेब्रुवारी", "ਫ਼ਰਵਰੀ", "ਫਰਵਰੀ"],
    3: ["march", "mar", "मार्च", "ਮਾਰਚ"],
    4: ["april", "apr", "अप्रैल", "एप्रिल", "ਅਪ੍ਰੈਲ", "ਅਪਰੈਲ"],
    5: ["may", "मई", "मे", "ਮਈ"],
    6: ["june", "jun", "जून", "ਜੂਨ"],
    7: ["july", "jul", "जुलाई", "जुलै", "ਜੁਲਾਈ"],
    8: ["august", "aug", "अगस्त", "ऑगस्ट", "ਅਗਸਤ"],
    9: ["september", "sep", "sept", "सितंबर", "सितम्बर", "सप्टेंबर", "ਸਤੰਬਰ"],
    10: ["october", "oct", "अक्टूबर", "अक्तूबर", "ऑक्टोबर", "ਅਕਤੂਬਰ"],
    11: ["november", "nov", "नवंबर", "नवम्बर", "नोव्हेंबर", "ਨਵੰਬਰ"],
    12: ["december", "dec", "दिसंबर", "दिसम्बर", "डिसेंबर", "ਦਸੰਬਰ"],
}
MONTHS = {nfc(name): num for num, names in _MONTHS_RAW.items() for name in names}
_MONTH_ALT = "|".join(sorted((re.escape(k) for k in MONTHS), key=len, reverse=True))

DATE_NUMERIC = re.compile(r"(?<!\d)(\d{1,2})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(\d{4}|\d{2})(?!\d)")
DATE_ISO = re.compile(r"(?<!\d)(\d{4})-(\d{1,2})-(\d{1,2})(?!\d)")
DATE_WORD = re.compile(rf"(?<!\d)(\d{{1,2}})\s*(?:st|nd|rd|th)?\s+({_MONTH_ALT})[,.]?\s+(\d{{4}})(?!\d)", re.IGNORECASE)


def _mk(y: int, m: int, d: int) -> date | None:
    if y < 100:
        y += 2000 if y < 50 else 1900
    try:
        return date(y, m, d)
    except ValueError:
        return None


def find_dates(text: str) -> list[tuple[str, date]]:
    """All dates in text (Indian dd/mm/yyyy order), as (matched text, date)."""
    t = ascii_digits(nfc(text))
    out: list[tuple[str, date]] = []
    for m in DATE_WORD.finditer(t):
        d = _mk(int(m.group(3)), MONTHS[nfc(m.group(2)).lower()], int(m.group(1)))
        if d:
            out.append((m.group(0), d))
    for m in DATE_ISO.finditer(t):
        d = _mk(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if d:
            out.append((m.group(0), d))
    for m in DATE_NUMERIC.finditer(t):
        d = _mk(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        if d and not any(m.group(0) in o[0] or o[0] in m.group(0) for o in out):
            out.append((m.group(0), d))
    return out


def parse_date(text: str) -> date | None:
    found = find_dates(text)
    return found[0][1] if found else None


# ---------------------------------------------------------------- area units
AREA_UNITS = {
    "hectare": ["hectare", "hectares", "ha", "हेक्टेयर", "हेक्टर", "हे.", "ਹੈਕਟੇਅਰ"],
    "are": ["are", "आर", "आर."],
    "acre": ["acre", "acres", "एकड़", "एकर", "ਏਕੜ"],
    "bigha": ["bigha", "बीघा", "बिघा", "ਵਿੱਘਾ", "ਬਿਘਾ"],
    "kanal": ["kanal", "कनाल", "ਕਨਾਲ"],
    "marla": ["marla", "मरला", "ਮਰਲਾ", "ਮਰਲੇ"],
    "sq_m": ["sq m", "sq. m", "sqm", "वर्ग मीटर", "चौ.मी.", "ਵਰਗ ਮੀਟਰ"],
}
_UNIT_LOOKUP = {nfc(a).lower(): unit for unit, al in AREA_UNITS.items() for a in al}
_UNIT_ALT = "|".join(sorted((re.escape(k) for k in _UNIT_LOOKUP), key=len, reverse=True))
AREA_RE = re.compile(rf"(?<![\w.])(\d+(?:\.\d+)?)\s*({_UNIT_ALT})(?![\w])", re.IGNORECASE)


def find_areas(text: str) -> list[tuple[str, float, str]]:
    t = ascii_digits(nfc(text))
    return [(m.group(0), float(m.group(1)), _UNIT_LOOKUP[m.group(2).lower()]) for m in AREA_RE.finditer(t)]


def known_area_unit(unit_text: str) -> str | None:
    return _UNIT_LOOKUP.get(nfc(unit_text).strip().lower())
