"""Government-ID checksums and PII masking."""
from __future__ import annotations

import re

# Verhoeff tables (used by UIDAI for Aadhaar check digits)
_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]
_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def verhoeff_valid(number: str) -> bool:
    digits = re.sub(r"\D", "", number)
    if not digits:
        return False
    c = 0
    for i, ch in enumerate(reversed(digits)):
        c = _D[c][_P[i % 8][int(ch)]]
    return c == 0


def verhoeff_check_digit(number: str) -> str:
    digits = re.sub(r"\D", "", number)
    c = 0
    for i, ch in enumerate(reversed(digits)):
        c = _D[c][_P[(i + 1) % 8][int(ch)]]
    return str(_INV[c])


AADHAAR_RE = re.compile(r"(?<!\d)([2-9]\d{3})[\s-]?(\d{4})[\s-]?(\d{4})(?!\d)")
PAN_RE = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")


def is_valid_aadhaar(value: str) -> bool:
    digits = re.sub(r"\D", "", value)
    return len(digits) == 12 and digits[0] not in "01" and verhoeff_valid(digits)


def is_valid_pan(value: str) -> bool:
    return bool(PAN_RE.fullmatch(value.strip().upper()))


def mask_aadhaar(value: str) -> str:
    digits = re.sub(r"\D", "", value)
    return f"XXXX-XXXX-{digits[-4:]}" if len(digits) >= 4 else "XXXX"


def mask_text(text: str) -> str:
    """Mask every Verhoeff-valid Aadhaar-looking number in free text."""

    def repl(m: re.Match) -> str:
        num = "".join(m.groups())
        return mask_aadhaar(num) if verhoeff_valid(num) else m.group(0)

    return AADHAAR_RE.sub(repl, text)
