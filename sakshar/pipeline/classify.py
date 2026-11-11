"""Document-type classification: multilingual keyword rules with an LLM fallback."""
from __future__ import annotations

import logging

from ..models import Block, DocType
from .normalize import ascii_digits, nfc

log = logging.getLogger(__name__)

# keyword -> weight, per document type (hi / mr / pa / en)
KEYWORDS: dict[DocType, dict[str, float]] = {
    DocType.land_record_up_khatauni: {
        "खतौनी": 3, "खसरा": 2, "खाता संख्या": 2, "खातेदार": 2, "भूमिधर": 2, "तहसील": 1,
        "परगना": 1.5, "फसली": 2, "दाखिल-खारिज": 1.5, "khatauni": 3, "khasra": 2, "bhumidhar": 2,
    },
    DocType.land_record_mh_712: {
        "7/12": 3, "सातबारा": 3, "गाव नमुना": 3, "भोगवटादार": 2.5, "भूमापन क्रमांक": 2.5, "गट क्रमांक": 2,
        "तालुका": 1, "हेक्टर": 1, "आर": 0.5, "फेरफार": 2, "satbara": 3, "village form vii": 2,
    },
    DocType.land_record_pb_jamabandi: {
        "ਜਮ੍ਹਾਂਬੰਦੀ": 3, "ਜਮਾਬੰਦੀ": 3, "ਫ਼ਰਦ": 2.5, "ਫਰਦ": 2.5, "ਖਸਰਾ": 2, "ਖੇਵਟ": 2.5, "ਖਤੌਨੀ": 2,
        "ਮਾਲਕ": 1.5, "ਕਾਸ਼ਤਕਾਰ": 2, "ਇੰਤਕਾਲ": 2, "ਕਨਾਲ": 1.5, "ਮਰਲੇ": 1.5, "ਪਟਵਾਰੀ": 1.5,
        "jamabandi": 3, "fard": 2, "khewat": 2.5, "kanal": 1, "marla": 1,
    },
    DocType.application_form: {
        "आवेदन": 2, "आवेदक": 2, "आवेदन संख्या": 3, "प्रार्थना पत्र": 2, "अर्ज": 2, "अर्जदार": 2,
        "अर्ज क्रमांक": 3, "ਅਰਜ਼ੀ": 2, "ਬਿਨੈਕਾਰ": 2, "ਬਿਨੈ-ਪੱਤਰ": 2, "application": 2, "applicant": 2,
        "application no": 3, "acknowledgement": 1.5, "form": 0.5, "fee": 1, "शुल्क": 1, "ਫੀਸ": 1,
    },
    DocType.certificate: {
        "प्रमाण पत्र": 3, "प्रमाणपत्र": 3, "प्रमाणित किया जाता है": 3, "दाखला": 2.5, "प्रमाणित करण्यात": 3,
        "ਸਰਟੀਫਿਕੇਟ": 3, "ਪ੍ਰਮਾਣ ਪੱਤਰ": 3, "certificate": 3, "certified that": 3, "income": 1, "caste": 1,
        "आय": 1, "जाति": 1, "निवास": 1, "domicile": 1.5,
    },
}
MIN_RULE_SCORE = 3.0


def rule_scores(text: str) -> dict[DocType, float]:
    t = ascii_digits(nfc(text)).lower()
    scores = {}
    for dt, kws in KEYWORDS.items():
        scores[dt] = sum(w * min(3, t.count(nfc(k).lower())) for k, w in kws.items())
    return scores


def classify(blocks: list[Block], llm=None) -> tuple[DocType, float]:
    text = "\n".join(b.text for b in blocks)
    scores = rule_scores(text)
    best = max(scores, key=scores.get)
    total = sum(scores.values()) or 1.0
    top = scores[best]
    if top >= MIN_RULE_SCORE:
        return best, min(0.99, 0.5 + 0.5 * top / total)
    if llm is not None:
        try:
            dt, conf = llm.classify(text[:4000], [d.value for d in DocType])
            return DocType(dt), conf
        except Exception as e:
            log.warning("LLM classification failed: %s", e)
    return (best, 0.4) if top > 0 else (DocType.generic, 0.3)
