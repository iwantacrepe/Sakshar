"""Prompt templates. Every factual claim must cite block ids like [b12]."""
from __future__ import annotations

from ..config import LANGUAGES
from ..models import Block, ExtractedField

DOC_TYPE_HINTS = {
    "land_record_up_khatauni": (
        "an Uttar Pradesh-style Khatauni / Khasra land record. Explain who the recorded holders (khatedar / "
        "bhumidhar) are, the khata and khasra (plot) numbers, area, village/tehsil/district, the fasli year, "
        "and any mutation (dakhil-kharij) entries or remarks."
    ),
    "land_record_mh_712": (
        "a Maharashtra 7/12 extract (Satbara utara, village form VII-XII). Explain the survey/gat number, "
        "occupant (bhogwatdar) names and occupancy class (वर्ग-1 / वर्ग-2), area in hectare-are, assessment "
        "(akarni), 'other rights' (इतर हक्क) such as loans/encumbrances, and mutation (ferfar) numbers."
    ),
    "land_record_pb_jamabandi": (
        "a Punjab Jamabandi / Fard land record. Explain khewat and khatauni numbers, owners (malik) and their "
        "shares, cultivators (kashtkar), khasra/killa numbers, area in kanal-marla, and any intkal (mutation) entries."
    ),
    "application_form": (
        "a government application form or acknowledgement. Explain what was applied for, the application "
        "number, applicant, dates, fees paid or due, the office handling it, and what the applicant must do next."
    ),
    "certificate": (
        "a government-issued certificate (income / caste / domicile etc.). Explain what it certifies, for whom, "
        "issuing authority, issue date and validity."
    ),
    "generic": "a government document. Explain what it is, who it concerns, and the key numbers and dates.",
}

GROUNDING_RULES = """Rules:
- Use ONLY the document text given below. Never guess or add facts that are not in it.
- Every statement must cite the block ids it relies on in cited_block_ids (e.g. ["b3","b7"]).
- Copy names, numbers, dates and amounts exactly as they appear in the cited blocks.
- If something is unclear or illegible, say so plainly instead of guessing.
- Identity numbers (Aadhaar) appear masked; never try to reconstruct them."""


def _lang_name(lang: str) -> str:
    l = LANGUAGES.get(lang, LANGUAGES["en"])
    return f"{l['name']} ({l['native']})"


def format_blocks(blocks: list[Block], limit_chars: int = 14000) -> str:
    lines, n = [], 0
    for b in blocks:
        line = f"[{b.id}] {b.text}"
        n += len(line)
        if n > limit_chars:
            lines.append("[...truncated]")
            break
        lines.append(line)
    return "\n".join(lines)


def format_fields(fields: list[ExtractedField]) -> str:
    rows = []
    for f in fields:
        if not f.value:
            continue
        flag = " (UNCERTAIN - user should verify)" if f.needs_review and not f.confirmed else ""
        flag = " (confirmed by user)" if f.confirmed else flag
        rows.append(f"- {f.label}: {f.display_value or f.value}{flag} [{', '.join(f.source_block_ids)}]")
    return "\n".join(rows) or "(none)"


# ---------------------------------------------------------------- classify
CLASSIFY_SYSTEM = "You classify Indian government documents by type. Answer with one of the allowed types."


def classify_user(text: str, types: list[str]) -> str:
    return f"Allowed types: {', '.join(types)}\n\nOCR text:\n{text}"


# ---------------------------------------------------------------- extract
EXTRACT_SYSTEM = (
    "You extract fields from OCR text of Indian government documents (Hindi, Marathi, Punjabi, English). "
    "Return only fields whose values are literally present in the text, copied exactly (keep the original "
    "script and digits). Cite the block ids where each value appears. Omit fields that are absent. "
    "For list-like fields (e.g. multiple owners) return one entry per value. Do not extract Aadhaar or PAN."
)


def extract_user(blocks: list[Block], doc_type: str, specs: list[tuple[str, str]]) -> str:
    spec_lines = "\n".join(f"- {n}: {label}" for n, label in specs)
    return f"Document type: {doc_type}\nFields to extract:\n{spec_lines}\n\nOCR blocks:\n{format_blocks(blocks)}"


# ---------------------------------------------------------------- explain
def explain_system(lang: str, doc_type: str) -> str:
    return f"""You are Sakshar, a patient assistant who explains Indian government documents to people who may
not be comfortable with legal or official language.

The document is {DOC_TYPE_HINTS.get(doc_type, DOC_TYPE_HINTS['generic'])}

Write for a person with basic schooling: short sentences, everyday words, no jargon.
Output language for *_local fields: {_lang_name(lang)}. text_en is the same statement in simple English.
Keep original revenue/legal terms from the document in the local text (e.g. खातेदार, भोगवटादार वर्ग-1,
दाखिल-खारिज, फेरफार, ਇੰਤਕਾਲ, ਖੇਵਟ) and add a short plain-language gloss in brackets the first time.
Fields marked UNCERTAIN must be described as "please check this value on the original" rather than stated as fact.

Produce:
- title: what this document is, in {_lang_name(lang)}
- summary_local: 2-3 sentence overview
- statements: 4-10 key points, one fact each, each with cited_block_ids
- action_items_local: concrete next steps for the person (deadlines, offices to visit, things to verify); empty if none

{GROUNDING_RULES}"""


def explain_user(blocks: list[Block], fields: list[ExtractedField]) -> str:
    return f"Extracted fields:\n{format_fields(fields)}\n\nDocument blocks:\n{format_blocks(blocks)}"


# ---------------------------------------------------------------- Q&A
def qa_system(lang: str) -> str:
    return f"""You answer a person's follow-up question about their government document.
Answer in {_lang_name(lang)} (answer_local) and simple English (answer_en), in 1-4 short sentences
suitable to be read aloud. If the document does not contain the answer, set found_in_document=false and say
so honestly; you may then add one sentence of general guidance clearly marked as general, not from the document.

{GROUNDING_RULES}"""


def qa_user(question: str, blocks: list[Block], fields: list[ExtractedField], history: list[tuple[str, str]]) -> str:
    hist = "\n".join(f"{r}: {t}" for r, t in history[-6:]) or "(none)"
    return (
        f"Conversation so far:\n{hist}\n\nExtracted fields:\n{format_fields(fields)}\n\n"
        f"Relevant document blocks:\n{format_blocks(blocks, 8000)}\n\nQuestion: {question}"
    )


# ---------------------------------------------------------------- translate
def translate_system(lang_name: str) -> str:
    return (
        f"Translate each input string into simple, natural {lang_name}. Keep names, numbers and legal terms "
        "unchanged. Return exactly one translation per input, in order."
    )


def translate_user(texts: list[str]) -> str:
    return "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
