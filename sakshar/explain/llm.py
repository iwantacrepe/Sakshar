"""OpenAI wrapper using Pydantic structured outputs (chat.completions.parse).

Works with any OpenAI-compatible endpoint via OPENAI_BASE_URL.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import TypeVar

from pydantic import BaseModel

from ..config import LANGUAGES, get_settings
from ..models import Block, ExtractedField
from . import prompts

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------- response schemas
class ClassifyOut(BaseModel):
    doc_type: str
    confidence: float


class LLMField(BaseModel):
    name: str
    value: str
    cited_block_ids: list[str]


class ExtractOut(BaseModel):
    fields: list[LLMField]


class StatementOut(BaseModel):
    text_en: str
    text_local: str
    cited_block_ids: list[str]


class ExplainOut(BaseModel):
    title: str
    summary_local: str
    statements: list[StatementOut]
    action_items_local: list[str]


class AnswerOut(BaseModel):
    found_in_document: bool
    answer_en: str
    answer_local: str
    cited_block_ids: list[str]


class TranslateOut(BaseModel):
    translations: list[str]


# ---------------------------------------------------------------- client
class LLM:
    def __init__(self, client=None, model: str | None = None):
        s = get_settings()
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=s.openai_api_key, base_url=s.openai_base_url or None, max_retries=3, timeout=90)
        self.client = client
        self.model = model or s.openai_model

    def parse(self, system: str, user: str, schema: type[T], temperature: float = 0.1) -> T:
        completions = self.client.chat.completions
        parse = getattr(completions, "parse", None) or self.client.beta.chat.completions.parse
        resp = parse(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format=schema,
            temperature=temperature,
        )
        msg = resp.choices[0].message
        if getattr(msg, "refusal", None):
            raise RuntimeError(f"model refused: {msg.refusal}")
        if msg.parsed is None:
            raise RuntimeError("model returned no structured output")
        return msg.parsed

    # -- tasks ---------------------------------------------------------------
    def classify(self, text: str, types: list[str]) -> tuple[str, float]:
        out = self.parse(prompts.CLASSIFY_SYSTEM, prompts.classify_user(text, types), ClassifyOut)
        dt = out.doc_type if out.doc_type in types else "generic"
        return dt, max(0.0, min(1.0, out.confidence))

    def extract(self, blocks: list[Block], doc_type: str, field_specs: list[tuple[str, str]]) -> list[ExtractedField]:
        out = self.parse(prompts.EXTRACT_SYSTEM, prompts.extract_user(blocks, doc_type, field_specs), ExtractOut)
        allowed = {n for n, _ in field_specs}
        return [
            ExtractedField(name=f.name, label=f.name, value=f.value.strip(), source_block_ids=f.cited_block_ids, source="llm")
            for f in out.fields
            if f.name in allowed and f.value and f.value.strip()
        ]

    def explain(self, blocks: list[Block], fields: list[ExtractedField], doc_type: str, lang: str) -> ExplainOut:
        return self.parse(
            prompts.explain_system(lang, doc_type),
            prompts.explain_user(blocks, fields),
            ExplainOut,
            temperature=0.2,
        )

    def answer(self, question: str, blocks: list[Block], fields: list[ExtractedField], lang: str,
               history: list[tuple[str, str]] | None = None) -> AnswerOut:
        return self.parse(
            prompts.qa_system(lang),
            prompts.qa_user(question, blocks, fields, history or []),
            AnswerOut,
        )

    def translate(self, texts: list[str], lang: str) -> list[str]:
        if not texts:
            return []
        name = LANGUAGES[lang]["name"]
        out = self.parse(prompts.translate_system(name), prompts.translate_user(texts), TranslateOut)
        return out.translations if len(out.translations) == len(texts) else texts


def friendly_error(e: Exception) -> str:
    """User-facing message for LLM failures; never echoes provider error bodies (they can
    contain fragments of the API key)."""
    name = type(e).__name__
    status = getattr(e, "status_code", None)
    if name == "AuthenticationError" or status == 401:
        return "The OpenAI API key was rejected. Check OPENAI_API_KEY in .env and restart the server."
    if name == "PermissionDeniedError" or status == 403:
        return f"This API key has no access to the model '{get_settings().openai_model}'. Change OPENAI_MODEL in .env."
    if name == "NotFoundError" or status == 404:
        return f"Model '{get_settings().openai_model}' was not found. Change OPENAI_MODEL in .env."
    if name == "RateLimitError" or status == 429:
        return "OpenAI rate limit or quota reached. Check billing/quota, or try again in a minute."
    if name in ("APIConnectionError", "APITimeoutError"):
        return "Could not reach the OpenAI API (network). Check your internet connection."
    log.warning("LLM error: %s", e)
    return "The explanation service failed. See the server log for details."


_override: LLM | None = None


def set_llm(llm: LLM | None) -> None:
    """Test hook."""
    global _override
    _override = llm
    get_llm.cache_clear()


@lru_cache
def get_llm() -> LLM | None:
    if _override is not None:
        return _override
    if not get_settings().llm_enabled:
        log.warning("OPENAI_API_KEY not set: LLM features disabled")
        return None
    return LLM()
