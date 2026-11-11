"""Explanation generation and grounded follow-up Q&A."""
from __future__ import annotations

import logging

from ..config import get_settings
from ..models import ChatTurn, DocumentAnalysis, Explanation, Statement
from . import grounding, retrieve
from .llm import get_llm

log = logging.getLogger(__name__)

NO_LLM = {
    "en": "The explanation service is not configured (OPENAI_API_KEY missing). Extracted fields are shown on the left.",
    "hi": "व्याख्या सेवा चालू नहीं है (OPENAI_API_KEY नहीं मिला)। निकाली गई जानकारी बाईं ओर दिखाई गई है।",
    "mr": "स्पष्टीकरण सेवा सुरू नाही (OPENAI_API_KEY मिळाली नाही). काढलेली माहिती डावीकडे दाखवली आहे.",
    "pa": "ਵਿਆਖਿਆ ਸੇਵਾ ਚਾਲੂ ਨਹੀਂ ਹੈ (OPENAI_API_KEY ਨਹੀਂ ਮਿਲੀ)। ਕੱਢੀ ਗਈ ਜਾਣਕਾਰੀ ਖੱਬੇ ਪਾਸੇ ਦਿਖਾਈ ਗਈ ਹੈ।",
}


def explain(doc: DocumentAnalysis, lang: str) -> Explanation:
    llm = get_llm()
    if llm is None:
        return Explanation(lang=lang, title=doc.filename, summary_local=NO_LLM.get(lang, NO_LLM["en"]))
    out = llm.explain(doc.blocks, doc.fields, doc.doc_type.value if doc.doc_type else "generic", lang)
    statements = [Statement(text_en=s.text_en, text_local=s.text_local, cited_block_ids=s.cited_block_ids)
                  for s in out.statements]

    if get_settings().translation_engine == "indictrans2" and lang != "en":
        try:
            from ..voice import translate

            local = translate.translate([s.text_en for s in statements], src="en", tgt=lang)
            for s, t in zip(statements, local):
                s.text_local = t
        except Exception as e:  # keep the LLM's own local text
            log.warning("IndicTrans2 translation failed, keeping LLM text: %s", e)

    statements = grounding.ground(statements, doc.blocks)
    return Explanation(lang=lang, title=out.title, summary_local=out.summary_local,
                       statements=statements, action_items_local=out.action_items_local)


def answer(doc: DocumentAnalysis, question: str, lang: str, history: list[ChatTurn],
           question_en: str | None = None) -> ChatTurn:
    llm = get_llm()
    if llm is None:
        return ChatTurn(role="assistant", text=NO_LLM.get(lang, NO_LLM["en"]), lang=lang)
    queries = [question] + ([question_en] if question_en and question_en != question else [])
    ctx = retrieve.context_blocks(queries, doc.blocks, doc.fields)
    hist = [(t.role, t.text) for t in history]
    out = llm.answer(question if not question_en else f"{question}\n(English: {question_en})", ctx, doc.fields, lang, hist)
    bmap = doc.block_map()
    cited = [c for c in out.cited_block_ids if c in bmap]
    # numeric claims in the answer must be backed by the cited blocks
    st = grounding.check_statement(
        Statement(text_en=out.answer_en, text_local=out.answer_local, cited_block_ids=cited), bmap
    )
    text = out.answer_local
    if st is not None and not st.grounded and out.found_in_document:
        text += {"hi": " (कृपया मूल दस्तावेज़ में जाँचें)", "mr": " (कृपया मूळ कागदपत्रात तपासा)",
                 "pa": " (ਕਿਰਪਾ ਕਰਕੇ ਅਸਲ ਦਸਤਾਵੇਜ਼ ਵਿੱਚ ਜਾਂਚੋ)"}.get(lang, " (please verify on the original)")
    return ChatTurn(role="assistant", text=text, text_en=out.answer_en, lang=lang,
                    cited_block_ids=st.cited_block_ids if st else [])
