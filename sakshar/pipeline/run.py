"""Pipeline orchestrator.

Every CPU/GPU-heavy stage runs in a worker thread via asyncio.to_thread so the event
loop stays free to stream SSE progress events to the browser while a document is processed.
"""
from __future__ import annotations

import asyncio
import logging
import traceback
from collections import Counter
from pathlib import Path

from .. import pii
from ..config import get_settings
from ..events import Event, bus
from ..explain import qa
from ..explain.llm import friendly_error, get_llm
from ..models import Block, DocumentAnalysis, OCRWord, PageInfo
from ..store import get_store
from . import classify, extract, layout, ocr, preprocess, verify
from .workflows import get_workflow

log = logging.getLogger(__name__)
MAX_PAGES = 6

STAGES = [
    ("preprocess", "Cleaning and straightening the image"),
    ("ocr", "Detecting script and reading text"),
    ("layout", "Finding lines, columns and table cells"),
    ("classify", "Identifying the document type"),
    ("extract", "Extracting names, dates, amounts and numbers"),
    ("verify", "Checking values (checksums, confidence, source text)"),
    ("explain", "Writing a plain-language explanation"),
]
_LABEL = dict(STAGES)


def _emit(doc: DocumentAnalysis, stage: str, status: str = "running", detail: str = "") -> None:
    idx = [s for s, _ in STAGES].index(stage) if stage in _LABEL else len(STAGES)
    progress = (idx + (1 if status == "done" else 0.3)) / len(STAGES)
    bus.publish(doc.id, Event(stage=stage, label=_LABEL.get(stage, stage), status=status,
                              progress=round(progress, 3), detail=detail))


async def run_document(doc_id: str, path: Path, explain_lang: str | None = None) -> DocumentAnalysis:
    store = get_store()
    s = get_settings()
    doc = store.get(doc_id)
    assert doc is not None
    doc.status = "processing"
    stage = "preprocess"
    try:
        # 1. preprocess -------------------------------------------------------
        doc.stage = stage
        store.save(doc)
        _emit(doc, stage)
        raw_pages = await asyncio.to_thread(preprocess.load_pages, path)
        raw_pages = raw_pages[:MAX_PAGES]
        prepared = []
        for i, rgb in enumerate(raw_pages):
            page = await asyncio.to_thread(preprocess.prepare_page, rgb, i)
            rel = f"pages/{doc.id}_{i}.png"
            await asyncio.to_thread(preprocess.save_rgb, page.rgb, s.data_dir / rel)
            w, h = page.size
            doc.pages.append(PageInfo(index=i, image=rel, width=w, height=h, skew_deg=round(page.skew_deg, 2)))
            prepared.append(page)
        skew = ", ".join(f"{p.skew_deg:+.1f}°" for p in prepared)
        _emit(doc, stage, "done", f"{len(prepared)} page(s), deskew {skew}")

        # 2. OCR ----------------------------------------------------------------
        stage = "ocr"
        _emit(doc, stage)
        words_by_page: list[list[OCRWord]] = []
        votes: Counter = Counter()
        engines = []
        for page in prepared:
            res = await asyncio.to_thread(ocr.run_ocr, page)
            words_by_page.append(res.words)
            votes[(res.script, res.lang)] += sum(len(w.text) for w in res.words)
            engines.append(res.engine)
            _emit(doc, stage, detail=f"page {page.index + 1}: {res.engine}, {res.script}/{res.lang}, "
                                     f"mean conf {res.mean_conf:.0%}")
        if not any(words_by_page):
            raise ValueError("No text could be read from this document. Try a sharper, well-lit photo.")
        (doc.script, doc.lang), _ = votes.most_common(1)[0]
        doc.ocr_engine = ", ".join(sorted(set(engines)))
        _emit(doc, stage, "done", f"{doc.lang} ({doc.script}) via {doc.ocr_engine}")

        # 3. layout -------------------------------------------------------------
        stage = "layout"
        _emit(doc, stage)
        raw_blocks: list[Block] = []
        for page, words in zip(prepared, words_by_page):
            raw_blocks += await asyncio.to_thread(layout.build_blocks, words, len(raw_blocks),
                                                  page.hlines, page.vlines)
        # Aadhaar numbers are masked before any text leaves the machine or is stored;
        # only the local regex/checksum step sees raw_blocks.
        masked_blocks = [b.model_copy(update={"text": pii.mask_text(b.text),
                                              "value": pii.mask_text(b.value) if b.value else None})
                         for b in raw_blocks]
        _emit(doc, stage, "done", f"{len(raw_blocks)} text blocks")

        # 4. classify -----------------------------------------------------------
        stage = "classify"
        _emit(doc, stage)
        llm = get_llm()
        doc.doc_type, doc.doc_type_conf = await asyncio.to_thread(classify.classify, masked_blocks, llm)
        wf = get_workflow(doc.doc_type)
        _emit(doc, stage, "done", f"{doc.doc_type.value} ({doc.doc_type_conf:.0%})")

        # 5. extract ------------------------------------------------------------
        stage = "extract"
        _emit(doc, stage)
        regex_fields = await asyncio.to_thread(extract.regex_extract, raw_blocks, doc.doc_type)
        llm_fields = []
        if llm is not None:
            try:
                llm_fields = await asyncio.to_thread(llm.extract, masked_blocks, doc.doc_type.value,
                                                     wf.llm_field_specs)
            except Exception as e:
                log.warning("LLM extraction failed: %s", type(e).__name__)
                _emit(doc, stage, detail=f"AI extraction skipped: {friendly_error(e)}")
        fields = extract.merge(regex_fields, llm_fields, raw_blocks, doc.doc_type)
        _emit(doc, stage, "done", f"{len(fields)} fields ({len(regex_fields)} rule-based, {len(llm_fields)} LLM)")

        # 6. verify -------------------------------------------------------------
        stage = "verify"
        _emit(doc, stage)
        fields = await asyncio.to_thread(verify.verify_all, fields, raw_blocks, doc.doc_type)
        for f in fields:
            if f.name == "aadhaar" and f.value:
                f.display_value = pii.mask_aadhaar(f.value)
                if s.mask_pii:
                    f.value = f.display_value
        missing = [n for n in wf.required if not any(f.name == n for f in fields)]
        doc.fields, doc.blocks = fields, masked_blocks
        flagged = sum(f.needs_review for f in fields)
        detail = f"{flagged} value(s) need your review"
        if missing:
            detail += f"; not found: {', '.join(missing)}"
        _emit(doc, stage, "done", detail)
        store.save(doc)

        # 7. explain ------------------------------------------------------------
        stage = "explain"
        _emit(doc, stage)
        lang = explain_lang or doc.lang or "hi"
        try:
            doc.explanations[lang] = await asyncio.to_thread(qa.explain, doc, lang)
            n_bad = sum(not st.grounded for st in doc.explanations[lang].statements)
            _emit(doc, stage, "done", f"{len(doc.explanations[lang].statements)} points"
                                      + (f", {n_bad} unverified" if n_bad else ""))
        except Exception as e:
            log.warning("explanation failed: %s", type(e).__name__)
            _emit(doc, stage, "error", friendly_error(e))

        doc.status, doc.stage = "done", "done"
        store.save(doc)
        bus.publish(doc.id, Event(stage="done", label="Done", status="done", progress=1.0))
    except Exception as e:
        log.error("pipeline failed at %s: %s\n%s", stage, e, traceback.format_exc())
        doc.status, doc.error = "error", f"{_LABEL.get(stage, stage)}: {e}"
        store.save(doc)
        bus.publish(doc.id, Event(stage="error", label="Failed", status="error", detail=doc.error))
    return doc
