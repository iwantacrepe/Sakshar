from __future__ import annotations

import asyncio
import html
import json
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from sse_starlette.sse import EventSourceResponse

from ..config import LANGUAGES, get_settings
from ..events import bus
from ..explain import qa
from ..explain.llm import friendly_error
from ..models import ChatTurn, DocumentAnalysis
from ..pipeline import verify
from ..pipeline.run import run_document
from ..store import get_store
from .templating import templates

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

ALLOWED = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp", ".pdf"}
MAX_BYTES = 20 * 1024 * 1024
_tasks: set[asyncio.Task] = set()  # keep strong refs to background pipeline tasks


def _doc_or_404(doc_id: str) -> DocumentAnalysis:
    doc = get_store().get(doc_id)
    if doc is None:
        raise HTTPException(404, "Document not found")
    return doc


def _lang(lang: str | None, doc: DocumentAnalysis | None = None) -> str:
    if lang in LANGUAGES:
        return lang
    return (doc.lang if doc and doc.lang else None) or "hi"


# ---------------------------------------------------------------- upload + progress
@router.post("/documents")
async def upload(request: Request, file: UploadFile = File(...), lang: str = Form("auto")):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED:
        raise HTTPException(400, f"Unsupported file type {ext or '(none)'}; use an image or PDF")
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "File too large (max 20 MB)")
    if not data:
        raise HTTPException(400, "Empty file")

    s = get_settings()
    doc_id = uuid.uuid4().hex[:12]
    path = s.data_dir / "uploads" / f"{doc_id}{ext}"
    path.write_bytes(data)
    doc = DocumentAnalysis(id=doc_id, filename=Path(file.filename or "document").name)
    get_store().save(doc)

    task = asyncio.create_task(run_document(doc_id, path, lang if lang in LANGUAGES else None))
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)

    if request.headers.get("HX-Request"):
        return Response(status_code=204, headers={"HX-Redirect": f"/doc/{doc_id}"})
    return JSONResponse({"id": doc_id, "url": f"/doc/{doc_id}"}, status_code=202)


@router.get("/documents/{doc_id}/events")
async def events(request: Request, doc_id: str):
    doc = _doc_or_404(doc_id)
    tpl = templates.env.get_template("partials/stage.html")

    async def gen():
        q, history = bus.subscribe(doc_id)
        state: dict[str, dict] = {}
        try:
            def render(ev):
                if ev.stage not in ("done", "error"):
                    prev = state.get(ev.stage, {})
                    state[ev.stage] = {"status": ev.status, "detail": ev.detail or prev.get("detail", "")}
                return tpl.render(state=state, progress=ev.progress,
                                  error=ev.detail if ev.stage == "error" else None)

            # Already finished before we connected?
            if doc.status in ("done", "error") and not history:
                yield {"event": doc.status, "data": doc.error or "ok"}
                return
            for ev in history:
                yield {"event": "stage", "data": render(ev)}
                if ev.stage in ("done", "error"):
                    yield {"event": ev.stage, "data": ev.detail or "ok"}
                    return
            while True:
                if await request.is_disconnected():
                    return
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=15)
                except asyncio.TimeoutError:
                    continue  # sse-starlette sends pings
                yield {"event": "stage", "data": render(ev)}
                if ev.stage in ("done", "error"):
                    yield {"event": ev.stage, "data": ev.detail or "ok"}
                    return
        finally:
            bus.unsubscribe(doc_id, q)

    return EventSourceResponse(gen(), ping=10)


@router.get("/documents/{doc_id}")
def get_document(doc_id: str):
    return JSONResponse(json.loads(_doc_or_404(doc_id).model_dump_json()))


# ---------------------------------------------------------------- review loop
@router.post("/documents/{doc_id}/fields/{idx}", response_class=HTMLResponse)
def review_field(request: Request, doc_id: str, idx: int, action: str = Form(...),
                 value: str = Form(""), lang: str = Form("hi")):
    store = get_store()
    doc = _doc_or_404(doc_id)
    if not 0 <= idx < len(doc.fields):
        raise HTTPException(404, "Field not found")
    f = doc.fields[idx]
    if action == "confirm":
        f.confirmed, f.needs_review = True, False
    elif action == "correct":
        value = value.strip()
        if not value:
            raise HTTPException(400, "Value required")
        store.log_correction(doc_id, f.name, f.value, value)
        f.value, f.display_value, f.source, f.confirmed = value, None, "user", True
        verify.verify_field(f, doc.block_map(), doc.doc_type)
    else:
        raise HTTPException(400, "Unknown action")
    doc.explanations = {}  # explanations must be regenerated with the corrected values
    store.save(doc)
    resp = templates.TemplateResponse(request, "partials/fields.html",
                                      {"doc": doc, "ui_lang": _lang(lang, doc)})
    resp.headers["HX-Trigger"] = "fields-changed"
    return resp


# ---------------------------------------------------------------- explanation
@router.post("/documents/{doc_id}/explain", response_class=HTMLResponse)
async def explain(request: Request, doc_id: str, lang: str | None = None, refresh: bool = False):
    store = get_store()
    doc = _doc_or_404(doc_id)
    if doc.status != "done":
        raise HTTPException(409, "Document is still processing")
    lang = _lang(lang, doc)
    if refresh or lang not in doc.explanations:
        try:
            doc.explanations[lang] = await asyncio.to_thread(qa.explain, doc, lang)
            store.save(doc)
        except Exception as e:
            return HTMLResponse(f'<div class="error">{html.escape(friendly_error(e))}</div>', status_code=200)
    return templates.TemplateResponse(request, "partials/explanation.html",
                                      {"doc": doc, "explanation": doc.explanations[lang], "ui_lang": lang})


# ---------------------------------------------------------------- Q&A
def _speak(text: str, lang: str) -> str | None:
    if not get_settings().enable_voice:
        return None
    try:
        from ..voice import tts

        return f"/audio/{tts.synthesize(text, lang).name}"
    except Exception as e:
        log.warning("TTS unavailable: %s", e)
        return None


@router.post("/documents/{doc_id}/ask", response_class=HTMLResponse)
async def ask(request: Request, doc_id: str, question: str = Form(...), lang: str = Form("hi"),
              speak: bool = Form(False)):
    store = get_store()
    doc = _doc_or_404(doc_id)
    lang = _lang(lang, doc)
    question = question.strip()[:1000]
    if not question:
        raise HTTPException(400, "Empty question")
    history = store.turns(doc_id)
    q_en = await asyncio.to_thread(_to_english, question, lang)
    user = ChatTurn(role="user", text=question, lang=lang, text_en=q_en)
    try:
        reply = await asyncio.to_thread(qa.answer, doc, question, lang, history, q_en)
    except Exception as e:
        return templates.TemplateResponse(request, "partials/chat_message.html", {
            "turns": [user, ChatTurn(role="assistant", text="⚠ " + friendly_error(e), lang="en")],
            "ui_lang": lang, "autoplay": False})
    if speak:
        reply.audio = await asyncio.to_thread(_speak, reply.text, lang)
    store.add_turn(doc_id, user)
    store.add_turn(doc_id, reply)
    return templates.TemplateResponse(request, "partials/chat_message.html",
                                      {"turns": [user, reply], "ui_lang": lang, "autoplay": speak})


def _to_english(text: str, lang: str) -> str | None:
    """IndicTrans2 query translation for cross-lingual retrieval (best effort)."""
    if lang == "en" or not get_settings().enable_voice:
        return None
    try:
        from ..voice import translate

        return translate.to_english(text, lang)
    except Exception as e:
        log.info("query translation skipped: %s", e)
        return None


@router.post("/documents/{doc_id}/voice", response_class=HTMLResponse)
async def voice(request: Request, doc_id: str, audio: UploadFile = File(...), lang: str = Form("auto")):
    s = get_settings()
    if not s.enable_voice:
        raise HTTPException(503, "Voice is disabled (ENABLE_VOICE=false)")
    store = get_store()
    doc = _doc_or_404(doc_id)
    data = await audio.read()
    if not data or len(data) > 10 * 1024 * 1024:
        raise HTTPException(400, "Audio missing or too large")
    tmp = s.data_dir / "audio" / f"in_{uuid.uuid4().hex[:10]}.webm"
    tmp.write_bytes(data)
    try:
        from ..voice import asr

        text, heard_lang, _ = await asyncio.to_thread(asr.transcribe, tmp, lang if lang in LANGUAGES else None)
    except Exception as e:
        log.exception("ASR failed")
        return HTMLResponse(f'<div class="error">Could not process audio: {e}</div>', status_code=502)
    finally:
        tmp.unlink(missing_ok=True)
    if not text:
        return HTMLResponse('<div class="error">No speech detected. Please try again.</div>')

    history = store.turns(doc_id)
    q_en = await asyncio.to_thread(_to_english, text, heard_lang)
    user = ChatTurn(role="user", text=text, lang=heard_lang, text_en=q_en)
    try:
        reply = await asyncio.to_thread(qa.answer, doc, text, heard_lang, history, q_en)
    except Exception as e:
        # still show what was heard, so speech recognition problems and answer problems are distinguishable
        return templates.TemplateResponse(request, "partials/chat_message.html", {
            "turns": [user, ChatTurn(role="assistant", text="⚠ " + friendly_error(e), lang="en")],
            "ui_lang": heard_lang, "autoplay": False})
    reply.audio = await asyncio.to_thread(_speak, reply.text, heard_lang)
    store.add_turn(doc_id, user)
    store.add_turn(doc_id, reply)
    return templates.TemplateResponse(request, "partials/chat_message.html",
                                      {"turns": [user, reply], "ui_lang": heard_lang, "autoplay": True})


@router.post("/speak", response_class=JSONResponse)
async def speak(text: str = Form(...), lang: str = Form("hi")):
    url = await asyncio.to_thread(_speak, text[:1500], _lang(lang))
    if url is None:
        raise HTTPException(503, "Text-to-speech unavailable")
    return {"audio": url}
