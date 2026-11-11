from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

from ..config import LANGUAGES
from ..store import get_store
from .templating import templates

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"docs": get_store().list(12)})


@router.get("/doc/{doc_id}", response_class=HTMLResponse)
def document(request: Request, doc_id: str, lang: str | None = None):
    store = get_store()
    doc = store.get(doc_id)
    if doc is None:
        raise HTTPException(404, "Document not found")
    ui_lang = lang if lang in LANGUAGES else (next(iter(doc.explanations), None) or doc.lang or "hi")
    return templates.TemplateResponse(request, "document.html", {
        "doc": doc,
        "ui_lang": ui_lang,
        "explanation": doc.explanations.get(ui_lang),
        "turns": store.turns(doc_id),
        "blocks_json": [b.model_dump(include={"id", "text", "bbox", "page", "conf"}) for b in doc.blocks],
    })
