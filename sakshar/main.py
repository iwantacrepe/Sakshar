"""FastAPI application factory.

Run:  uvicorn sakshar.main:app --reload
"""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .config import get_settings

PKG = Path(__file__).parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def create_app() -> FastAPI:
    s = get_settings()
    app = FastAPI(title="Sakshar", version="0.1.0")
    app.mount("/static", StaticFiles(directory=PKG / "static"), name="static")
    app.mount("/pages", StaticFiles(directory=s.data_dir / "pages"), name="pages")
    app.mount("/audio", StaticFiles(directory=s.data_dir / "audio"), name="audio")

    from .web import api, pages

    app.include_router(pages.router)
    app.include_router(api.router)

    @app.get("/healthz")
    def healthz():
        from .pipeline.tess import available_langs, tesseract_cmd
        from .voice.registry import registry

        return {
            "llm": s.llm_enabled,
            "model": s.openai_model,
            "tesseract": tesseract_cmd(),
            "tess_langs": sorted(available_langs()),
            "voice": s.enable_voice,
            "loaded_models": registry.status(),
        }

    return app


app = create_app()
