"""Application settings (env vars / .env)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# Languages Sakshar supports end-to-end.
LANGUAGES: dict[str, dict[str, str]] = {
    "en": {"name": "English", "native": "English", "script": "Latn", "it2": "eng_Latn", "tess": "eng"},
    "hi": {"name": "Hindi", "native": "हिन्दी", "script": "Deva", "it2": "hin_Deva", "tess": "hin"},
    "mr": {"name": "Marathi", "native": "मराठी", "script": "Deva", "it2": "mar_Deva", "tess": "mar"},
    "pa": {"name": "Punjabi", "native": "ਪੰਜਾਬੀ", "script": "Guru", "it2": "pan_Guru", "tess": "pan"},
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str | None = None

    hf_token: str | None = None
    tesseract_cmd: str | None = None

    enable_voice: bool = True
    enable_paddle: bool = True
    translation_engine: Literal["llm", "indictrans2"] = "llm"
    device: Literal["cuda", "cpu"] = "cuda"

    review_threshold: float = 0.75
    review_threshold_strict: float = 0.85
    fuzzy_presence_threshold: float = 0.85
    ocr_fallback_conf: float = 0.60
    mask_pii: bool = True

    data_dir: Path = Path("data")

    # large-v3-turbo: ~1.8 GB in int8_float16; "small" produced unusable Punjabi on our tests
    whisper_model: str = "large-v3-turbo"
    indictrans_en_indic: str = "ai4bharat/indictrans2-en-indic-dist-200M"
    indictrans_indic_en: str = "ai4bharat/indictrans2-indic-en-dist-200M"
    tts_model: str = "ai4bharat/indic-parler-tts"
    tts_fallback_pa: str = "facebook/mms-tts-pan"

    @property
    def llm_enabled(self) -> bool:
        return bool(self.openai_api_key)

    def ensure_dirs(self) -> None:
        for sub in ("uploads", "pages", "audio"):
            (self.data_dir / sub).mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
