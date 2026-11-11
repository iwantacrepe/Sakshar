"""Speech recognition with faster-whisper (restricted to hi / mr / pa / en)."""
from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ..config import LANGUAGES, get_settings
from .registry import registry

log = logging.getLogger(__name__)

SUPPORTED = list(LANGUAGES)  # en, hi, mr, pa

# Domain vocabulary primes the decoder toward land-record / form terms in each language.
INITIAL_PROMPTS = {
    "hi": "यह खसरा, खतौनी, जमाबंदी, खाता संख्या, तहसील, पट्टा और सरकारी आवेदन पत्र के बारे में प्रश्न है।",
    "mr": "हा सातबारा उतारा, गट क्रमांक, भोगवटादार, फेरफार, तालुका आणि सरकारी अर्जाबद्दल प्रश्न आहे.",
    "pa": "ਇਹ ਜਮ੍ਹਾਂਬੰਦੀ, ਫ਼ਰਦ, ਖੇਵਟ, ਖਸਰਾ, ਇੰਤਕਾਲ, ਕਨਾਲ ਮਰਲੇ ਅਤੇ ਸਰਕਾਰੀ ਅਰਜ਼ੀ ਬਾਰੇ ਸਵਾਲ ਹੈ।",
    "en": "This is a question about a land record, khasra number, application form or government certificate.",
}


def _load(device: str):
    from faster_whisper import WhisperModel

    s = get_settings()
    compute = "int8_float16" if device == "cuda" else "int8"
    return WhisperModel(s.whisper_model, device=device, compute_type=compute)


registry.register("whisper", _load, movable=False)


def load_audio(path: Path) -> np.ndarray:
    """16 kHz mono float32. PyAV first; ffmpeg CLI fallback for header-less browser webm blobs."""
    from faster_whisper import decode_audio

    try:
        audio = decode_audio(str(path), sampling_rate=16000)
        if audio.size > 1600:
            return audio
    except Exception as e:
        log.info("PyAV decode failed (%s); falling back to ffmpeg", e)
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "a.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-ac", "1", "-ar", "16000", str(wav)],
                       check=True, timeout=60)
        return decode_audio(str(wav), sampling_rate=16000)


def transcribe(path: Path, lang: str | None = None) -> tuple[str, str, float]:
    """Return (text, language, language_probability)."""
    m = registry.get("whisper")
    audio = load_audio(path)
    vad = dict(min_silence_duration_ms=500)
    with m.lock:
        prob = 1.0
        if lang not in SUPPORTED:
            # Language detection runs eagerly inside transcribe(); segments are a lazy generator
            # we simply never iterate, so this costs one encoder pass.
            _, info = m.obj.transcribe(audio, vad_filter=True, vad_parameters=vad, beam_size=1)
            probs = dict(info.all_language_probs or [(info.language, info.language_probability)])
            lang = max(SUPPORTED, key=lambda l: probs.get(l, 0.0))
            prob = probs.get(lang, 0.0)
        segments, _ = m.obj.transcribe(
            audio,
            language=lang,
            task="transcribe",
            beam_size=5,
            vad_filter=True,  # trims dead air -> no initial_prompt echo hallucinations
            vad_parameters=vad,
            initial_prompt=INITIAL_PROMPTS.get(lang),
            condition_on_previous_text=False,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
    return fix_script(text, lang), lang, prob


# Devanagari (U+0900) and Gurmukhi (U+0A00) blocks follow the same ISCII layout, so a fixed
# codepoint offset transliterates between them. Whisper sometimes writes Punjabi in Devanagari.
_OFFSET = 0x0A00 - 0x0900


def fix_script(text: str, lang: str) -> str:
    deva = sum(0x0900 <= ord(c) <= 0x097F for c in text)
    guru = sum(0x0A00 <= ord(c) <= 0x0A7F for c in text)
    if lang == "pa" and deva > guru:
        return "".join(chr(ord(c) + _OFFSET) if 0x0901 <= ord(c) <= 0x096F else c for c in text)
    if lang in ("hi", "mr") and guru > deva:
        return "".join(chr(ord(c) - _OFFSET) if 0x0A01 <= ord(c) <= 0x0A6F else c for c in text)
    return text
