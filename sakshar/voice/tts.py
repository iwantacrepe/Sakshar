"""Text-to-speech with Indic Parler-TTS (hi / mr / pa / en); MMS-TTS fallback for Punjabi."""
from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

import numpy as np

from ..config import get_settings
from .registry import registry

log = logging.getLogger(__name__)

# Named speakers from the Indic Parler-TTS model card
DESCRIPTIONS = {
    "hi": "Divya speaks at a moderate pace with a warm, clear voice in a close-sounding recording with no background noise.",
    "mr": "Sunita speaks at a moderate pace with a clear voice in a close-sounding recording with no background noise.",
    "pa": "Divjot speaks at a moderate pace in a clear, close-sounding recording with no background noise.",
    "en": "Mary speaks at a moderate pace with a clear Indian English accent in a close-sounding recording.",
}


def _load_parler(device: str):
    import torch
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer

    s = get_settings()
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = ParlerTTSForConditionalGeneration.from_pretrained(s.tts_model, torch_dtype=dtype, token=s.hf_token)
    model = model.to(device).eval()
    tok = AutoTokenizer.from_pretrained(s.tts_model, token=s.hf_token)
    desc_tok = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path, token=s.hf_token)
    return _Parler(model, tok, desc_tok)


class _Parler:
    def __init__(self, model, tok, desc_tok):
        self.model, self.tok, self.desc_tok = model, tok, desc_tok

    def to(self, device: str):
        self.model.to(device)
        if device == "cpu":
            self.model.float()
        return self


def _load_mms(device: str):
    from transformers import AutoTokenizer, VitsModel

    s = get_settings()
    return _MMS(VitsModel.from_pretrained(s.tts_fallback_pa).to(device).eval(),
                AutoTokenizer.from_pretrained(s.tts_fallback_pa))


class _MMS:
    def __init__(self, model, tok):
        self.model, self.tok = model, tok

    def to(self, device: str):
        self.model.to(device)
        return self


registry.register("parler", _load_parler)
registry.register("mms_pa", _load_mms)


DIGIT_WORDS = {
    "hi": "शून्य एक दो तीन चार पाँच छह सात आठ नौ".split(),
    "mr": "शून्य एक दोन तीन चार पाच सहा सात आठ नऊ".split(),
    "pa": "ਸਿਫ਼ਰ ਇੱਕ ਦੋ ਤਿੰਨ ਚਾਰ ਪੰਜ ਛੇ ਸੱਤ ਅੱਠ ਨੌਂ".split(),
}
SEP_WORDS = {
    "hi": {".": "दशमलव", "/": "बटा"},
    "mr": {".": "पूर्णांक", "/": "बटा"},
    "pa": {".": "ਦਸ਼ਮਲਵ", "/": "ਬਟਾ"},
}


def speakable(text: str, lang: str) -> str:
    """Parler-TTS mispronounces numerals inside Indic text (214 came back from ASR as 'दुख शिचाला').
    Spell numbers digit by digit, the way khasra/survey numbers are read aloud: 45/2 -> चार पाँच बटा दो."""
    words = DIGIT_WORDS.get(lang)
    if not words:
        return text
    from ..pipeline.normalize import ascii_digits

    seps = SEP_WORDS[lang]

    def spell(m: re.Match) -> str:
        out = []
        for ch in m.group(0):
            if ch.isdigit():
                out.append(words[int(ch)])
            elif ch in seps:
                out.append(seps[ch])
        return " " + " ".join(out) + " "

    text = re.sub(r"(?<=\d),(?=\d)", "", ascii_digits(text))  # 1,25,000 -> 125000
    return re.sub(r"\s+", " ", re.sub(r"\d+(?:[./]+\d+)*", spell, text)).strip()


def _sentences(text: str, max_chars: int = 220) -> list[str]:
    parts = [p.strip() for p in re.split(r"(?<=[.!?।|])\s+", text) if p.strip()]
    out, cur = [], ""
    for p in parts:
        if len(cur) + len(p) < max_chars:
            cur = f"{cur} {p}".strip()
        else:
            if cur:
                out.append(cur)
            cur = p
    if cur:
        out.append(cur)
    return out or [text]


def _parler_synth(text: str, lang: str) -> tuple[np.ndarray, int]:
    import torch

    m = registry.get("parler")
    p: _Parler = m.obj
    desc = p.desc_tok(DESCRIPTIONS.get(lang, DESCRIPTIONS["hi"]), return_tensors="pt").to(m.device)
    pieces = []
    sr = p.model.config.sampling_rate
    with m.lock, torch.inference_mode():
        for sent in _sentences(text):
            prompt = p.tok(sent, return_tensors="pt").to(m.device)
            gen = p.model.generate(input_ids=desc.input_ids, attention_mask=desc.attention_mask,
                                   prompt_input_ids=prompt.input_ids, prompt_attention_mask=prompt.attention_mask)
            pieces.append(gen.float().cpu().numpy().squeeze())
            pieces.append(np.zeros(int(0.25 * sr), dtype=np.float32))
    return np.concatenate(pieces).astype(np.float32), sr


def _mms_synth(text: str) -> tuple[np.ndarray, int]:
    import torch

    m = registry.get("mms_pa")
    mm: _MMS = m.obj
    with m.lock, torch.inference_mode():
        inputs = mm.tok(text, return_tensors="pt").to(m.device)
        wav = mm.model(**inputs).waveform
    return wav.float().cpu().numpy().squeeze(), mm.model.config.sampling_rate


def synthesize(text: str, lang: str) -> Path:
    """Return a cached WAV path for (text, lang)."""
    import soundfile as sf

    s = get_settings()
    key = hashlib.sha1(f"v2|{lang}|{text}".encode("utf-8")).hexdigest()[:20]  # bump on pipeline changes
    out = s.data_dir / "audio" / f"{key}.wav"
    if out.exists():
        return out
    text = speakable(text, lang)
    try:
        audio, sr = _parler_synth(text, lang)
        if lang == "pa" and (audio.size == 0 or float(np.abs(audio).max()) < 1e-3):
            raise RuntimeError("silent Parler output for Punjabi")
    except Exception as e:
        if lang != "pa":
            raise
        log.warning("Parler-TTS Punjabi failed (%s); using MMS-TTS", e)
        audio, sr = _mms_synth(text)
    peak = float(np.abs(audio).max()) or 1.0
    sf.write(str(out), (audio / peak * 0.95).astype(np.float32), sr)
    return out
