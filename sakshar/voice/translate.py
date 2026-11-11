"""IndicTrans2 (distilled 200M) translation between English and hi / mr / pa."""
from __future__ import annotations

import logging

from ..config import LANGUAGES, get_settings
from .registry import registry

log = logging.getLogger(__name__)


def _processor():
    try:
        from IndicTransToolkit.processor import IndicProcessor
    except ImportError:
        from IndicTransToolkit import IndicProcessor
    return IndicProcessor(inference=True)


def _make_loader(model_name_attr: str):
    def load(device: str):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        s = get_settings()
        name = getattr(s, model_name_attr)
        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True, token=s.hf_token)
        dtype = torch.float16 if device == "cuda" else torch.float32
        model = AutoModelForSeq2SeqLM.from_pretrained(name, trust_remote_code=True, torch_dtype=dtype,
                                                      token=s.hf_token).to(device).eval()
        return _IT2(tok, model, _processor())

    return load


class _IT2:
    def __init__(self, tok, model, ip):
        self.tok, self.model, self.ip = tok, model, ip

    def to(self, device: str):
        self.model.to(device)
        if device == "cpu":
            self.model.float()
        return self


registry.register("it2_en_indic", _make_loader("indictrans_en_indic"))
registry.register("it2_indic_en", _make_loader("indictrans_indic_en"))


def translate(texts: list[str], src: str, tgt: str, batch_size: int = 8) -> list[str]:
    if src == tgt or not texts:
        return list(texts)
    import torch

    name = "it2_en_indic" if src == "en" else "it2_indic_en"
    if src != "en" and tgt != "en":  # indic -> indic via English pivot
        return translate(translate(texts, src, "en"), "en", tgt)
    m = registry.get(name)
    it2: _IT2 = m.obj
    src_code, tgt_code = LANGUAGES[src]["it2"], LANGUAGES[tgt]["it2"]
    out: list[str] = []
    with m.lock, torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            chunk = [t if t.strip() else "." for t in texts[i:i + batch_size]]
            batch = it2.ip.preprocess_batch(chunk, src_lang=src_code, tgt_lang=tgt_code)
            enc = it2.tok(batch, truncation=True, padding="longest", return_tensors="pt",
                          return_attention_mask=True).to(m.device)
            gen = it2.model.generate(**enc, use_cache=True, min_length=0, max_length=256, num_beams=4,
                                     num_return_sequences=1)
            if hasattr(it2.tok, "as_target_tokenizer"):
                with it2.tok.as_target_tokenizer():
                    dec = it2.tok.batch_decode(gen.detach().cpu().tolist(), skip_special_tokens=True,
                                               clean_up_tokenization_spaces=True)
            else:
                dec = it2.tok.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            out.extend(it2.ip.postprocess_batch(dec, lang=tgt_code))
    return out


# Land-revenue terms that general-purpose MT gets wrong (खसरा is also the Hindi word for
# "measles"; खाता -> "account"). They are swapped for their standard romanised revenue terms
# before translation, which IndicTrans2 passes through unchanged. Longest match first.
REVENUE_TERMS = {
    "दाखिल-खारिज": "dakhil-kharij (mutation)", "खातेदार": "khatedar (recorded holder)", "खतौनी": "khatauni",
    "खसरा": "khasra", "खाता": "khata", "जमाबंदी": "jamabandi", "भूमिधर": "bhumidhar", "फसली": "fasli",
    "सातबारा": "satbara (7/12 extract)", "भोगवटादार": "bhogwatdar (occupant)", "फेरफार": "ferfar (mutation)",
    "गट क्रमांक": "survey (gat) number", "गट नंबर": "survey (gat) number", "आकारणी": "akarni (assessment)",
    "ਜਮ੍ਹਾਂਬੰਦੀ": "jamabandi", "ਖਸਰਾ": "khasra", "ਖੇਵਟ": "khewat", "ਖਤੌਨੀ": "khatauni",
    "ਇੰਤਕਾਲ": "intkal (mutation)", "ਫ਼ਰਦ": "fard", "ਕਨਾਲ": "kanal", "ਮਰਲੇ": "marla",
}


def protect_terms(text: str) -> str:
    import unicodedata

    text = unicodedata.normalize("NFC", text)
    for term in sorted(REVENUE_TERMS, key=len, reverse=True):
        text = text.replace(unicodedata.normalize("NFC", term), REVENUE_TERMS[term])
    return text


def to_english(text: str, src: str) -> str:
    if src == "en" or not text.strip():
        return text
    try:
        return translate([protect_terms(text)], src, "en")[0]
    except Exception as e:
        log.warning("IndicTrans2 to English failed: %s", e)
        return text
