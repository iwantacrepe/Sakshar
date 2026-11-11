"""Regression tests for bugs found while running the real pipeline."""
import threading

import numpy as np

from sakshar.models import BBox
from sakshar.pipeline import ocr
from sakshar.pipeline.preprocess import PreparedPage
from sakshar.voice.translate import protect_terms


def _page():
    img = np.full((100, 200, 3), 255, np.uint8)
    return PreparedPage(index=0, rgb=img, clahe=img, binary=img[..., 0], skew_deg=0.0, hlines=[], vlines=[])


def test_revenue_terms_are_protected_before_translation():
    # खसरा also means "measles"; IndicTrans2 translated the query literally before this fix
    assert protect_terms("मेरी ज़मीन का खसरा नंबर क्या है?") == "मेरी ज़मीन का khasra नंबर क्या है?"
    assert "khatedar" in protect_terms("खातेदार का नाम")  # longest match wins over खाता
    assert "khewat" in protect_terms("ਖੇਵਟ ਨੰਬਰ ਕੀ ਹੈ")


def test_paddle_runtime_failure_falls_back_instead_of_failing(monkeypatch):
    def boom(*a):
        raise RuntimeError("Unknown exception")

    monkeypatch.setattr(ocr, "_detect_job", boom)
    monkeypatch.setattr(ocr, "_recognize_job", boom)
    assert ocr.paddle_detect(_page()) is None
    assert ocr.paddle_recognize(_page(), "en") is None


def test_paddle_work_always_runs_on_one_thread():
    seen = set()

    def job(*a):
        seen.add(threading.get_ident())

    threads = [threading.Thread(target=ocr._on_paddle_thread, args=(job,)) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(seen) == 1


def test_digit_repair_replaces_misread_numbers_but_not_words():
    from conftest import make_word

    indic = [make_word("खाता", 0.1, 0.1, 0.2, 0.13), make_word("0027", 0.3, 0.1, 0.38, 0.13, conf=0.5)]
    eng = [make_word("00127", 0.3, 0.1, 0.39, 0.13, conf=0.9), make_word("214", 0.5, 0.1, 0.55, 0.13, conf=0.9),
           make_word("4", 0.1, 0.1, 0.2, 0.13, conf=0.9)]  # sits on a Devanagari word: must be ignored
    texts = sorted(w.text for w in ocr.digit_repair(indic, eng))
    assert texts == ["00127", "214", "खाता"]


def test_numbers_are_spelled_for_tts():
    from sakshar.voice.tts import speakable

    assert speakable("खसरा संख्या 214 है", "hi") == "खसरा संख्या दो एक चार है"
    assert speakable("गट 45/2", "mr") == "गट चार पाच बटा दोन"
    assert speakable("ਰੁ. 1,20", "pa") == "ਰੁ. ਇੱਕ ਦੋ ਸਿਫ਼ਰ"
    assert speakable("Khasra 214", "en") == "Khasra 214"


def test_whisper_script_fix():
    from sakshar.voice.asr import fix_script

    assert fix_script("खसरा नम्बर", "pa") == "ਖਸਰਾ ਨਮ੍ਬਰ"
    assert fix_script("ਖਸਰਾ", "hi") == "खसरा"
    assert fix_script("खसरा", "hi") == "खसरा"


def test_torch_then_paddle_import_order_works():
    """Paddle-before-torch breaks torch on Windows (WinError 127); the app must load torch first."""
    import subprocess
    import sys

    code = ("import os; os.environ['ENABLE_VOICE']='true'; "
            "from sakshar.pipeline import ocr; ocr._torch_first(); import paddle; import torch; print('ok')")
    try:
        import paddle  # noqa: F401  (skip where paddle isn't installed)
    except ImportError:
        return
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300,
                         cwd=str(__import__("pathlib").Path(__file__).resolve().parent.parent))
    assert out.stdout.strip().endswith("ok"), out.stderr[-500:]


def test_user_correction_renormalizes_value():
    """A corrected area showed the stale normalized value of the misread one ('7332 hectare')."""
    from conftest import make_block

    from sakshar.models import DocType, ExtractedField, FieldCategory
    from sakshar.pipeline import verify

    f = ExtractedField(name="area", label="Area", value="0.632 हेक्टेयर", normalized="7332 hectare",
                       category=FieldCategory.legal, source="user", source_block_ids=["b0"])
    v = verify.verify_field(f, {"b0": make_block(0, "क्षेत्रफल : (७3३2 हेक्टेयर")}, DocType.land_record_up_khatauni)
    assert v.normalized == "0.632 hectare" and not v.needs_review


def test_area_conventions():
    from sakshar.pipeline.schemas import v_area

    assert v_area("1.20.00") == ("1 ha 20 are 0 sq m (1.2 hectare)", None)
    assert v_area("8 ਕਨਾਲ 4 ਮਰਲੇ")[1] is None
    assert v_area("0.632")[1] == "area unit not stated"
