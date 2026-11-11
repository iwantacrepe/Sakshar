"""End-to-end API flow with mocked OCR and a mocked OpenAI client."""
import re
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from sakshar import pii
from sakshar.explain import llm as llm_mod
from sakshar.explain.llm import LLM, AnswerOut, ClassifyOut, ExplainOut, ExtractOut, LLMField, StatementOut
from sakshar.models import BBox, OCRResult, OCRWord
from sakshar.pipeline import ocr
from sakshar.store import Store, set_store

AADHAAR_BASE = "23456789012"
AADHAAR = AADHAAR_BASE + pii.verhoeff_check_digit(AADHAAR_BASE)

LINES = [
    ("खतौनी फसली वर्ष 1431", 0.97),
    ("ग्राम : रामपुर कलां", 0.96),
    ("खाता संख्या : 00127", 0.95),
    ("खातेदार का नाम : राम प्रसाद पुत्र श्याम लाल", 0.94),
    ("खसरा संख्या : 214", 0.96),
    ("क्षेत्रफल : 0.632 हेक्टेयर", 0.62),  # smudged -> low confidence
    (f"आधार संख्या : {AADHAAR[:4]} {AADHAAR[4:8]} {AADHAAR[8:]}", 0.93),
    ("दाखिल-खारिज आदेश दिनांक : 12/03/2023", 0.95),
]


def fake_ocr(page):
    words = []
    for i, (text, conf) in enumerate(LINES):
        y = 0.06 + i * 0.06
        words.append(OCRWord(text=text, bbox=BBox(x1=0.08, y1=y, x2=0.75, y2=y + 0.03), conf=conf, engine="mock"))
    return OCRResult(words=words, script="Deva", lang="hi", engine="mock", mean_conf=0.9)


class FakeLLM(LLM):
    def __init__(self):
        self.model = "fake"
        self.calls = []

    def parse(self, system, user, schema, temperature=0.1):
        self.calls.append((schema.__name__, user))
        assert AADHAAR not in user and AADHAAR[:8] not in user.replace(" ", ""), "unmasked Aadhaar sent to LLM"
        if schema is ClassifyOut:
            return ClassifyOut(doc_type="land_record_up_khatauni", confidence=0.9)
        if schema is ExtractOut:
            ids = dict(re.findall(r"\[(b\d+)\] (.*)", user))
            bid = next(k for k, v in ids.items() if "खसरा" in v)
            return ExtractOut(fields=[LLMField(name="khasra_no", value="214", cited_block_ids=[bid])])
        if schema is ExplainOut:
            return ExplainOut(
                title="खतौनी (भूमि का रिकॉर्ड)",
                summary_local="यह राम प्रसाद की ज़मीन का रिकॉर्ड है।",
                statements=[
                    StatementOut(text_en="Khasra number is 214.", text_local="खसरा संख्या 214 है।", cited_block_ids=["b4"]),
                    StatementOut(text_en="Area is 9.99 hectare.", text_local="क्षेत्रफल 9.99 हेक्टेयर है।", cited_block_ids=["b5"]),
                ],
                action_items_local=["क्षेत्रफल मूल दस्तावेज़ में जाँचें।"],
            )
        if schema is AnswerOut:
            return AnswerOut(found_in_document=True, answer_en="Khasra is 214.", answer_local="आपका खसरा नंबर 214 है।",
                             cited_block_ids=["b4"])
        raise AssertionError(schema)


@pytest.fixture
def client():
    set_store(Store(":memory:"))
    ocr.set_override(fake_ocr)
    fake = FakeLLM()
    llm_mod.set_llm(fake)
    from sakshar.main import create_app

    with TestClient(create_app()) as c:
        c.fake = fake
        yield c
    ocr.set_override(None)
    llm_mod.set_llm(None)


def png_bytes():
    img = np.full((900, 700, 3), 250, np.uint8)
    cv2.putText(img, "sample", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return cv2.imencode(".png", img)[1].tobytes()


def wait_done(client, doc_id, timeout=20):
    t0 = time.time()
    while time.time() - t0 < timeout:
        d = client.get(f"/api/documents/{doc_id}").json()
        if d["status"] in ("done", "error"):
            return d
        time.sleep(0.1)
    raise TimeoutError(d["stage"])


def test_rejects_bad_upload(client):
    r = client.post("/api/documents", files={"file": ("x.exe", b"MZ", "application/octet-stream")})
    assert r.status_code == 400


def test_full_flow(client):
    r = client.post("/api/documents", files={"file": ("khatauni.png", png_bytes(), "image/png")}, data={"lang": "hi"})
    assert r.status_code == 202
    doc_id = r.json()["id"]
    d = wait_done(client, doc_id)
    assert d["status"] == "done", d["error"]
    assert d["doc_type"] == "land_record_up_khatauni"
    assert d["lang"] == "hi"

    fields = {f["name"]: f for f in d["fields"]}
    assert fields["khasra_no"]["source"] == "both" and not fields["khasra_no"]["needs_review"]
    assert fields["area"]["needs_review"], "smudged low-confidence area must be flagged"
    # Aadhaar: validated, masked at rest, never stored in full
    assert fields["aadhaar"]["value"].startswith("XXXX-XXXX-")
    assert AADHAAR not in client.get(f"/api/documents/{doc_id}").text.replace(" ", "")

    exp = d["explanations"]["hi"]
    assert exp["statements"][0]["grounded"] is True
    assert exp["statements"][1]["grounded"] is False  # 9.99 is not in the document

    page = client.get(f"/doc/{doc_id}")
    assert page.status_code == 200 and "खसरा" in page.text

    # Q&A
    r = client.post(f"/api/documents/{doc_id}/ask", data={"question": "मेरा खसरा नंबर क्या है?", "lang": "hi"})
    assert r.status_code == 200 and "214" in r.text and 'data-cites="b4"' in r.text
    # chat history is stored and shown again after a page reload
    reloaded = client.get(f"/doc/{doc_id}").text
    assert "मेरा खसरा नंबर क्या है?" in reloaded and "आपका खसरा नंबर 214 है।" in reloaded

    # review loop: confirm the flagged area, explanation cache is invalidated
    idx = [f["name"] for f in d["fields"]].index("area")
    r = client.post(f"/api/documents/{doc_id}/fields/{idx}", data={"action": "correct", "value": "0.632 हेक्टेयर", "lang": "hi"})
    assert r.status_code == 200 and r.headers.get("HX-Trigger") == "fields-changed"
    d2 = client.get(f"/api/documents/{doc_id}").json()
    area = d2["fields"][idx]
    assert area["source"] == "user" and area["confirmed"] and not area["needs_review"]
    assert d2["explanations"] == {}

    r = client.post(f"/api/documents/{doc_id}/explain?lang=hi")
    assert r.status_code == 200 and "खसरा संख्या 214" in r.text


def test_sse_stream_reports_stages(client):
    r = client.post("/api/documents", files={"file": ("a.png", png_bytes(), "image/png")})
    doc_id = r.json()["id"]
    wait_done(client, doc_id)
    with client.stream("GET", f"/api/documents/{doc_id}/events") as resp:
        body = "".join(resp.iter_text())
    assert "event: stage" in body and "event: done" in body
    assert "Extracting names" in body
