import os
import sys
import tempfile
from pathlib import Path

# Configure settings BEFORE sakshar.config is imported anywhere.
_TMP = tempfile.mkdtemp(prefix="sakshar-test-")
os.environ.update({
    "DATA_DIR": _TMP,
    "OPENAI_API_KEY": "",
    "ENABLE_VOICE": "false",
    "ENABLE_PADDLE": "false",
    "TRANSLATION_ENGINE": "llm",
})
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from sakshar.models import BBox, Block, OCRWord  # noqa: E402


def make_block(i: int, text: str, conf: float = 0.95, y: float | None = None, x1: float = 0.1, x2: float = 0.6) -> Block:
    y = 0.05 + i * 0.04 if y is None else y
    return Block(id=f"b{i}", text=text, bbox=BBox(x1=x1, y1=y, x2=x2, y2=y + 0.025), conf=conf, word_confs=[conf])


def make_word(text: str, x1: float, y1: float, x2: float, y2: float, conf: float = 0.95) -> OCRWord:
    return OCRWord(text=text, bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2), conf=conf)


@pytest.fixture
def blocks_khatauni():
    texts = [
        "खतौनी (अधिकार अभिलेख) फसली वर्ष 1431",
        "ग्राम : रामपुर कलां",
        "तहसील : नगीना",
        "जिला : बिजनौर",
        "खाता संख्या : 00127",
        "खातेदार का नाम : राम प्रसाद पुत्र श्याम लाल",
        "खसरा संख्या : 214",
        "क्षेत्रफल : 0.632 हेक्टेयर",
        "दाखिल-खारिज आदेश दिनांक : 12/03/2023",
        "लगान : ₹ 45.00",
    ]
    return [make_block(i, t) for i, t in enumerate(texts)]
