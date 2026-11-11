"""Core data models shared across the pipeline, explainer and web layer."""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class BBox(BaseModel):
    """Bounding box normalized to [0, 1] relative to the deskewed page image."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def h(self) -> float:
        return self.y2 - self.y1

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @staticmethod
    def union(boxes: list["BBox"]) -> "BBox":
        return BBox(
            x1=min(b.x1 for b in boxes),
            y1=min(b.y1 for b in boxes),
            x2=max(b.x2 for b in boxes),
            y2=max(b.y2 for b in boxes),
        )


class OCRWord(BaseModel):
    text: str
    bbox: BBox
    conf: float = Field(ge=0, le=1)
    page: int = 0
    engine: Literal["paddle", "tesseract", "mock"] = "tesseract"


class Block(BaseModel):
    """A line (or key:value cell) of text with a stable id the LLM can cite."""

    id: str
    text: str
    bbox: BBox
    page: int = 0
    conf: float = 1.0
    word_confs: list[float] = Field(default_factory=list)
    region: int = 0
    row: int | None = None  # table row index (count of ruling lines above), when inside a ruled table
    col: int | None = None  # table column index (count of vertical rules to the left), when between rules
    key: str | None = None  # set when the line looks like "label: value"
    value: str | None = None


class PageInfo(BaseModel):
    index: int
    image: str  # relative path served under /data
    width: int
    height: int
    skew_deg: float = 0.0


class OCRResult(BaseModel):
    words: list[OCRWord]
    script: str
    lang: str
    script_conf: float = 1.0
    engine: str = "tesseract"
    mean_conf: float = 0.0


class DocType(str, Enum):
    land_record_up_khatauni = "land_record_up_khatauni"
    land_record_mh_712 = "land_record_mh_712"
    land_record_pb_jamabandi = "land_record_pb_jamabandi"
    application_form = "application_form"
    certificate = "certificate"
    generic = "generic"


class FieldCategory(str, Enum):
    identity = "identity"
    legal = "legal"
    financial = "financial"
    date = "date"
    reference = "reference"
    location = "location"
    other = "other"


class ExtractedField(BaseModel):
    name: str
    label: str
    value: str | None
    normalized: str | None = None
    category: FieldCategory = FieldCategory.other
    source_block_ids: list[str] = Field(default_factory=list)
    bbox: BBox | None = None
    source: Literal["regex", "llm", "both", "user"] = "regex"
    confidence: float = 0.0
    needs_review: bool = False
    review_reasons: list[str] = Field(default_factory=list)
    validation_error: str | None = None
    confirmed: bool = False
    display_value: str | None = None  # masked for PII


class Statement(BaseModel):
    text_en: str
    text_local: str
    cited_block_ids: list[str] = Field(default_factory=list)
    grounded: bool = True
    note: str | None = None


class Explanation(BaseModel):
    lang: str
    title: str = ""
    summary_local: str = ""
    statements: list[Statement] = Field(default_factory=list)
    action_items_local: list[str] = Field(default_factory=list)


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    text: str
    lang: str
    cited_block_ids: list[str] = Field(default_factory=list)
    audio: str | None = None
    text_en: str | None = None


class DocumentAnalysis(BaseModel):
    id: str
    filename: str
    status: Literal["queued", "processing", "done", "error"] = "queued"
    stage: str = "queued"
    error: str | None = None
    pages: list[PageInfo] = Field(default_factory=list)
    ocr_engine: str | None = None
    script: str | None = None
    lang: str | None = None
    doc_type: DocType | None = None
    doc_type_conf: float = 0.0
    blocks: list[Block] = Field(default_factory=list)
    fields: list[ExtractedField] = Field(default_factory=list)
    explanations: dict[str, Explanation] = Field(default_factory=dict)

    def block_map(self) -> dict[str, Block]:
        return {b.id: b for b in self.blocks}
