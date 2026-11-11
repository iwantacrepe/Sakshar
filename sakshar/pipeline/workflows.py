"""Route each document type to its processing workflow."""
from __future__ import annotations

from dataclasses import dataclass

from ..explain.prompts import DOC_TYPE_HINTS
from ..models import DocType
from .schemas import FieldSpec, schema_for

IDENTITY_FIELDS = {"aadhaar", "pan"}  # regex + checksum only; never sent to the LLM


@dataclass
class Workflow:
    doc_type: DocType
    fields: list[FieldSpec]
    prompt_hint: str

    @property
    def llm_field_specs(self) -> list[tuple[str, str]]:
        return [(f.name, f.label) for f in self.fields if f.name not in IDENTITY_FIELDS]

    @property
    def required(self) -> list[str]:
        return [f.name for f in self.fields if f.required]


def get_workflow(doc_type: DocType) -> Workflow:
    return Workflow(doc_type=doc_type, fields=schema_for(doc_type),
                    prompt_hint=DOC_TYPE_HINTS.get(doc_type.value, DOC_TYPE_HINTS["generic"]))
