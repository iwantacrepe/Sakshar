"""Per-document-type field schemas and validators."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Callable

from .. import pii
from ..models import DocType, FieldCategory
from . import normalize

Validator = Callable[[str], tuple[str | None, str | None]]  # value -> (normalized, error)


# ---------------------------------------------------------------- validators
def v_date(value: str) -> tuple[str | None, str | None]:
    d = normalize.parse_date(value)
    if d is None:
        return None, "not a recognizable date"
    if not (date(1900, 1, 1) <= d <= date(date.today().year + 5, 12, 31)):
        return d.isoformat(), "date outside plausible range"
    return d.isoformat(), None


def v_amount(value: str) -> tuple[str | None, str | None]:
    a = normalize.parse_amount(value)
    if a is None:
        digits = normalize.ascii_digits(value).replace(",", "")
        m = re.search(r"\d+(?:\.\d+)?", digits)
        if not m:
            return None, "no numeric amount"
        a = float(m.group(0))
    if a < 0 or a > 1e10:
        return None, "implausible amount"
    return normalize.format_inr(a), None


def v_area(value: str) -> tuple[str | None, str | None]:
    found = normalize.find_areas(value)
    if found:
        _, num, unit = found[0]
        return f"{num:g} {unit}", None
    t = normalize.ascii_digits(value).strip()
    # Maharashtra 7/12 convention: H.AA.SS = hectare . are . square metre (unit is in the column header)
    m = re.fullmatch(r"(\d+)\.(\d{2})\.(\d{2})", t)
    if m:
        ha, are, sqm = (int(g) for g in m.groups())
        return f"{ha} ha {are} are {sqm} sq m ({ha + are / 100 + sqm / 10000:g} hectare)", None
    # Punjab kanal-marla written as "4-12"
    m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", t)
    if m:
        return f"{m.group(1)} kanal {m.group(2)} marla", None
    if re.fullmatch(r"\d+(?:\.\d+)?", t):
        return t, "area unit not stated"
    return None, "no area value or unknown unit"


def v_aadhaar(value: str) -> tuple[str | None, str | None]:
    if not pii.is_valid_aadhaar(value):
        return pii.mask_aadhaar(value), "fails Aadhaar Verhoeff checksum"
    return pii.mask_aadhaar(value), None


def v_pan(value: str) -> tuple[str | None, str | None]:
    v = value.strip().upper().replace(" ", "")
    return (v, None) if pii.is_valid_pan(v) else (v, "not a valid PAN format")


def v_ref(value: str) -> tuple[str | None, str | None]:
    t = normalize.ascii_digits(value).strip()
    if not re.search(r"\d", t):
        return t, "reference number has no digits"
    if len(t) > 40:
        return t, "reference number too long"
    return t, None


def v_text(value: str) -> tuple[str | None, str | None]:
    t = normalize.clean(value)
    if len(t) < 2:
        return t, "too short"
    if re.fullmatch(r"[\d\W_]+", t):
        return t, "expected a name/text, got only digits or symbols"
    return t, None


# ---------------------------------------------------------------- schemas
@dataclass
class FieldSpec:
    name: str
    label: str  # English label (UI translates via LLM explanation)
    category: FieldCategory
    validator: Validator = v_text
    aliases: list[str] = field(default_factory=list)  # label words seen on documents
    required: bool = False
    multi: bool = False


COMMON = [
    FieldSpec("aadhaar", "Aadhaar number", FieldCategory.identity, v_aadhaar, ["aadhaar", "आधार", "ਆਧਾਰ"]),
    FieldSpec("pan", "PAN", FieldCategory.identity, v_pan, ["pan", "पैन", "ਪੈਨ"]),
]

SCHEMAS: dict[DocType, list[FieldSpec]] = {
    DocType.land_record_up_khatauni: [
        FieldSpec("owner_names", "Owner / Bhumidhar name(s)", FieldCategory.legal, v_text,
                  ["खातेदार", "भूमिधर", "नाम खातेदार", "owner"], required=True, multi=True),
        FieldSpec("father_name", "Father's / husband's name", FieldCategory.identity, v_text, ["पिता", "पति", "पुत्र"]),
        FieldSpec("khata_no", "Khata number", FieldCategory.reference, v_ref, ["खाता संख्या", "खाता सं", "khata"]),
        FieldSpec("khasra_no", "Khasra number", FieldCategory.reference, v_ref, ["खसरा", "गाटा संख्या", "khasra"], required=True),
        FieldSpec("area", "Area", FieldCategory.legal, v_area, ["क्षेत्रफल", "रकबा", "area"]),
        FieldSpec("village", "Village", FieldCategory.location, v_text, ["ग्राम", "गांव", "मौजा"]),
        FieldSpec("tehsil", "Tehsil", FieldCategory.location, v_text, ["तहसील"]),
        FieldSpec("district", "District", FieldCategory.location, v_text, ["जिला", "जनपद"]),
        FieldSpec("fasli_year", "Fasli year", FieldCategory.date, v_ref, ["फसली", "fasli"]),
        FieldSpec("mutation_date", "Mutation (dakhil-kharij) date", FieldCategory.date, v_date, ["दाखिल-खारिज", "नामांतरण", "आदेश दिनांक"]),
        FieldSpec("land_revenue", "Land revenue", FieldCategory.financial, v_amount, ["लगान", "मालगुजारी"]),
    ],
    DocType.land_record_mh_712: [
        FieldSpec("owner_names", "Occupant (Bhogwatdar) name(s)", FieldCategory.legal, v_text,
                  ["भोगवटादाराचे नाव", "भोगवटादार", "खातेदार"], required=True, multi=True),
        FieldSpec("survey_no", "Survey / Gat number", FieldCategory.reference, v_ref,
                  ["भूमापन क्रमांक", "गट क्रमांक", "सर्व्हे नंबर", "survey"], required=True),
        FieldSpec("occupancy_class", "Occupancy class", FieldCategory.legal, v_text, ["भोगवटादार वर्ग", "वर्ग"]),
        FieldSpec("area", "Area (hectare.are)", FieldCategory.legal, v_area, ["क्षेत्र", "एकूण क्षेत्र", "हेक्टर"]),
        FieldSpec("khata_no", "Khata number", FieldCategory.reference, v_ref, ["खाते क्रमांक", "खाता क्रमांक"]),
        FieldSpec("village", "Village", FieldCategory.location, v_text, ["गाव"]),
        FieldSpec("taluka", "Taluka", FieldCategory.location, v_text, ["तालुका"]),
        FieldSpec("district", "District", FieldCategory.location, v_text, ["जिल्हा"]),
        FieldSpec("mutation_no", "Mutation (ferfar) number", FieldCategory.reference, v_ref, ["फेरफार क्रमांक", "फेरफार"]),
        FieldSpec("mutation_date", "Mutation (ferfar) date", FieldCategory.date, v_date, ["फेरफार दिनांक", "फेरफार तारीख"]),
        FieldSpec("assessment", "Assessment (akarni)", FieldCategory.financial, v_amount, ["आकारणी", "रु."]),
        FieldSpec("encumbrance", "Other rights / encumbrance", FieldCategory.legal, v_text, ["इतर हक्क", "बोजा", "कर्ज"]),
    ],
    DocType.land_record_pb_jamabandi: [
        FieldSpec("owner_names", "Owner (malik) name(s)", FieldCategory.legal, v_text,
                  ["ਮਾਲਕ", "ਨਾਮ ਮਾਲਕ", "owner"], required=True, multi=True),
        FieldSpec("cultivator", "Cultivator (kashtkar)", FieldCategory.legal, v_text, ["ਕਾਸ਼ਤਕਾਰ", "ਕਾਸ਼ਤਕਾਰ ਦਾ ਨਾਮ"]),
        FieldSpec("khewat_no", "Khewat number", FieldCategory.reference, v_ref, ["ਖੇਵਟ", "ਖੇਵਟ ਨੰਬਰ"]),
        FieldSpec("khatauni_no", "Khatauni number", FieldCategory.reference, v_ref, ["ਖਤੌਨੀ", "ਖਤੌਨੀ ਨੰਬਰ"]),
        FieldSpec("khasra_no", "Khasra / killa number", FieldCategory.reference, v_ref, ["ਖਸਰਾ", "ਕਿੱਲਾ", "ਮੁਰੱਬਾ"], required=True),
        FieldSpec("area", "Area (kanal-marla)", FieldCategory.legal, v_area, ["ਰਕਬਾ", "ਕਨਾਲ", "ਮਰਲੇ"]),
        FieldSpec("village", "Village", FieldCategory.location, v_text, ["ਪਿੰਡ"]),
        FieldSpec("tehsil", "Tehsil", FieldCategory.location, v_text, ["ਤਹਿਸੀਲ"]),
        FieldSpec("district", "District", FieldCategory.location, v_text, ["ਜ਼ਿਲ੍ਹਾ", "ਜਿਲ੍ਹਾ"]),
        FieldSpec("year", "Jamabandi year", FieldCategory.date, v_ref, ["ਸਾਲ", "ਜਮ੍ਹਾਂਬੰਦੀ ਸਾਲ"]),
        FieldSpec("mutation_no", "Mutation (intkal) number", FieldCategory.reference, v_ref, ["ਇੰਤਕਾਲ", "ਇੰਤਕਾਲ ਨੰਬਰ"]),
        FieldSpec("mutation_date", "Mutation (intkal) date", FieldCategory.date, v_date, ["ਇੰਤਕਾਲ ਮਿਤੀ", "ਮਿਤੀ ਇੰਤਕਾਲ"]),
        FieldSpec("revenue", "Land revenue (mamla)", FieldCategory.financial, v_amount, ["ਮਾਮਲਾ"]),
    ],
    DocType.application_form: [
        FieldSpec("applicant_name", "Applicant name", FieldCategory.identity, v_text,
                  ["आवेदक का नाम", "आवेदक", "अर्जदाराचे नाव", "अर्जदार", "ਬਿਨੈਕਾਰ ਦਾ ਨਾਮ", "ਬਿਨੈਕਾਰ", "applicant name", "name"], required=True),
        FieldSpec("application_no", "Application number", FieldCategory.reference, v_ref,
                  ["आवेदन संख्या", "आवेदन क्रमांक", "अर्ज क्रमांक", "ਅਰਜ਼ੀ ਨੰਬਰ", "application no", "application number", "ref"], required=True),
        FieldSpec("application_date", "Application date", FieldCategory.date, v_date,
                  ["आवेदन तिथि", "दिनांक", "अर्जाचा दिनांक", "ਮਿਤੀ", "date of application", "date"]),
        FieldSpec("scheme", "Scheme / service", FieldCategory.other, v_text, ["योजना", "सेवा", "ਯੋਜਨਾ", "scheme", "service"]),
        FieldSpec("fee", "Fee / amount", FieldCategory.financial, v_amount, ["शुल्क", "फीस", "रक्कम", "ਫੀਸ", "fee", "amount"]),
        FieldSpec("office", "Office / department", FieldCategory.other, v_text, ["कार्यालय", "विभाग", "ਦਫ਼ਤਰ", "office", "department"]),
        FieldSpec("due_date", "Last date / next hearing", FieldCategory.date, v_date, ["अंतिम तिथि", "अंतिम दिनांक", "ਆਖਰੀ ਮਿਤੀ", "last date", "due date"]),
    ],
    DocType.certificate: [
        FieldSpec("holder_name", "Certificate holder", FieldCategory.identity, v_text,
                  ["श्री", "श्रीमती", "कुमारी", "नाव", "ਸ਼੍ਰੀ", "name"], required=True),
        FieldSpec("certificate_no", "Certificate number", FieldCategory.reference, v_ref,
                  ["प्रमाण पत्र संख्या", "क्रमांक", "ਨੰਬਰ", "certificate no"], required=True),
        FieldSpec("issue_date", "Issue date", FieldCategory.date, v_date, ["जारी करने की तिथि", "दिनांक", "ਮਿਤੀ", "date of issue", "date"]),
        FieldSpec("valid_till", "Valid until", FieldCategory.date, v_date, ["वैधता", "ਵੈਧਤਾ", "valid"]),
        FieldSpec("annual_income", "Annual income", FieldCategory.financial, v_amount, ["वार्षिक आय", "वार्षिक उत्पन्न", "ਸਾਲਾਨਾ ਆਮਦਨ", "annual income"]),
        FieldSpec("issuing_authority", "Issuing authority", FieldCategory.legal, v_text, ["तहसीलदार", "उपजिल्हाधिकारी", "ਤਹਿਸੀਲਦਾਰ", "issuing authority"]),
    ],
    DocType.generic: [
        FieldSpec("person_names", "Names", FieldCategory.identity, v_text, ["नाम", "नाव", "ਨਾਮ", "name"], multi=True),
        FieldSpec("reference_no", "Reference number", FieldCategory.reference, v_ref, ["संख्या", "क्रमांक", "ਨੰਬਰ", "no."]),
        FieldSpec("dates", "Dates", FieldCategory.date, v_date, ["दिनांक", "तिथि", "ਮਿਤੀ", "date"], multi=True),
        FieldSpec("amounts", "Amounts", FieldCategory.financial, v_amount, ["रुपये", "रक्कम", "ਰੁਪਏ", "amount"], multi=True),
    ],
}


def schema_for(doc_type: DocType) -> list[FieldSpec]:
    return SCHEMAS.get(doc_type, SCHEMAS[DocType.generic]) + COMMON


def spec_by_name(doc_type: DocType) -> dict[str, FieldSpec]:
    return {f.name: f for f in schema_for(doc_type)}
