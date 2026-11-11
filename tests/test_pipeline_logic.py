"""Layout, classification, extraction, verification, grounding, retrieval."""
from conftest import make_block, make_word

from sakshar.explain import grounding, retrieve
from sakshar.models import DocType, ExtractedField, FieldCategory, Statement
from sakshar.pipeline import classify, extract, layout, verify


# ---------------------------------------------------------------- layout
def test_xy_cut_keeps_table_columns_apart():
    words = []
    for r in range(4):
        y = 0.2 + r * 0.05
        words.append(make_word(f"मालिक{r}", 0.10, y, 0.22, y + 0.02))
        words.append(make_word(f"नाम{r}", 0.23, y, 0.32, y + 0.02))
        words.append(make_word(f"{200 + r}", 0.62, y, 0.70, y + 0.02))  # survey no. column
    blocks = layout.build_blocks(words)
    for b in blocks:
        has_name = any(t in b.text for t in ("मालिक", "नाम"))
        has_num = any(str(200 + r) in b.text for r in range(4))
        assert not (has_name and has_num), f"columns merged: {b.text}"
    assert any(b.text == "मालिक0 नाम0" for b in blocks)


def test_key_value_detection():
    words = [make_word("खसरा", 0.1, 0.1, 0.18, 0.12), make_word("संख्या", 0.185, 0.1, 0.26, 0.12),
             make_word(":", 0.265, 0.1, 0.27, 0.12), make_word("214", 0.28, 0.1, 0.32, 0.12)]
    b = layout.build_blocks(words)[0]
    assert b.key == "खसरा संख्या" and b.value == "214"


def test_label_paired_with_right_cell():
    words = [make_word("Applicant Name:", 0.1, 0.1, 0.3, 0.12), make_word("Meena", 0.6, 0.1, 0.7, 0.12),
             make_word("Kumari", 0.71, 0.1, 0.8, 0.12)]
    blocks = layout.build_blocks(words)
    label = next(b for b in blocks if b.text.startswith("Applicant"))
    assert label.value == "Meena Kumari"


# ---------------------------------------------------------------- classify
def test_classify_rules(blocks_khatauni):
    dt, conf = classify.classify(blocks_khatauni)
    assert dt == DocType.land_record_up_khatauni and conf > 0.5


def test_classify_712():
    bl = [make_block(0, "गाव नमुना सात सातबारा उतारा"), make_block(1, "भोगवटादाराचे नाव : सुनील पाटील")]
    assert classify.classify(bl)[0] == DocType.land_record_mh_712


def test_classify_unknown_is_generic():
    assert classify.classify([make_block(0, "hello world")])[0] == DocType.generic


# ---------------------------------------------------------------- extract + verify
def test_regex_extract_khatauni(blocks_khatauni):
    fields = {f.name: f for f in extract.regex_extract(blocks_khatauni, DocType.land_record_up_khatauni)}
    assert fields["khasra_no"].value == "214"
    assert fields["khata_no"].value == "00127"
    assert "राम प्रसाद" in fields["owner_names"].value
    assert fields["mutation_date"].value == "12/03/2023"
    assert fields["land_revenue"].source_block_ids == ["b9"]


def test_verify_passes_clean_field(blocks_khatauni):
    f = ExtractedField(name="khasra_no", label="Khasra", value="214", category=FieldCategory.reference,
                       source_block_ids=["b6"], source="both")
    v = verify.verify_all([f], blocks_khatauni, DocType.land_record_up_khatauni)[0]
    assert not v.needs_review, v.review_reasons
    assert v.confidence > 0.9


def test_verify_gates_low_ocr_conf_on_financial():
    blocks = [make_block(0, "लगान : ₹ 45.00", conf=0.80)]  # >= normal threshold, < strict threshold
    f = ExtractedField(name="land_revenue", label="Land revenue", value="₹ 45.00", category=FieldCategory.financial,
                       source_block_ids=["b0"], source="both")
    v = verify.verify_all([f], blocks, DocType.land_record_up_khatauni)[0]
    assert v.needs_review and any("OCR confidence" in r for r in v.review_reasons)


def test_verify_gates_value_not_in_source(blocks_khatauni):
    f = ExtractedField(name="khasra_no", label="Khasra", value="9981", category=FieldCategory.reference,
                       source_block_ids=["b6"], source="llm")
    v = verify.verify_all([f], blocks_khatauni, DocType.land_record_up_khatauni)[0]
    assert v.needs_review and any("not clearly found" in r for r in v.review_reasons)


def test_verify_gates_validator_failure():
    blocks = [make_block(0, "दाखिल-खारिज आदेश दिनांक : 45/19/2023")]
    f = ExtractedField(name="mutation_date", label="Mutation date", value="45/19/2023", category=FieldCategory.date,
                       source_block_ids=["b0"], source="both")
    v = verify.verify_all([f], blocks, DocType.land_record_up_khatauni)[0]
    assert v.needs_review and v.validation_error


def test_merge_marks_disagreement(blocks_khatauni):
    rf = extract.regex_extract(blocks_khatauni, DocType.land_record_up_khatauni)
    lf = [ExtractedField(name="khasra_no", label="khasra_no", value="2140", source_block_ids=["b6"], source="llm"),
          ExtractedField(name="khata_no", label="khata_no", value="00127", source_block_ids=["b4"], source="llm")]
    merged = {f.name: f for f in extract.merge(rf, lf, blocks_khatauni, DocType.land_record_up_khatauni)}
    assert merged["khata_no"].source == "both"
    assert any("disagree" in r for r in merged["khasra_no"].review_reasons)


# ---------------------------------------------------------------- grounding
def test_grounding_rejects_missing_block_and_mismatched_value(blocks_khatauni):
    good = Statement(text_en="The khasra number is 214.", text_local="खसरा संख्या 214 है।", cited_block_ids=["b6"])
    wrong_value = Statement(text_en="The area is 0.999 hectare.", text_local="क्षेत्रफल 0.999 हेक्टेयर है।",
                            cited_block_ids=["b7"])
    ghost = Statement(text_en="Revenue is 9999.", text_local="लगान 9999 है।", cited_block_ids=["b404"])
    out = grounding.ground([good, wrong_value, ghost], blocks_khatauni)
    assert len(out) == 2  # ghost: numeric claim, no valid citation -> dropped
    assert out[0].grounded
    assert not out[1].grounded and "0.999" in out[1].note


def test_grounding_repairs_citation(blocks_khatauni):
    st = Statement(text_en="Khata 00127 lists the owner.", text_local="खाता 00127", cited_block_ids=["b5"])
    out = grounding.ground([st], blocks_khatauni)[0]
    assert out.grounded and "b4" in out.cited_block_ids


# ---------------------------------------------------------------- retrieval
def test_hybrid_retrieval_tolerates_ocr_typo():
    blocks = [make_block(0, "ग्राम : रामपुर कलां"), make_block(1, "तहसील : नगीना"),
              make_block(2, "खसस 214 क्षेत्रफल 0.632"), make_block(3, "जिला : बिजनौर")]
    top, score = retrieve.search(["खसरा"], blocks, k=1)[0]
    assert top.id == "b2" and score > 0


def test_field_labels_bridge_english_questions(blocks_khatauni):
    f = ExtractedField(name="area", label="Area", value="0.632 हेक्टेयर", source_block_ids=["b7"])
    top, _ = retrieve.search(["what is the area of my land"], blocks_khatauni, [f], k=1)[0]
    assert top.id == "b7"
