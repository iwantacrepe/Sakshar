"""Script detection, normalization, PII validators."""
from datetime import date

from sakshar import pii
from sakshar.pipeline import normalize, script_detect


# ---------------------------------------------------------------- script detection
def test_detect_hindi():
    g = script_detect.detect("खतौनी में खातेदार का नाम राम प्रसाद है और खसरा संख्या 214 है। तहसील नगीना जिला बिजनौर")
    assert (g.script, g.lang) == ("Deva", "hi")


def test_detect_marathi():
    g = script_detect.detect("गाव नमुना सात. भोगवटादाराचे नाव सुनील पाटील आहे आणि तालुका हवेली, जिल्हा पुणे. क्षेत्र १.२० हेक्टर आहे.")
    assert (g.script, g.lang) == ("Deva", "mr")


def test_detect_punjabi():
    g = script_detect.detect("ਫ਼ਰਦ ਜਮ੍ਹਾਂਬੰਦੀ ਸਾਲ 2021-22 ਪਿੰਡ ਘੁੰਮਣ ਕਲਾਂ ਮਾਲਕ ਗੁਰਮੀਤ ਸਿੰਘ")
    assert (g.script, g.lang) == ("Guru", "pa")


def test_detect_english():
    g = script_detect.detect("Application No INC/2024/0045123 Applicant Name Meena Kumari")
    assert (g.script, g.lang) == ("Latn", "en")


def test_indic_wins_over_english_labels():
    g = script_detect.detect("Application No: 123 Name: Ram  आवेदक का नाम राम प्रसाद पिता का नाम श्याम लाल")
    assert g.script == "Deva"


# ---------------------------------------------------------------- normalize
def test_indic_digits():
    assert normalize.ascii_digits("खसरा २१४, ਖਸਰਾ ੨੩") == "खसरा 214, ਖਸਰਾ 23"


def test_amounts():
    assert normalize.parse_amount("लगान ₹ 1,25,000/-") == 125000.0
    assert normalize.parse_amount("४५,००० रुपये") == 45000.0
    assert normalize.parse_amount("Fee Paid : Rs. 30/-") == 30.0
    assert normalize.parse_amount("ਮਾਮਲਾ ਰੁ. 120") == 120.0
    assert normalize.parse_amount("खसरा 214") is None


def test_format_inr():
    assert normalize.format_inr(1234567) == "₹12,34,567"
    assert normalize.format_inr(45.5) == "₹45.50"


def test_dates():
    assert normalize.parse_date("आदेश दिनांक 12/03/2023") == date(2023, 3, 12)
    assert normalize.parse_date("१२-०३-२०२३") == date(2023, 3, 12)
    assert normalize.parse_date("20 August 2024") == date(2024, 8, 20)
    assert normalize.parse_date("5 मार्च 2021") == date(2021, 3, 5)
    assert normalize.parse_date("15 जानेवारी 2022") == date(2022, 1, 15)
    assert normalize.parse_date("9 ਨਵੰਬਰ 2021") == date(2021, 11, 9)
    assert normalize.parse_date("31/02/2023") is None


def test_areas():
    assert normalize.find_areas("0.632 हेक्टेयर")[0][1:] == (0.632, "hectare")
    assert normalize.find_areas("8 ਕਨਾਲ 4 ਮਰਲੇ")[0][2] == "kanal"


# ---------------------------------------------------------------- PII
def test_verhoeff_roundtrip():
    base = "23456789012"
    full = base + pii.verhoeff_check_digit(base)
    assert pii.verhoeff_valid(full)
    assert pii.is_valid_aadhaar(full)
    bad = full[:-1] + str((int(full[-1]) + 1) % 10)
    assert not pii.verhoeff_valid(bad)


def test_known_verhoeff_vector():
    assert pii.verhoeff_check_digit("236") == "3"
    assert pii.verhoeff_valid("2363")


def test_aadhaar_rules():
    assert not pii.is_valid_aadhaar("1234 5678 9012")  # cannot start with 0/1


def test_pan():
    assert pii.is_valid_pan("ABCPK1234F")
    assert not pii.is_valid_pan("ABC1234F")


def test_masking():
    base = "98765432101"
    num = base + pii.verhoeff_check_digit(base)
    spaced = f"{num[:4]} {num[4:8]} {num[8:]}"
    masked = pii.mask_text(f"आधार संख्या : {spaced}, खसरा 214")
    assert num[:8] not in masked.replace(" ", "")
    assert masked.endswith(f"XXXX-XXXX-{num[-4:]}, खसरा 214")
    assert pii.mask_text("phone 9876543210 12") == "phone 9876543210 12"
