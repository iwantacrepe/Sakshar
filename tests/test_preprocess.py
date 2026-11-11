import cv2
import numpy as np
import pytest

from sakshar.pipeline import preprocess


def synthetic_page(w=1600, h=2000):
    img = np.full((h, w, 3), 250, np.uint8)
    for i in range(22):
        y = 150 + i * 75
        cv2.putText(img, "Khasra number 214 area 0.632 hectare village Rampur" if i % 2 else
                    "Application No INC/2024/0045123 Applicant Meena Kumari",
                    (120, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 20, 20), 2, cv2.LINE_AA)
    return img


@pytest.mark.parametrize("angle", [3.5, -3.5, 6.0])
def test_deskew_recovers_angle(angle):
    rotated = preprocess.rotate(synthetic_page(), angle)
    gray = cv2.cvtColor(rotated, cv2.COLOR_RGB2GRAY)
    est = preprocess.estimate_skew(gray)
    assert abs(est - angle) < 0.8, (est, angle)
    fixed, skew = preprocess.deskew(rotated)
    residual = preprocess.estimate_skew(cv2.cvtColor(fixed, cv2.COLOR_RGB2GRAY))
    assert abs(residual) < 0.6


def test_straight_page_untouched():
    page = synthetic_page()
    out, skew = preprocess.deskew(page)
    assert skew == 0.0
    assert out.shape == page.shape


def test_forks_share_dimensions_and_stamp_keeps_colour():
    page = synthetic_page()
    cv2.circle(page, (1200, 600), 120, (120, 40, 160), 10)  # purple stamp (RGB)
    prepared = preprocess.prepare_page(preprocess.rotate(page, 3.5))
    h, w = prepared.rgb.shape[:2]
    assert prepared.clahe.shape[:2] == (h, w)
    assert prepared.binary.shape == (h, w)
    assert set(np.unique(prepared.binary)) <= {0, 255}
    r, g, b = (prepared.rgb[..., i].astype(int) for i in range(3))
    purple = ((r - g) > 40) & ((b - g) > 40)
    assert purple.sum() > 500, "deskewed RGB used by the viewer must keep stamp colour"


def test_load_png_with_unicode_path(tmp_path):
    p = tmp_path / "खतौनी नमूना.png"
    ok, buf = cv2.imencode(".png", synthetic_page(400, 300))
    buf.tofile(str(p))
    pages = preprocess.load_pages(p)
    assert len(pages) == 1 and pages[0].shape == (300, 400, 3)
