"""Download Tesseract traineddata (tessdata_best) for hin / mar / pan / eng / osd.

    python scripts/setup_tessdata.py

Installs into the tessdata directory of the Tesseract that Sakshar uses (conda env first).
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sakshar.pipeline.tess import tessdata_dir, tesseract_cmd  # noqa: E402

LANGS = {
    "hin": "tessdata_best", "mar": "tessdata_best", "pan": "tessdata_best",
    "eng": "tessdata_best", "osd": "tessdata",  # osd only exists in the legacy repo
}
URL = "https://github.com/tesseract-ocr/{repo}/raw/main/{lang}.traineddata"


def main() -> None:
    cmd = tesseract_cmd()
    td = tessdata_dir()
    if not cmd or not td:
        raise SystemExit("Tesseract not found. Install it into the env: conda install -c conda-forge tesseract")
    dest = Path(td)
    print(f"tesseract: {cmd}\ntessdata:  {dest}")
    for lang, repo in LANGS.items():
        target = dest / f"{lang}.traineddata"
        if target.exists() and target.stat().st_size > 100_000:
            print(f"  {lang}: present")
            continue
        url = URL.format(repo=repo, lang=lang)
        print(f"  {lang}: downloading {url}")
        tmp = target.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(target)
    print("done")


if __name__ == "__main__":
    main()
