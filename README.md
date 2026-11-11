# Sakshar · साक्षर · ਸਾਖਰ

Sakshar is an OCR and voice assistant for Indian government paperwork. Upload a photo, scan or PDF of a land record (UP Khatauni, Maharashtra 7/12 Satbara, Punjab Jamabandi), an application form or a certificate. Sakshar then:

1. **Reads it.** OpenCV preprocessing, then PaddleOCR with a Tesseract fallback, with script and language detected automatically.
2. **Extracts key details.** Layout analysis with XY-cut finds names, dates, amounts and application, khasra or survey numbers, then checks each value against a schema for that document type.
3. **Flags uncertain values.** Low OCR confidence, a failed validator (including the Aadhaar Verhoeff checksum), a value missing from its source text, or two extractors disagreeing all send a value to amber review. Review shows a crop of the source region.
4. **Explains it in plain language,** in **Hindi, Marathi, Punjabi or English**. Every point cites the text it came from (click a point to highlight it on the page). Claims that can't be grounded are marked as unverified.
5. **Answers follow-up questions by voice or text.** Speech goes through Whisper large-v3-turbo, IndicTrans2, grounded Q&A over hybrid BM25 and fuzzy retrieval, then Indic Parler-TTS for the spoken answer.

```
upload ─► preprocess ─► script detect ─► OCR ─► layout ─► classify ─► extract ─► verify ─► explain
          (deskew RGB,   (OSD + Unicode  (Paddle  (XY-cut   (keyword   (regex +   (gates,   (OpenAI, cited
           CLAHE fork,    blocks, hi/mr)  / Tess)  + lines)  + LLM)     LLM)       masking)  + grounded)
           binary fork)
                                              voice:  webm ─► faster-whisper ─► IndicTrans2 ─► Q&A ─► Parler-TTS
```

## Setup (Windows, conda)

> Install conda to a path **without spaces** (e.g. `C:\miniconda3`). Conda's post-link scripts break under `C:\Users\FIRST LAST\...`.

```powershell
# 1. environment (Python 3.11, Tesseract from conda-forge, CUDA 12.4 torch)
conda env create -f environment.yml
conda activate sakshar

# 2. Tesseract language data: conda-forge's tesseract already ships hin/mar/pan/eng/osd.
#    Only needed for a non-conda Tesseract install:
python scripts/setup_tessdata.py

# 3. configuration
copy .env.example .env      # then set OPENAI_API_KEY (and HF_TOKEN for voice)
```

**Voice models.** Two of the models are gated on Hugging Face. Accept the terms for `ai4bharat/indictrans2-en-indic-dist-200M`, `ai4bharat/indictrans2-indic-en-dist-200M` and `ai4bharat/indic-parler-tts`, then put your token in `HF_TOKEN`. All voice models download on first use (about 3 GB). To run without them, set `ENABLE_VOICE=false`.

**Channels.** The env uses conda-forge only, so no Anaconda ToS acceptance is needed.

## Run

```powershell
conda activate sakshar
python scripts/make_samples.py          # synthetic hi / mr / pa / en documents -> samples/
uvicorn sakshar.main:app --reload       # http://localhost:8000
```

`GET /healthz` reports the Tesseract path, the installed language packs, whether the LLM is configured, and which voice models are loaded on which device.

## Tests

```powershell
pytest
```

The tests cover:

- deskew angle recovery, and stamp colour surviving the viewer image
- the preprocessing forks sharing the same dimensions
- script detection, including hi vs mr
- Devanagari and Gurmukhi digits, amounts and multilingual dates
- Verhoeff, PAN and masking
- the XY-cut column separation
- each verification gate
- grounding: drops a citation to a missing block, flags a wrong value, repairs a citation
- hybrid retrieval finding `खसरा` when the OCR read `खसस`
- a full API flow (upload → SSE → fields → review → explanation → Q&A) with mocked OCR and a mocked OpenAI client, which also asserts that no unmasked Aadhaar number ever reaches the LLM

## Configuration (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | – | Enables classification fallback, LLM extraction, explanations and Q&A |
| `OPENAI_MODEL` | `gpt-4o-mini` | Any model that supports structured outputs |
| `OPENAI_BASE_URL` | – | Any OpenAI-compatible endpoint (vLLM, Azure, Groq, Ollama `/v1`) |
| `TRANSLATION_ENGINE` | `llm` | `llm`: the model writes the local-language text directly, keeping revenue terms. `indictrans2`: translate offline |
| `ENABLE_VOICE` | `true` | Whisper, IndicTrans2 and Parler-TTS |
| `ENABLE_PADDLE` | `true` | `false` forces Tesseract only |
| `DEVICE` | `cuda` | `cpu` to keep the GPU free |
| `REVIEW_THRESHOLD` / `REVIEW_THRESHOLD_STRICT` | 0.75 / 0.85 | Minimum OCR word confidence for ordinary fields / for legal, financial and identity fields |
| `MASK_PII` | `true` | Store Aadhaar numbers masked (`XXXX-XXXX-1234`) |

## Design notes

- **Two preprocessing forks.** The original RGB is deskewed first and served to the viewer, so stamps keep their colour. Both OCR inputs come from that image and share its dimensions: CLAHE grey for PaddleOCR, and an adaptive-threshold binary for Tesseract only, since hard binarization damages Devanagari matras for neural OCR. Boxes therefore map 1:1 onto the displayed page.
- **Gurmukhi.** PaddleOCR has no Gurmukhi recognizer. Punjabi gets one full-page Tesseract pass, then PaddleOCR detects text boxes and any lines the page pass missed are re-read with `--psm 7`. Those re-reads run 8-wide in a thread pool, because Windows process start-up is slow.
- **Concurrency.** Pipeline stages run in `asyncio.to_thread`, so SSE progress keeps streaming. PaddleOCR calls are serialized with a lock. SQLite runs in WAL mode with `check_same_thread=False`.
- **GPU (6 GB).** Whisper large-v3-turbo in int8_float16, IndicTrans2 in fp16 for both directions, and Parler-TTS in fp16 stay resident together. In testing, three of these models used about 3.6 GB. Models are offloaded only on a CUDA out-of-memory error.
- **OCR engine per script, chosen by measurement.** PaddleOCR's Devanagari recognizer dropped spaces and conjuncts while still reporting ~0.95 confidence. Hindi, Marathi and Punjabi therefore use Tesseract for recognition and PaddleOCR only to detect text lines. The Hindi and Marathi Tesseract models also misread Latin digits, so an English-only pass repairs numbers. English uses PaddleOCR fully, with the mobile models (8 s per page instead of 50 s).
- **Paddle runs on a single dedicated thread.** Its CPU predictor fails ("Unknown exception") when used from a thread other than the one that created it. Torch is imported before Paddle because the reverse order breaks torch on Windows (WinError 127, `shm.dll`).
- **Voice details.**
  - Revenue terms such as खसरा (which is also Hindi for "measles"), खाता and ਖੇਵਟ are swapped for their romanised forms before IndicTrans2 sees them.
  - Numbers are spelled out digit by digit before TTS, because Parler mispronounces numerals.
  - Whisper "small" produced unusable Punjabi, so the default is `large-v3-turbo`.
  - The language selected in the UI is passed to Whisper, because auto-detection mislabels Marathi and Punjabi as Hindi.
- **Grounding.** Statements carry `text_en` and `text_local` plus the block ids they cite. Any hard value (a number with 3+ digits, a date or an amount) must appear in the cited text. If it appears elsewhere in the document, the citation is repaired; otherwise the statement is marked unverified.
- **Privacy.** Aadhaar numbers are validated locally with the Verhoeff checksum and masked before any text is sent to the LLM or stored.
