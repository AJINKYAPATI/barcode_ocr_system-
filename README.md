# 📦 Barcode + OCR Verification System

A web app that checks whether the **barcode printed on a product label** matches the **text printed next to it**. You upload a photo of a label, and the system reads the barcode, reads the text using OCR (Optical Character Recognition), compares the two, and tells you if they match — with confidence scores, reports, and a full history of every scan.

Built with **Python** and **Streamlit**, it's meant to work like a real quality-control tool you'd find in a warehouse or packaging line, not just a demo script.

---
<img width="716" height="422" alt="image" src="https://github.com/user-attachments/assets/a2940fc6-fe45-4304-ab52-c78c192684a1" />







## 1. What problem does this solve?

On many product labels, the same code appears twice: once as a **barcode** and once as **printed text** (for example, `8901234567890` printed both as a scannable barcode and as plain numbers underneath it). In real-world manufacturing and logistics, printing errors happen — the barcode and the text next to it can end up not matching, because of a printer glitch, a copy-paste error, or a damaged label.

Checking this by eye, one label at a time, is slow and easy to get wrong. This system automates that check:

1. It finds and decodes the barcode.
2. It reads the nearby printed text using OCR.
3. It compares the two values intelligently (not just a strict character-by-character match).
4. It flags every label as **MATCH**, **MISMATCH**, or **UNKNOWN**.
5. It can do this for one image at a time, or for an entire folder of hundreds of images in one go.

---

## 2. Key features

- **Single image scan** — upload one label photo and get an instant result.
- **Batch scanning** — point it at a folder and it processes every image inside, with a live progress bar and time estimate.
- **Multiple barcode-reading engines** — OpenCV, ZXing, and pyzbar. If one engine fails to read a barcode, the system automatically tries the others and keeps the best result.
- **Multiple OCR engines** — PaddleOCR, EasyOCR, and Tesseract. Each engine "votes" with its own reading, and the system picks the most confident, most agreed-upon answer.
- **Smart image cleanup before reading** — blurry, tilted, poorly lit, or noisy photos are automatically detected and cleaned up (sharpening, straightening, noise removal, contrast fixing) before the barcode/OCR engines even see them.
- **Intelligent text matching** — instead of requiring an exact match, it uses fuzzy text comparison (similarity scoring, edit-distance, and known OCR misreadings like the letter "O" vs. the number "0") so that a genuine misread doesn't wrongly count as a real mismatch.
- **Dashboard** — charts and summary numbers (total scans, match rate, engine performance) over all the scans done so far.
- **History table** — every past scan, searchable, with the option to clear it.
- **Exports** — download all results as CSV, Excel, JSON, or a formatted PDF report.
- **Live logs tab** — see what the system is doing under the hood, useful for troubleshooting a specific image.
- **Dark / light theme toggle** and a clean, tab-based interface.
- **Configurable everything** — confidence thresholds, which engines to use, GPU on/off, and every preprocessing step can be turned on or off from the sidebar, without touching code.

---

## 3. How it works (in plain language)

Think of it as an assembly line with four stations:

**Station 1 — Clean up the image.**
The uploaded photo is checked for quality (Is it blurry? Too dark? Noisy? Tilted?). Based on what it finds, the system applies only the fixes that image actually needs — for example, straightening a tilted label, or sharpening a blurry one — rather than blindly running every possible filter on every image.

**Station 2 — Find and read the barcode.**
The cleaned-up image is passed to up to three different barcode-reading engines. Each one tries to locate the barcode and decode its value. Their results are scored, and the most confident, correct-looking result wins.

**Station 3 — Read the printed text.**
The system crops out the area just below/around the barcode (where the human-readable text usually is) and runs it through up to three OCR engines. Their readings are "voted" on, so a single engine's mistake doesn't automatically become the final answer.

**Station 4 — Compare and decide.**
The barcode's value and the OCR's text are cleaned up (removing stray spaces, fixing obvious OCR letter/number confusions) and compared using a similarity score. If they're close enough (based on a threshold you can adjust), it's marked **MATCH**. If not, it's marked **MISMATCH**, along with a plain-English reason. If either the barcode or the text couldn't be read at all, it's marked **UNKNOWN**.

Every result — the barcode value, the OCR text, the similarity score, which engines were used, how long it took, and a timestamp — is saved and shown in the History and Dashboard tabs.

---

## 4. Tech stack

| Layer | Technology |
|---|---|
| App / UI | Streamlit (Python web framework), Plotly (charts) |
| Image processing | OpenCV, Pillow (PIL), NumPy |
| Barcode decoding | OpenCV `BarcodeDetector`, ZXing-C++, pyzbar |
| OCR (text reading) | PaddleOCR, EasyOCR, Tesseract (`pytesseract`) |
| Data & reports | Pandas, OpenPyXL (Excel), ReportLab (PDF), JSON |
| Testing | Pytest |
| Packaging | PyInstaller (optional, for building a standalone executable) |

The OCR and barcode engines are all **optional and auto-detected**: the app checks which ones are installed and simply skips any that aren't, so it still works even with a minimal install (Tesseract + OpenCV alone are enough to run it).

---

## 5. Project structure

```
barcode_ocr_system/
├── app.py                     # Main Streamlit app — wires everything together
├── config.py                  # All settings in one place (thresholds, engine priority, paths)
├── logger.py                  # Logging setup
├── utils.py                   # Small shared helper functions
│
├── core/                      # The "thinking" logic of the system
│   ├── models.py              # Shared data structures (ScanResult, BarcodeResult, etc.)
│   ├── preprocessing.py       # Image cleanup: sharpening, deskewing, noise removal, quality scoring
│   ├── barcode_detection.py   # Multi-engine barcode reading + best-result selection
│   ├── matching.py            # Fuzzy comparison between barcode value and OCR text
│   └── text_cleaning.py       # Normalizes OCR text (removes noise, fixes common misreads)
│
├── services/                  # Higher-level orchestration
│   ├── pipeline.py            # Runs the full scan pipeline for one image or a whole batch
│   ├── ocr_service.py         # Multi-engine OCR with confidence voting
│   └── report_service.py      # Generates CSV / Excel / JSON / PDF reports
│
├── ui/                        # Everything the user sees
│   ├── sidebar.py             # Settings panel
│   ├── dashboard.py           # Charts & summary stats
│   ├── components.py          # Reusable UI pieces (result cards, tables, image panels)
│   └── theme.py               # Dark/light theme styling
│
├── tests/                     # Automated tests (Pytest)
├── input_images/              # Sample/batch images live here
├── models/                    # Local model weight files (if any engine needs them)
└── outputs/                   # Everything the app generates
    ├── results.csv / results.xlsx
    ├── json/all_results.json
    ├── reports/*.pdf
    ├── cropped_regions/        # Saved crops of the text region for debugging
    └── logs/                   # Timestamped run logs
```

---

## 6. Installation

**Requirements:** Python 3.9+

1. Clone or download the project, then move into its folder:
   ```bash
   cd barcode_ocr_system
   ```

2. (Recommended) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

3. Install the core dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Some engines need a system-level program installed too, not just a Python package:
   - **Tesseract OCR** (required for the lightweight OCR path):
     ```bash
     sudo apt-get install tesseract-ocr        # Linux
     brew install tesseract                    # macOS
     ```
   - **pyzbar** needs the ZBar system library:
     ```bash
     sudo apt-get install libzbar0
     ```
   - **PaddleOCR** and **EasyOCR** are optional, heavier engines (they pull in PyTorch or PaddlePaddle). They're commented as optional in `requirements.txt` — install them only if you want extra OCR accuracy:
     ```bash
     pip install paddleocr paddlepaddle
     pip install easyocr torch torchvision
     ```

   You don't need every engine installed — the app automatically detects and uses whichever ones are available.

---

## 7. Running the app

```bash
streamlit run app.py
```

This opens the app in your browser (usually at `http://localhost:8501`). From there:

- **🔍 Single Scan** — upload one image (or pick a sample from `input_images/`) and click **Run scan**.
- **📂 Batch Scan** — processes every image in the `input_images/` folder at once.
- **📊 Dashboard** — view match/mismatch statistics and charts across all scans done so far.
- **🗂️ History** — search past results and export them as CSV, Excel, JSON, or PDF.
- **📝 Logs** — view the most recent run's detailed log file, useful for debugging a tricky image.

All settings (OCR confidence threshold, similarity threshold for a match, which engines to prefer, GPU usage, dark/light theme) are adjustable from the sidebar without editing any code.

---

## 8. Configuration

Every tunable value lives in `config.py`, so nothing is hardcoded elsewhere in the codebase. A few can also be overridden with environment variables, for example:

```bash
export BARCODE_OCR_CONF_THRESHOLD=70
```

Key settings include:
- Which barcode/OCR engines to try, and in what order.
- The OCR confidence threshold and the similarity threshold that decides MATCH vs. MISMATCH.
- Which image-cleanup steps are enabled (deskewing, denoising, sharpening, contrast fixing, etc.).
- Number of parallel workers used for batch scans.

---

## 9. Testing

The project includes a Pytest test suite covering the core logic:

```bash
pytest
```

Tests cover: barcode detection, image preprocessing, OCR text cleaning, the fuzzy matching logic, and report generation — so changes to any core module can be verified quickly.

---

## 10. Output files

Every scan is saved automatically under `outputs/`:

- `results.csv` / `results.xlsx` — a spreadsheet of every scan result.
- `json/all_results.json` — the same data in JSON form.
- `reports/*.pdf` — a formatted, shareable PDF summary report.
- `cropped_regions/` — the cropped text-region images, useful for double-checking what the OCR engine actually saw.
- `logs/` — one timestamped log file per run, showing exactly what happened step by step.

---

## 11. Notes on design choices

- **Nothing crashes if an engine is missing.** Every barcode/OCR engine is wrapped so that a missing dependency (e.g., PaddleOCR not installed) is simply logged and skipped — the app keeps working with whatever is available.
- **Adaptive preprocessing.** Rather than always running every image-cleanup filter (slow), the system scores each image's quality first and only applies the fixes that image actually needs.
- **Fuzzy, not strict, matching.** Real-world OCR is never 100% perfect, so exact string equality would create false mismatches. The matching logic accounts for common OCR mix-ups (like `O` vs `0`, or `S` vs `5`) before deciding.
- **Threaded batch processing.** Batch scans use a thread pool (rather than separate processes) since most of the heavy lifting happens in OpenCV/OCR libraries that release Python's GIL, giving good speed without extra complexity.

---

## 12. Possible future improvements

- Custom-trained barcode/label detection model (YOLO) for messier, real-world photos where the barcode isn't already front-and-center.
- Support for more label layouts (multi-line text, barcodes on curved surfaces).
- User accounts and persistent, multi-user history storage.
- REST API mode, so other systems can call the pipeline directly without the Streamlit UI.

---

## 13. Author

Built by **Ajinkya**, as part of an ongoing set of data/ML portfolio projects focused on practical, production-style tooling (see also: SegmentIQ customer segmentation, Customer Churn Analysis dashboard, and a Voice-Based Mental Health Monitoring System).
