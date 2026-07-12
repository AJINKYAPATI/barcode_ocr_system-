"""
app.py
======
Barcode + OCR Verification System — Streamlit UI.

Entry point:
    streamlit run app.py

This module is intentionally thin: all business logic lives in
`core/` and `services/`; all rendering helpers live in `ui/`. app.py
only wires session state, tabs, and user actions together.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List

import streamlit as st

from config import INPUT_IMAGES_DIR, get_default_settings
from core.models import ScanResult
from logger import get_logger
from services.pipeline import Pipeline
from services.report_service import ReportService
from ui.components import render_history_table, render_image_panels, render_result_card
from ui.dashboard import render_dashboard
from ui.sidebar import render_sidebar
from ui.theme import get_css
from utils import ensure_output_dirs, get_all_images, pil_to_cv2

log = get_logger("app")

st.set_page_config(page_title="Barcode + OCR Verification System", page_icon="📦", layout="wide")
ensure_output_dirs()


# ── Cached, expensive resources (built once per settings fingerprint) ───────

@st.cache_resource(show_spinner="Loading detection & OCR engines…")
def _get_pipeline(_settings_fingerprint: str) -> Pipeline:
    return Pipeline(st.session_state["settings"])


@st.cache_resource
def _get_report_service() -> ReportService:
    return ReportService()


def _settings_fingerprint() -> str:
    s = st.session_state["settings"]
    return f"{s.ocr.engine_priority}-{s.barcode.engine_priority}-{s.ocr.use_gpu}"


def _init_state() -> None:
    if "settings" not in st.session_state:
        st.session_state["settings"] = get_default_settings()
    if "history" not in st.session_state:
        st.session_state["history"] = []  # type: List[ScanResult]
    if "current_result" not in st.session_state:
        st.session_state["current_result"] = None


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    _init_state()
    settings = render_sidebar()
    st.markdown(get_css(settings.dark_theme), unsafe_allow_html=True)
    pipeline = _get_pipeline(_settings_fingerprint())
    report_service = _get_report_service()

    st.title("📦 Barcode + OCR Verification System")
    st.caption(
        f"Barcode engines: {', '.join(pipeline.barcode_detector.available_engines())}  |  "
        f"OCR engines: {', '.join(pipeline.ocr_service.available_engines())}"
    )

    tab_scan, tab_batch, tab_dashboard, tab_history, tab_logs = st.tabs(
        ["🔍 Single Scan", "📂 Batch Scan", "📊 Dashboard", "🗂️ History", "📝 Logs"]
    )

    # ── Single scan ──────────────────────────────────────────────────────
    with tab_scan:
        col_left, col_right = st.columns([1, 1])
        with col_left:
            uploaded = st.file_uploader("Upload a label image", type=["png", "jpg", "jpeg", "bmp", "tiff", "tif"])
        with col_right:
            sample_paths = get_all_images(str(INPUT_IMAGES_DIR))
            chosen_sample = st.selectbox(
                "...or pick a sample image", ["(none)"] + [Path(p).name for p in sample_paths]
            )

        image = None
        filename = ""
        if uploaded is not None:
            from PIL import Image as PILImage
            image = pil_to_cv2(PILImage.open(uploaded))
            filename = uploaded.name
        elif chosen_sample != "(none)":
            path = str(INPUT_IMAGES_DIR / chosen_sample)
            image = pipeline.processor.load_image(path)
            filename = chosen_sample

        if image is not None and st.button("Run scan", type="primary"):
            with st.spinner("Processing…"):
                result = pipeline.process_image(image, filename=filename)
            st.session_state["current_result"] = result
            st.session_state["history"].append(result)

        result: ScanResult | None = st.session_state.get("current_result")
        if result is not None and image is not None:
            barcode_annotated = pipeline.barcode_detector.get_annotated_image(image, result.barcode)
            roi_bbox = pipeline._sanitize_bbox(result.barcode.bbox, image.shape)  # noqa: SLF001 — internal helper reused for display only
            ocr_roi = pipeline.processor.crop_text_below_barcode(image, roi_bbox)
            render_image_panels(image, barcode_annotated, ocr_roi)
            st.markdown("---")
            render_result_card(result)
        elif image is not None:
            render_image_panels(image, None, None)
            st.info("Click **Run scan** to process this image.")

    # ── Batch scan ───────────────────────────────────────────────────────
    with tab_batch:
        st.caption(f"Scanning folder: `{INPUT_IMAGES_DIR}`")
        paths = get_all_images(str(INPUT_IMAGES_DIR))
        st.write(f"Found **{len(paths)}** images.")

        if paths and st.button("Run batch scan", type="primary"):
            progress = st.progress(0.0)
            status_text = st.empty()
            t0 = time.perf_counter()
            batch_results: List[ScanResult] = []

            def _cb(done: int, total: int, latest: ScanResult) -> None:
                elapsed = time.perf_counter() - t0
                eta = (elapsed / done) * (total - done) if done else 0.0
                progress.progress(done / total)
                status_text.text(
                    f"{done}/{total} processed — last: {latest.filename} "
                    f"[{latest.match.status.value}] — ETA {eta:.1f}s"
                )

            batch_results = pipeline.process_batch(paths, progress_callback=_cb)
            st.session_state["history"].extend(batch_results)

            matched = sum(1 for r in batch_results if r.match.status.value == "MATCH")
            mismatched = sum(1 for r in batch_results if r.match.status.value == "MISMATCH")
            st.success(
                f"Batch complete in {time.perf_counter() - t0:.1f}s — "
                f"🟢 {matched} MATCH / 🔴 {mismatched} MISMATCH / "
                f"⚠️ {len(batch_results) - matched - mismatched} UNKNOWN"
            )

    # ── Dashboard ────────────────────────────────────────────────────────
    with tab_dashboard:
        render_dashboard(st.session_state["history"])

    # ── History + exports ───────────────────────────────────────────────
    with tab_history:
        history: List[ScanResult] = st.session_state["history"]
        col_a, col_b = st.columns([3, 1])
        with col_a:
            search = st.text_input("Search history", "")
        with col_b:
            if st.button("Clear history"):
                st.session_state["history"] = []
                st.rerun()

        render_history_table(history, search)

        if history:
            st.markdown("---")
            st.markdown('<p class="section-title">Export</p>', unsafe_allow_html=True)
            d1, d2, d3, d4 = st.columns(4)
            with d1:
                st.download_button("⬇️ CSV", report_service.get_csv_bytes(history), "results.csv", "text/csv")
            with d2:
                st.download_button(
                    "⬇️ Excel", report_service.get_excel_bytes(history), "results.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            with d3:
                st.download_button("⬇️ JSON", report_service.get_json_bytes(history), "results.json", "application/json")
            with d4:
                pdf_bytes = report_service.get_pdf_bytes(
                    history,
                    summary={
                        "Total": len(history),
                        "Matched": sum(1 for r in history if r.match.status.value == "MATCH"),
                    },
                )
                if pdf_bytes:
                    st.download_button("⬇️ PDF", pdf_bytes, "report.pdf", "application/pdf")

            report_service.save_to_csv(history)
            report_service.save_to_excel(history)
            report_service.save_all_json(history)

            failed = [r for r in history if r.match.status.value in ("MISMATCH", "ERROR") or r.error]
            if failed:
                with st.expander(f"⚠️ Failed / Mismatch report ({len(failed)} items)"):
                    render_history_table(failed)

    # ── Logs ─────────────────────────────────────────────────────────────
    with tab_logs:
        from config import LOGS_DIR
        log_files = sorted(LOGS_DIR.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not log_files:
            st.info("No logs yet.")
        else:
            latest = log_files[0]
            st.caption(f"Showing: `{latest.name}` (most recent)")
            try:
                text = latest.read_text(encoding="utf-8", errors="ignore")
                st.code(text[-8000:], language="log")
            except Exception as exc:
                st.error(f"Could not read log file: {exc}")


if __name__ == "__main__":
    main()
