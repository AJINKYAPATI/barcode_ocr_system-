"""
ui/sidebar.py
==============
Renders the Streamlit sidebar and returns an updated `AppSettings`
instance reflecting the user's choices. No engine objects live here —
this module only ever deals with plain config values.
"""

from __future__ import annotations

import streamlit as st

from config import AppSettings, get_default_settings


def render_sidebar() -> AppSettings:
    settings = st.session_state.get("settings") or get_default_settings()

    with st.sidebar:
        st.markdown("## ⚙️ Settings")

        settings.dark_theme = st.toggle("Dark theme", value=settings.dark_theme)

        st.markdown('<p class="section-title">OCR Engines</p>', unsafe_allow_html=True)
        settings.ocr.confidence_threshold = st.slider(
            "OCR confidence threshold (%)", 0, 100, int(settings.ocr.confidence_threshold)
        )
        settings.ocr.multi_engine_voting = st.checkbox(
            "Multi-engine voting (use every available OCR engine)", value=settings.ocr.multi_engine_voting
        )
        settings.ocr.use_gpu = st.checkbox("Use GPU (EasyOCR/PaddleOCR, if installed)", value=settings.ocr.use_gpu)

        st.markdown('<p class="section-title">Matching</p>', unsafe_allow_html=True)
        settings.matching.similarity_threshold = st.slider(
            "Similarity threshold for MATCH (%)", 50, 100, int(settings.matching.similarity_threshold)
        )
        settings.matching.apply_ocr_error_correction = st.checkbox(
            "Apply O/I/S/B/Z/G OCR-confusion correction", value=settings.matching.apply_ocr_error_correction
        )

        st.markdown('<p class="section-title">Preprocessing</p>', unsafe_allow_html=True)
        settings.preprocess.dynamic_pipeline = st.checkbox(
            "Dynamic pipeline (adapt steps to image quality)", value=settings.preprocess.dynamic_pipeline
        )
        settings.preprocess.rotation_correction_enabled = st.checkbox(
            "Rotation / deskew correction", value=settings.preprocess.rotation_correction_enabled
        )
        settings.preprocess.clahe_enabled = st.checkbox("CLAHE contrast enhancement", value=settings.preprocess.clahe_enabled)
        settings.preprocess.denoise_enabled = st.checkbox("Denoising", value=settings.preprocess.denoise_enabled)

        st.markdown('<p class="section-title">Performance</p>', unsafe_allow_html=True)
        settings.performance.enable_parallel_batch = st.checkbox(
            "Parallel batch processing", value=settings.performance.enable_parallel_batch
        )
        settings.performance.max_workers = st.slider(
            "Worker threads", 1, 16, int(settings.performance.max_workers)
        )

        st.markdown("---")
        st.caption(
            "OCR engine priority: PaddleOCR \u2192 EasyOCR \u2192 Tesseract "
            "(missing engines are skipped automatically).\n\n"
            "Barcode engine priority: OpenCV \u2192 ZXing \u2192 pyzbar."
        )

    st.session_state["settings"] = settings
    return settings
