"""
ui/components.py
==================
Reusable Streamlit rendering helpers: image panels, result cards,
history tables. Pure presentation — takes `ScanResult`/list-of-dict
data in, renders Streamlit widgets, returns nothing.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from core.models import ScanResult
from ui.theme import badge_html
from utils import cv2_to_pil


def render_image_panels(
    original: Optional[np.ndarray],
    barcode_annotated: Optional[np.ndarray],
    ocr_roi: Optional[np.ndarray],
) -> None:
    cols = st.columns(3)
    labels_images = [("Original", original), ("Barcode detection", barcode_annotated), ("OCR region", ocr_roi)]
    for col, (label, img) in zip(cols, labels_images):
        with col:
            st.caption(label)
            if img is not None:
                st.image(cv2_to_pil(img) if isinstance(img, np.ndarray) else img, use_container_width=True)
            else:
                st.info("No image")


def render_result_card(result: ScanResult) -> None:
    flat = result.to_flat_dict()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Barcode value", flat["barcode_value"] or "—", flat["barcode_type"])
    with c2:
        st.metric("OCR text", flat["ocr_text"] or "—", f"{flat['ocr_confidence']:.1f}% ({flat['ocr_engine'] or 'n/a'})")
    with c3:
        st.metric("Similarity", f"{flat['similarity_percent']:.1f}%", f"{flat['processing_time']:.3f}s")

    st.markdown(badge_html(flat["match_status"]), unsafe_allow_html=True)
    if flat["mismatch_reason"]:
        st.caption(flat["mismatch_reason"])
    if flat["error"]:
        st.error(flat["error"])


def render_history_table(results: List[ScanResult], search: str = "") -> None:
    if not results:
        st.info("No scan history yet — run a scan to populate this table.")
        return

    rows = [r.to_flat_dict() for r in results]
    df = pd.DataFrame(rows)
    if search:
        mask = df.apply(lambda row: row.astype(str).str.contains(search, case=False).any(), axis=1)
        df = df[mask]
        if df.empty:
            st.warning(f"No results matching '{search}'")
            return

    display_cols = [
        "image_number", "filename", "barcode_value", "barcode_type",
        "ocr_text", "ocr_confidence", "similarity_percent", "match_status",
        "processing_time", "scan_time",
    ]
    st.dataframe(df[display_cols], use_container_width=True, hide_index=True)
