"""
ui/dashboard.py
=================
Analytics dashboard: summary metrics + charts over scan history.
"""

from __future__ import annotations

from typing import List

import pandas as pd
import streamlit as st

from core.models import ScanResult


def render_dashboard(results: List[ScanResult]) -> None:
    if not results:
        st.info("No data yet — run a scan to populate the dashboard.")
        return

    rows = [r.to_flat_dict() for r in results]
    df = pd.DataFrame(rows)

    total = len(df)
    matched = int((df["match_status"] == "MATCH").sum())
    mismatched = int((df["match_status"] == "MISMATCH").sum())
    unknown = int((df["match_status"] == "UNKNOWN").sum())
    barcode_success = int((df["barcode_value"] != "").sum())
    ocr_success = int((df["ocr_text"] != "").sum())
    avg_conf = df["ocr_confidence"].astype(float).mean() if total else 0.0
    avg_time = df["processing_time"].astype(float).mean() if total else 0.0
    accuracy = (matched / total * 100.0) if total else 0.0

    m = st.columns(4)
    m[0].metric("Total images", total)
    m[1].metric("Matched", matched, f"{accuracy:.1f}% accuracy")
    m[2].metric("Mismatched", mismatched)
    m[3].metric("Unknown", unknown)

    m2 = st.columns(4)
    m2[0].metric("Barcode success rate", f"{barcode_success / total * 100:.1f}%" if total else "0%")
    m2[1].metric("OCR success rate", f"{ocr_success / total * 100:.1f}%" if total else "0%")
    m2[2].metric("Avg. OCR confidence", f"{avg_conf:.1f}%")
    m2[3].metric("Avg. processing time", f"{avg_time:.3f}s")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Match status breakdown")
        status_counts = df["match_status"].value_counts()
        st.bar_chart(status_counts)
    with c2:
        st.caption("Processing time per image")
        st.line_chart(df.set_index("filename")["processing_time"].astype(float))

    st.caption("OCR confidence distribution")
    st.bar_chart(df.set_index("filename")["ocr_confidence"].astype(float))
