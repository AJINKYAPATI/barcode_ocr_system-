"""
ui/theme.py
===========
Dark/light CSS theme for the Streamlit app, plus small reusable badge
helpers used throughout the UI.
"""

from __future__ import annotations


def get_css(dark: bool) -> str:
    if dark:
        bg, panel, text, muted, border = "#0B1220", "#121B2E", "#E5E9F0", "#8B98AC", "#223049"
        accent = "#3B82F6"
    else:
        bg, panel, text, muted, border = "#F8FAFC", "#FFFFFF", "#0F172A", "#64748B", "#E2E8F0"
        accent = "#2563EB"

    return f"""
    <style>
    .stApp {{ background-color: {bg}; color: {text}; }}
    section[data-testid="stSidebar"] {{ background-color: {panel}; border-right: 1px solid {border}; }}
    .metric-card {{
        background: {panel}; border: 1px solid {border}; border-radius: 12px;
        padding: 14px 16px; margin-bottom: 8px;
    }}
    .badge-match {{
        background: #D1FAE5; color: #065F46; padding: 4px 12px; border-radius: 999px;
        font-weight: 700; font-size: 0.85rem;
    }}
    .badge-mismatch {{
        background: #FEE2E2; color: #991B1B; padding: 4px 12px; border-radius: 999px;
        font-weight: 700; font-size: 0.85rem;
    }}
    .badge-unknown {{
        background: #FEF3C7; color: #92400E; padding: 4px 12px; border-radius: 999px;
        font-weight: 700; font-size: 0.85rem;
    }}
    .section-title {{
        color: {muted}; text-transform: uppercase; letter-spacing: 0.08em;
        font-size: 0.75rem; font-weight: 700; margin-top: 6px;
    }}
    .quality-pill {{
        display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.78rem;
        font-weight: 600; background: {border}; color: {text};
    }}
    a {{ color: {accent}; }}
    </style>
    """


def badge_html(status: str) -> str:
    status = (status or "UNKNOWN").upper()
    if status == "MATCH":
        return '<span class="badge-match">&#9989; MATCH</span>'
    if status == "MISMATCH":
        return '<span class="badge-mismatch">&#10060; MISMATCH</span>'
    return '<span class="badge-unknown">&#9888; UNKNOWN</span>'
