"""
core/text_cleaning.py
======================
Dedicated text normalization for OCR output.

Responsibilities:
  - Strip whitespace, special characters, unwanted boilerplate words
  - Remove empty lines
  - Normalize capitalization
  - Correct common OCR character-confusion mistakes (O/0, I/1, S/5, B/8, Z/2, G/6)
  - Detect numeric-only labels (e.g. pure barcode digit strings)
"""

from __future__ import annotations

import re
from typing import List

# Words that occasionally get OCR'd off packaging/labels and add no value
# to a barcode/SKU comparison (extend as needed for your label format).
_UNWANTED_WORDS = {
    "LOT", "BATCH", "EXP", "MFG", "REF", "QTY", "PCS", "NET", "WT",
    "MADE", "IN", "USA", "CHINA", "PACK", "OF",
}

# Character-confusion corrections applied ONLY when normalizing a token that
# is expected to be purely numeric (e.g. comparing against a digits-only
# barcode value) — applying these blindly to alphanumeric SKUs would corrupt
# legitimate letters, so callers choose when to invoke `fix_numeric_confusions`.
_NUMERIC_OCR_FIXES = {
    "O": "0", "Q": "0",
    "I": "1", "L": "1",
    "S": "5",
    "B": "8",
    "Z": "2",
    "G": "6",
}

_ALNUM_PATTERN = re.compile(r"[^A-Z0-9]")
_NUMERIC_PATTERN = re.compile(r"^[0-9]+$")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def remove_special_characters(text: str) -> str:
    """Keep only alphanumeric characters (after uppercasing)."""
    return _ALNUM_PATTERN.sub("", text.upper())


def remove_empty_lines(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines)


def remove_unwanted_words(text: str, extra_words: set | None = None) -> str:
    """Strip known boilerplate tokens (LOT, BATCH, EXP, ...) from a token stream."""
    unwanted = _UNWANTED_WORDS | (extra_words or set())
    tokens = text.split()
    kept = [t for t in tokens if t.upper().strip(":.-") not in unwanted]
    return " ".join(kept)


def normalize_capitalization(text: str) -> str:
    return text.upper().strip()


def normalize_whitespace(text: str) -> str:
    return _WHITESPACE_PATTERN.sub(" ", text).strip()


def is_numeric_only(text: str) -> bool:
    """True if *text* (after stripping non-alnum) is purely digits — typical for EAN/UPC labels."""
    return bool(_NUMERIC_PATTERN.match(remove_special_characters(text)))


def fix_numeric_confusions(text: str) -> str:
    """
    Apply O->0, I->1, S->5, B->8, Z->2, G->6 corrections.

    Intended for tokens that are *expected* to be numeric (e.g. being
    compared directly against a decoded EAN/UPC barcode value). Do not
    apply to genuinely alphanumeric SKUs.
    """
    return "".join(_NUMERIC_OCR_FIXES.get(ch, ch) for ch in text.upper())


def clean_text(text: str, remove_unwanted: bool = True, extra_unwanted_words: set | None = None) -> str:
    """
    Full normalization pipeline applied to raw OCR output:
    remove empty lines -> remove unwanted words -> remove special chars -> normalize case/whitespace.
    """
    if not text:
        return ""
    working = remove_empty_lines(text)
    working = normalize_whitespace(working)
    if remove_unwanted:
        working = remove_unwanted_words(working, extra_unwanted_words)
    working = remove_special_characters(working)
    return normalize_capitalization(working)


def clean_for_comparison(text: str, numeric_expected: bool = False) -> str:
    """
    Clean a token specifically for matching against a barcode value.

    If `numeric_expected` is True (the barcode value itself is digits-only),
    also runs the O/I/S/B/Z/G OCR-confusion correction, since most
    real-world mismatches at this stage are confusable-glyph errors.
    """
    cleaned = clean_text(text)
    if numeric_expected and not is_numeric_only(cleaned):
        cleaned = fix_numeric_confusions(cleaned)
    return cleaned


def split_candidate_tokens(text: str) -> List[str]:
    """Split raw multi-line/multi-word OCR output into individual candidate tokens for matching."""
    raw_tokens = re.split(r"[\s,;]+", text.strip())
    return [t for t in raw_tokens if t]
