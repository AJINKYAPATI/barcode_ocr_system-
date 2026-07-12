"""
core/matching.py
=================
Intelligent OCR-vs-barcode matching.

Replaces the original `ocr_text == barcode_value` comparison with:
  1. Normalization (via core.text_cleaning)
  2. Regex validation of the barcode's own format (EAN13/EAN8/UPC/Code128-ish)
  3. difflib.SequenceMatcher ratio
  4. A hand-rolled Levenshtein distance -> similarity score
     (no extra heavy dependency required)
  5. A weighted similarity score (configurable weights)
  6. A configurable acceptance threshold
  7. A human-readable mismatch reason when status != MATCH
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from functools import lru_cache

from config import MatchingConfig
from core.models import MatchResult, MatchStatus
from core.text_cleaning import clean_for_comparison, is_numeric_only
from logger import get_logger

log = get_logger(__name__)

# Loose format sanity-checks — these flag "this doesn't even look like a
# valid barcode value" cases distinctly from "right format, OCR mismatch".
_FORMAT_PATTERNS = {
    "EAN_13": re.compile(r"^\d{13}$"),
    "EAN_8": re.compile(r"^\d{8}$"),
    "UPC_A": re.compile(r"^\d{12}$"),
    "UPC_E": re.compile(r"^\d{6,8}$"),
    "CODE_128": re.compile(r"^[A-Z0-9\-\._/+% ]{4,}$"),
    "CODE_39": re.compile(r"^[A-Z0-9\-\.\$/+%\s]{4,}$"),
}


@lru_cache(maxsize=4096)
def levenshtein_distance(a: str, b: str) -> int:
    """Classic O(n*m) dynamic-programming edit distance. Cached — strings here are short SKUs."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev_row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr_row = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr_row[j] = min(
                prev_row[j] + 1,        # deletion
                curr_row[j - 1] + 1,    # insertion
                prev_row[j - 1] + cost,  # substitution
            )
        prev_row = curr_row
    return prev_row[-1]


def levenshtein_similarity(a: str, b: str) -> float:
    """Normalized similarity in [0, 100] derived from edit distance."""
    if not a and not b:
        return 100.0
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 100.0
    distance = levenshtein_distance(a, b)
    return max(0.0, (1 - distance / max_len)) * 100.0


def sequence_matcher_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio() * 100.0


def validate_barcode_format(value: str, symbology: str) -> bool:
    """Best-effort sanity check that *value* matches the expected symbology's shape."""
    pattern = _FORMAT_PATTERNS.get((symbology or "").upper())
    if pattern is None:
        return True  # unknown symbology -> don't penalise, just skip the check
    return bool(pattern.match(value))


def compute_similarity(ocr_text: str, barcode_value: str, config: MatchingConfig) -> float:
    """Weighted blend of Levenshtein + SequenceMatcher similarity (each optionally toggled off)."""
    scores, weights = [], []
    if config.use_levenshtein:
        scores.append(levenshtein_similarity(ocr_text, barcode_value))
        weights.append(config.levenshtein_weight)
    if config.use_sequence_matcher:
        scores.append(sequence_matcher_similarity(ocr_text, barcode_value))
        weights.append(config.sequence_matcher_weight)

    if not scores:
        # Both disabled — fall back to an exact-match boolean expressed as a score.
        return 100.0 if ocr_text == barcode_value else 0.0

    total_weight = sum(weights) or 1.0
    return sum(s * w for s, w in zip(scores, weights)) / total_weight


def match(
    ocr_text: str,
    barcode_value: str,
    config: MatchingConfig | None = None,
    barcode_symbology: str = "",
    ocr_confidence: float = 0.0,
    barcode_confidence: float = 0.0,
) -> MatchResult:
    """
    Compare OCR output against the decoded barcode value and return a
    structured `MatchResult` (status, similarity %, confidence %, reason).
    """
    config = config or MatchingConfig()

    if not ocr_text and not barcode_value:
        return MatchResult(status=MatchStatus.UNKNOWN, reason="Both OCR and barcode are empty")
    if not ocr_text:
        return MatchResult(status=MatchStatus.UNKNOWN, reason="OCR text missing — nothing to compare against the barcode")
    if not barcode_value:
        return MatchResult(status=MatchStatus.UNKNOWN, reason="Barcode value missing — nothing to compare against OCR text")

    numeric_expected = is_numeric_only(barcode_value)
    clean_ocr = clean_for_comparison(ocr_text, numeric_expected=numeric_expected)
    clean_barcode = clean_for_comparison(barcode_value, numeric_expected=numeric_expected) if config.apply_ocr_error_correction else barcode_value.upper()

    if config.use_regex_validation and barcode_symbology and not validate_barcode_format(barcode_value, barcode_symbology):
        log.warning(f"Barcode value '{barcode_value}' does not match expected {barcode_symbology} format")

    similarity = compute_similarity(clean_ocr, clean_barcode, config)
    # Blended "confidence" reflects both how similar the strings are AND how
    # much we trust the two upstream readings in the first place.
    upstream_confidence = (ocr_confidence + barcode_confidence) / 2.0 if (ocr_confidence or barcode_confidence) else similarity
    confidence = round((similarity * 0.7) + (upstream_confidence * 0.3), 1)

    if clean_ocr == clean_barcode:
        return MatchResult(status=MatchStatus.MATCH, similarity_percent=100.0, confidence_percent=max(confidence, 95.0), reason="Exact match after normalization")

    if similarity >= config.similarity_threshold:
        return MatchResult(
            status=MatchStatus.MATCH,
            similarity_percent=round(similarity, 1),
            confidence_percent=confidence,
            reason=f"Similarity {similarity:.1f}% >= threshold {config.similarity_threshold:.1f}%",
        )

    reason = _explain_mismatch(clean_ocr, clean_barcode, similarity, config.similarity_threshold)
    return MatchResult(
        status=MatchStatus.MISMATCH,
        similarity_percent=round(similarity, 1),
        confidence_percent=confidence,
        reason=reason,
    )


def _explain_mismatch(ocr: str, barcode: str, similarity: float, threshold: float) -> str:
    if len(ocr) != len(barcode):
        return (
            f"Length differs (OCR={len(ocr)} chars, barcode={len(barcode)} chars); "
            f"similarity {similarity:.1f}% below {threshold:.1f}% threshold"
        )
    distance = levenshtein_distance(ocr, barcode)
    if distance <= 2:
        return f"Likely OCR misread — only {distance} character(s) differ, similarity {similarity:.1f}%"
    return f"Substantially different strings (edit distance {distance}); similarity {similarity:.1f}% below {threshold:.1f}% threshold"
