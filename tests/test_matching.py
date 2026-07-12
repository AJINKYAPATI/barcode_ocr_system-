import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import MatchingConfig
from core import matching as m
from core.models import MatchStatus


def test_levenshtein_distance_basic():
    assert m.levenshtein_distance("kitten", "sitting") == 3
    assert m.levenshtein_distance("abc", "abc") == 0
    assert m.levenshtein_distance("", "abc") == 3


def test_levenshtein_similarity_identical():
    assert m.levenshtein_similarity("ABC123", "ABC123") == 100.0


def test_sequence_matcher_similarity_range():
    score = m.sequence_matcher_similarity("ABC123", "ABC124")
    assert 0.0 < score < 100.0


def test_validate_barcode_format_ean13():
    assert m.validate_barcode_format("1234567890123", "EAN_13") is True
    assert m.validate_barcode_format("123", "EAN_13") is False


def test_match_exact():
    result = m.match("02B0411241M", "02B0411241M")
    assert result.status == MatchStatus.MATCH
    assert result.similarity_percent == 100.0


def test_match_with_ocr_confusion_numeric():
    result = m.match("O123456", "0123456")
    assert result.status == MatchStatus.MATCH


def test_match_mismatch_reason_present():
    result = m.match("AAAAAAA", "0123456")
    assert result.status == MatchStatus.MISMATCH
    assert result.reason


def test_match_missing_values_returns_unknown():
    result = m.match("", "0123456")
    assert result.status == MatchStatus.UNKNOWN


def test_match_threshold_configurable():
    cfg = MatchingConfig(similarity_threshold=99.0)
    result = m.match("02B0411241X", "02B0411241M", config=cfg)
    assert result.status == MatchStatus.MISMATCH  # only 1 char differs but threshold is very strict
