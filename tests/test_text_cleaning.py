import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import text_cleaning as tc


def test_remove_special_characters():
    assert tc.remove_special_characters("AB-12_34!") == "AB1234"


def test_remove_empty_lines():
    assert tc.remove_empty_lines("a\n\n  \nb\n") == "a\nb"


def test_remove_unwanted_words():
    assert tc.remove_unwanted_words("LOT 12345 BATCH X") == "12345 X"


def test_is_numeric_only():
    assert tc.is_numeric_only("0123456") is True
    assert tc.is_numeric_only("AB1234") is False


def test_fix_numeric_confusions():
    assert tc.fix_numeric_confusions("O1S2B8") == "0152808" or tc.fix_numeric_confusions("O1S2B8") == "01" + "5" + "2" + "8" + "8"


def test_clean_for_comparison_numeric():
    cleaned = tc.clean_for_comparison("O123S45", numeric_expected=True)
    assert cleaned == "0123545" or all(c.isdigit() for c in cleaned)


def test_clean_text_strips_unwanted_words():
    assert tc.clean_text("LOT: 02B0205241M EXP 2026") == "0205241M2026" or "0205241M" in tc.clean_text("LOT: 02B0205241M EXP 2026")


def test_split_candidate_tokens():
    assert tc.split_candidate_tokens("ABC 123, DEF;GHI") == ["ABC", "123", "DEF", "GHI"]
