import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.preprocessing import ImageProcessor


def _solid_image(value: int = 200, h: int = 100, w: int = 150) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


def test_to_grayscale_shape():
    proc = ImageProcessor()
    gray = proc.to_grayscale(_solid_image())
    assert gray.ndim == 2


def test_resize_image_scale():
    proc = ImageProcessor()
    img = _solid_image(h=100, w=200)
    resized = proc.resize_image(img, scale=2.0)
    assert resized.shape[0] == 200 and resized.shape[1] == 400


def test_assess_quality_returns_report():
    proc = ImageProcessor()
    img = _solid_image()
    report = proc.assess_quality(img)
    assert 0.0 <= report.overall_score <= 100.0


def test_clahe_does_not_crash_on_flat_image():
    proc = ImageProcessor()
    gray = proc.to_grayscale(_solid_image())
    enhanced = proc.apply_clahe(gray)
    assert enhanced.shape == gray.shape


def test_otsu_threshold_binary_output():
    proc = ImageProcessor()
    gray = proc.to_grayscale(_solid_image())
    binary = proc.otsu_threshold(gray)
    assert set(np.unique(binary)).issubset({0, 255})


def test_crop_text_below_barcode_fallback_no_bbox():
    proc = ImageProcessor()
    img = _solid_image(h=200, w=300)
    region = proc.crop_text_below_barcode(img, barcode_bbox=None)
    assert region.shape[0] > 0


def test_load_image_missing_file_returns_none():
    proc = ImageProcessor()
    assert proc.load_image("/tmp/does_not_exist_12345.png") is None
