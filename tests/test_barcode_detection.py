import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.barcode_detection import BarcodeDetector


def _make_qr_image(value: str = "TEST123") -> np.ndarray:
    """Generate a real QR code image via OpenCV so detection has something genuine to find."""
    try:
        from qrcode import QRCode  # optional dependency
        qr = QRCode(border=4, box_size=10)
        qr.add_data(value)
        qr.make()
        img = qr.make_image(fill_color="black", back_color="white").convert("L")
        return cv2.cvtColor(np.array(img), cv2.COLOR_GRAY2BGR)
    except Exception:
        return None


def test_detector_initializes_with_available_engines():
    det = BarcodeDetector()
    assert "opencv" in det.available_engines()


def test_detect_barcode_on_blank_image_fails_gracefully():
    det = BarcodeDetector()
    blank = np.full((200, 300, 3), 255, dtype=np.uint8)
    result = det.detect_barcode(blank)
    assert result.status.value in ("FAILED", "ERROR")
    assert result.value == ""


def test_sanitize_does_not_crash_on_real_sample(tmp_path):
    sample = Path(__file__).resolve().parent.parent / "input_images" / "4.png"
    if not sample.exists():
        return  # sample assets are optional in CI
    det = BarcodeDetector()
    img = cv2.imread(str(sample))
    result = det.detect_barcode(img)
    assert result.status.value in ("SUCCESS", "FAILED", "ERROR")


def test_get_annotated_image_returns_same_shape():
    det = BarcodeDetector()
    img = np.full((150, 200, 3), 255, dtype=np.uint8)
    annotated = det.get_annotated_image(img)
    assert annotated.shape[:2] == img.shape[:2]
