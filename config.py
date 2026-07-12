"""
config.py
=========
Single source of truth for every configurable value in the system.

Nothing in `core/`, `services/`, or `ui/` should contain a hardcoded
threshold, path, weight, or engine name — it should be read from here
(or from a `Settings` instance built from here + the Streamlit sidebar).

Values can be overridden with environment variables, e.g.:
    export BARCODE_OCR_CONF_THRESHOLD=70
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
INPUT_IMAGES_DIR = BASE_DIR / "input_images"
OUTPUTS_DIR = BASE_DIR / "outputs"
JSON_DIR = OUTPUTS_DIR / "json"
CROPPED_DIR = OUTPUTS_DIR / "cropped_regions"
LOGS_DIR = OUTPUTS_DIR / "logs"
REPORTS_DIR = OUTPUTS_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"

CSV_PATH = OUTPUTS_DIR / "results.csv"
XLSX_PATH = OUTPUTS_DIR / "results.xlsx"
ALL_RESULTS_JSON_PATH = JSON_DIR / "all_results.json"

SUPPORTED_EXTENSIONS: List[str] = [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"]

ALL_OUTPUT_DIRS = [
    OUTPUTS_DIR, JSON_DIR, CROPPED_DIR, LOGS_DIR, REPORTS_DIR,
    INPUT_IMAGES_DIR, MODELS_DIR,
]


# ── Engine priority lists (first available / highest confidence wins) ──────
BARCODE_ENGINE_PRIORITY: List[str] = ["opencv", "zxing", "pyzbar"]
OCR_ENGINE_PRIORITY: List[str] = ["paddleocr", "easyocr", "tesseract"]

# Languages requested for OCR engines that support multilingual models.
OCR_LANGUAGES: List[str] = ["en"]


@dataclass
class PreprocessConfig:
    """Every preprocessing step is independently toggleable and tunable."""

    auto_quality_assessment: bool = True
    resize_enabled: bool = True
    resize_scale: float = 2.0
    preserve_aspect_ratio: bool = True

    clahe_enabled: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: int = 8

    bilateral_filter_enabled: bool = True
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0

    gaussian_blur_enabled: bool = False  # only triggered dynamically on noisy images
    gaussian_kernel: int = 3

    adaptive_threshold_enabled: bool = True
    otsu_threshold_enabled: bool = True

    morph_open_enabled: bool = True
    morph_close_enabled: bool = True
    morph_kernel_size: int = 2

    sharpen_enabled: bool = True
    denoise_enabled: bool = True

    deskew_enabled: bool = True
    perspective_correction_enabled: bool = True
    rotation_correction_enabled: bool = True

    dynamic_pipeline: bool = True  # adapt steps based on quality score


@dataclass
class MatchingConfig:
    """Controls how OCR text is compared against the decoded barcode value."""

    similarity_threshold: float = 85.0          # percent, configurable
    use_regex_validation: bool = True
    use_levenshtein: bool = True
    use_sequence_matcher: bool = True
    levenshtein_weight: float = 0.5
    sequence_matcher_weight: float = 0.5
    apply_ocr_error_correction: bool = True


@dataclass
class PerformanceConfig:
    max_workers: int = field(default_factory=lambda: _env_int("BARCODE_OCR_WORKERS", min(8, (os.cpu_count() or 4))))
    enable_caching: bool = True
    enable_parallel_batch: bool = True
    ocr_timeout_seconds: float = 30.0


@dataclass
class OCRConfig:
    engine_priority: List[str] = field(default_factory=lambda: list(OCR_ENGINE_PRIORITY))
    languages: List[str] = field(default_factory=lambda: list(OCR_LANGUAGES))
    use_gpu: bool = False
    confidence_threshold: float = _env_float("BARCODE_OCR_CONF_THRESHOLD", 40.0)
    multi_engine_voting: bool = True


@dataclass
class BarcodeConfig:
    engine_priority: List[str] = field(default_factory=lambda: list(BARCODE_ENGINE_PRIORITY))


@dataclass
class AppSettings:
    """Aggregate settings object passed around the application."""

    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    barcode: BarcodeConfig = field(default_factory=BarcodeConfig)
    dark_theme: bool = True
    log_level: str = "DEBUG"


def get_default_settings() -> AppSettings:
    """Factory for a fresh, default `AppSettings` instance."""
    return AppSettings()
