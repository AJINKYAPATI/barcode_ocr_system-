"""
utils.py
========
Shared, dependency-light utility functions: image discovery, path
helpers, and PIL/OpenCV conversions. No business logic lives here.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image

from config import ALL_OUTPUT_DIRS, SUPPORTED_EXTENSIONS
from logger import get_logger

log = get_logger(__name__)


def find_image_by_number(folder_path: str, image_number: int) -> Optional[str]:
    """Locate an image file whose base name matches *image_number*, trying common naming conventions."""
    if not os.path.isdir(folder_path):
        log.warning(f"Folder not found: {folder_path}")
        return None

    candidates = []
    for ext in SUPPORTED_EXTENSIONS:
        candidates += [
            os.path.join(folder_path, f"{image_number}{ext}"),
            os.path.join(folder_path, f"{image_number:02d}{ext}"),
            os.path.join(folder_path, f"{image_number:03d}{ext}"),
            os.path.join(folder_path, f"image_{image_number}{ext}"),
            os.path.join(folder_path, f"img_{image_number}{ext}"),
            os.path.join(folder_path, f"scan_{image_number}{ext}"),
        ]

    for path in candidates:
        if os.path.isfile(path):
            return path

    for fname in sorted(os.listdir(folder_path)):
        stem, ext = os.path.splitext(fname)
        if ext.lower() in SUPPORTED_EXTENSIONS and stem == str(image_number):
            return os.path.join(folder_path, fname)

    log.warning(f"Image number {image_number} not found in '{folder_path}'")
    return None


def get_all_images(folder_path: str) -> List[str]:
    """Return all supported image files in *folder_path*, sorted by embedded number."""
    if not os.path.isdir(folder_path):
        return []

    images = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ]

    def _sort_key(p: str) -> int:
        nums = re.findall(r"\d+", os.path.splitext(os.path.basename(p))[0])
        return int(nums[0]) if nums else 0

    return sorted(images, key=_sort_key)


def get_image_number_from_path(path: str) -> str:
    """Extract the first numeric substring from a filename, or fall back to the stem."""
    stem = os.path.splitext(os.path.basename(path))[0]
    nums = re.findall(r"\d+", stem)
    return nums[0] if nums else stem


def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    rgb = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
    if len(cv2_image.shape) == 2:
        return Image.fromarray(cv2_image)
    rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def ensure_output_dirs() -> None:
    """Create every required output/input subdirectory if missing."""
    for d in ALL_OUTPUT_DIRS:
        Path(d).mkdir(parents=True, exist_ok=True)


def format_confidence(confidence: float) -> str:
    return f"{confidence:.1f}%"


def is_supported_format(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in SUPPORTED_EXTENSIONS
