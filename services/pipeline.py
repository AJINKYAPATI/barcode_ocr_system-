"""
services/pipeline.py
======================
Orchestrates preprocessing -> barcode detection -> OCR -> matching for a
single image, and parallel batch processing across many images.

Performance choices
--------------------
- `ThreadPoolExecutor` for batch scans (I/O + OpenCV/Tesseract calls release
  the GIL for most of their work, so threads — not processes — give the
  best throughput/complexity trade-off here and avoid pickling overhead).
- Engine objects (OCRService/BarcodeDetector) are instantiated once per
  `Pipeline` and reused for the whole batch instead of per image, since
  model/engine init is the expensive part.
- No unnecessary `.copy()` of full-resolution images between pipeline
  stages; only the small ROI crops are copied.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Optional

import numpy as np

from config import AppSettings
from core.barcode_detection import BarcodeDetector
from core.matching import match as match_texts
from core.models import ScanResult
from core.preprocessing import ImageProcessor
from logger import get_logger
from services.ocr_service import OCRService
from utils import get_image_number_from_path

log = get_logger(__name__)


class Pipeline:
    """End-to-end scan pipeline, reusing one set of engine instances across many images."""

    def __init__(self, settings: Optional[AppSettings] = None) -> None:
        self.settings = settings or AppSettings()
        self.processor = ImageProcessor(self.settings.preprocess)
        self.barcode_detector = BarcodeDetector(self.settings.barcode.engine_priority)
        self.ocr_service = OCRService(self.settings.ocr)

    def process_image(self, image: np.ndarray, filename: str = "", image_number: str = "") -> ScanResult:
        """Run the full pipeline on a single already-loaded image. Never raises."""
        t0 = time.perf_counter()
        try:
            barcode_result = self.barcode_detector.detect_barcode(image)
            roi_bbox = self._sanitize_bbox(barcode_result.bbox, image.shape)
            ocr_result = self.ocr_service.extract_text_from_roi(image, roi_bbox)
            match_result = match_texts(
                ocr_result.text,
                barcode_result.value,
                config=self.settings.matching,
                barcode_symbology=barcode_result.symbology.value,
                ocr_confidence=ocr_result.confidence,
                barcode_confidence=barcode_result.confidence,
            )
            return ScanResult(
                image_number=image_number,
                filename=filename,
                ocr=ocr_result,
                barcode=barcode_result,
                match=match_result,
                total_processing_time=round(time.perf_counter() - t0, 3),
            )
        except Exception as exc:
            log.error(f"process_image error ({filename}): {exc}")
            return ScanResult(
                image_number=image_number, filename=filename,
                total_processing_time=round(time.perf_counter() - t0, 3), error=str(exc),
            )

    @staticmethod
    def _sanitize_bbox(bbox: Optional[tuple], image_shape: tuple) -> Optional[tuple]:
        """
        Discard barcode bboxes that are too degenerate (near-zero width/height,
        or implausibly small relative to the image) to be a trustworthy hint
        for the OCR ROI crop. Returning None lets `crop_text_below_barcode`
        fall back to its own morphological detection / proportional crop.
        """
        if bbox is None:
            return None
        x, y, w, h = bbox
        img_h, img_w = image_shape[:2]
        if w < 0.25 * img_w or h < 4 or y < 0 or x < -img_w * 0.1:
            return None
        return bbox

    def process_path(self, path: str) -> ScanResult:
        """Load an image from disk (handling missing/corrupted files gracefully) then process it."""
        filename = path.rsplit("/", 1)[-1]
        image_number = get_image_number_from_path(path)
        try:
            image = self.processor.load_image(path)
            if image is None:
                return ScanResult(
                    image_number=image_number, filename=filename,
                    error="Could not read image — missing, corrupted, or unsupported format",
                )
            return self.process_image(image, filename=filename, image_number=image_number)
        except MemoryError as exc:
            log.error(f"MemoryError processing {path}: {exc}")
            return ScanResult(image_number=image_number, filename=filename, error="Out of memory while processing image")
        except Exception as exc:
            log.error(f"process_path error ({path}): {exc}")
            return ScanResult(image_number=image_number, filename=filename, error=str(exc))

    def process_batch(
        self,
        paths: List[str],
        progress_callback: Optional[Callable[[int, int, ScanResult], None]] = None,
    ) -> List[ScanResult]:
        """
        Process many images in parallel via a thread pool.

        `progress_callback(done_count, total_count, latest_result)` is invoked
        as each image finishes (order of completion, not submission order) so
        the UI can drive a live progress bar / ETA.
        """
        if not paths:
            return []

        max_workers = self.settings.performance.max_workers if self.settings.performance.enable_parallel_batch else 1
        results: List[Optional[ScanResult]] = [None] * len(paths)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_idx = {pool.submit(self.process_path, p): i for i, p in enumerate(paths)}
            done = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                except Exception as exc:
                    log.error(f"Unhandled batch worker error for {paths[idx]}: {exc}")
                    result = ScanResult(filename=paths[idx].rsplit("/", 1)[-1], error=str(exc))
                results[idx] = result
                done += 1
                if progress_callback:
                    progress_callback(done, len(paths), result)

        return [r for r in results if r is not None]
