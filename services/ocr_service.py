"""
services/ocr_service.py
=========================
Multi-engine OCR pipeline with confidence voting.

Engine priority (config.OCR_ENGINE_PRIORITY): PaddleOCR -> EasyOCR -> Tesseract.

Design note — engine availability
----------------------------------
PaddleOCR and EasyOCR are powerful but heavy (PaddlePaddle / PyTorch
backends, GPU-oriented, hundreds of MB to multiple GB of weights).
This service treats every engine as **optional and lazily imported**:
if PaddleOCR/EasyOCR aren't installed in the deployment environment,
they're skipped with a log entry and Tesseract (lightweight, CPU-only,
installed by default here) carries the workload alone. Install either
extra engine (`pip install paddleocr` / `pip install easyocr`) and it
is picked up automatically on next run — no code changes required.

When more than one engine IS available, each engine "votes" with its
own (text, confidence) reading of the same ROI; `_vote()` picks the
highest-confidence result, breaking ties by majority text agreement.
"""

from __future__ import annotations

import re
import time
from typing import List, Optional

import numpy as np

from config import OCRConfig
from core.models import OCRCandidate, OCRResult, Status
from core.preprocessing import ImageProcessor
from core.text_cleaning import clean_text
from logger import get_logger

log = get_logger(__name__)

# Barcode-style alphanumeric token: uppercase letters/digits, 4+ chars.
_BARCODE_LIKE_PATTERN = re.compile(r"^[A-Z0-9\-]{4,}$")


class _TesseractEngine:
    name = "tesseract"

    def __init__(self, languages: List[str]) -> None:
        self.lang_code = "+".join(languages) if languages else "eng"
        try:
            import pytesseract  # noqa: PLC0415
            self._lib = pytesseract
            # Confirms the tesseract *binary* is actually on PATH, not just the python wrapper.
            self._lib.get_tesseract_version()
            self.available = True
        except Exception as exc:
            log.warning(f"Tesseract unavailable: {exc}")
            self._lib = None
            self.available = False

    def read(self, image: np.ndarray) -> List[OCRCandidate]:
        if not self.available:
            return []
        t0 = time.perf_counter()
        try:
            data = self._lib.image_to_data(
                image, lang=self._map_lang(), output_type=self._lib.Output.DICT,
                config="--psm 7",  # treat ROI as a single line — matches barcode-label text
            )
            tokens, confs = [], []
            for text, conf in zip(data.get("text", []), data.get("conf", [])):
                text = text.strip()
                if text and float(conf) > 0:
                    tokens.append(text)
                    confs.append(float(conf))
            if not tokens:
                return []
            joined = " ".join(tokens)
            avg_conf = sum(confs) / len(confs)
            return [OCRCandidate(self.name, joined, avg_conf, round(time.perf_counter() - t0, 3))]
        except Exception as exc:
            log.debug(f"Tesseract read failed: {exc}")
            return []

    def _map_lang(self) -> str:
        # pytesseract expects 3-letter codes; 'en' -> 'eng' is the common case.
        return "eng" if self.lang_code in ("en", "eng") else self.lang_code


class _EasyOCREngine:
    name = "easyocr"

    def __init__(self, languages: List[str], use_gpu: bool) -> None:
        self._reader = None
        self.available = False
        try:
            import easyocr  # noqa: PLC0415
            self._reader = easyocr.Reader(languages or ["en"], gpu=use_gpu)
            self.available = True
        except Exception as exc:
            log.info(f"EasyOCR not available (optional engine): {exc}")

    def read(self, image: np.ndarray) -> List[OCRCandidate]:
        if not self.available:
            return []
        t0 = time.perf_counter()
        try:
            results = self._reader.readtext(image, detail=1)
            if not results:
                return []
            best = max(results, key=lambda r: r[2])
            text = best[1].strip()
            conf = best[2] * 100.0
            return [OCRCandidate(self.name, text, conf, round(time.perf_counter() - t0, 3))]
        except Exception as exc:
            log.debug(f"EasyOCR read failed: {exc}")
            return []


class _PaddleOCREngine:
    name = "paddleocr"

    def __init__(self, languages: List[str], use_gpu: bool) -> None:
        self._engine = None
        self.available = False
        try:
            from paddleocr import PaddleOCR  # noqa: PLC0415
            lang = "en" if "en" in (languages or ["en"]) else (languages or ["en"])[0]
            self._engine = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
            self.available = True
        except Exception as exc:
            log.info(f"PaddleOCR not available (optional engine): {exc}")

    def read(self, image: np.ndarray) -> List[OCRCandidate]:
        if not self.available:
            return []
        t0 = time.perf_counter()
        try:
            result = self._engine.ocr(image, cls=True)
            flat = [line for block in (result or []) for line in (block or [])]
            if not flat:
                return []
            best = max(flat, key=lambda r: r[1][1])
            text, conf = best[1][0].strip(), best[1][1] * 100.0
            return [OCRCandidate(self.name, text, conf, round(time.perf_counter() - t0, 3))]
        except Exception as exc:
            log.debug(f"PaddleOCR read failed: {exc}")
            return []


class OCRService:
    """Multi-engine OCR with confidence-based voting across whichever engines are installed."""

    def __init__(self, config: Optional[OCRConfig] = None) -> None:
        self.config = config or OCRConfig()
        self._processor = ImageProcessor()
        self._engines = {
            "tesseract": _TesseractEngine(self.config.languages),
            "easyocr": _EasyOCREngine(self.config.languages, self.config.use_gpu),
            "paddleocr": _PaddleOCREngine(self.config.languages, self.config.use_gpu),
        }
        log.info(f"OCR engines available: {self.available_engines()}")

    def available_engines(self) -> List[str]:
        return [n for n in self.config.engine_priority if getattr(self._engines.get(n), "available", False)]

    def extract_text(self, image: np.ndarray) -> OCRResult:
        """Run every available engine (in priority order) on *image* and vote on the result."""
        t0 = time.perf_counter()
        try:
            enhanced, cleaned = self._processor.preprocess_for_ocr(image)
            candidates: List[OCRCandidate] = []

            for engine_name in self.config.engine_priority:
                engine = self._engines.get(engine_name)
                if engine is None or not getattr(engine, "available", False):
                    continue
                candidates.extend(engine.read(enhanced))
                if not self.config.multi_engine_voting and candidates:
                    break  # single-engine mode: stop at first available engine

            if not candidates:
                # Retry on the binary-cleaned variant before giving up — some
                # engines do noticeably better on a thresholded image.
                for engine_name in self.config.engine_priority:
                    engine = self._engines.get(engine_name)
                    if engine is None or not getattr(engine, "available", False):
                        continue
                    candidates.extend(engine.read(cleaned))

            elapsed = round(time.perf_counter() - t0, 3)
            if not candidates:
                log.warning("OCR returned no results from any engine")
                return OCRResult(status=Status.EMPTY, processing_time=elapsed, candidates=[])

            best_text, best_conf, best_engine = self._vote(candidates)
            cleaned_text = clean_text(best_text, remove_unwanted=True)

            log.info(f"OCR completed -> '{cleaned_text}' ({best_conf:.1f}%) via {best_engine} in {elapsed}s")
            return OCRResult(
                text=cleaned_text or best_text,
                confidence=round(best_conf, 1),
                status=Status.SUCCESS if (cleaned_text or best_text) else Status.EMPTY,
                engine_used=best_engine,
                candidates=candidates,
                processing_time=elapsed,
            )
        except Exception as exc:
            log.error(f"extract_text error: {exc}")
            return OCRResult(status=Status.ERROR, processing_time=round(time.perf_counter() - t0, 3), error=str(exc))

    def extract_text_from_roi(self, image: np.ndarray, barcode_bbox: Optional[tuple] = None) -> OCRResult:
        """Crop the text ROI near the barcode (instead of OCRing the whole image) then run `extract_text`."""
        try:
            region = self._processor.crop_text_below_barcode(image, barcode_bbox)
            if region is None or region.size == 0:
                region = image
            return self.extract_text(region)
        except Exception as exc:
            log.error(f"extract_text_from_roi error: {exc}")
            return self.extract_text(image)

    # ── Voting ───────────────────────────────────────────────────────────

    @staticmethod
    def _vote(candidates: List[OCRCandidate]) -> tuple[str, float, str]:
        """
        Pick the winning (text, confidence, engine) among all engine candidates.

        Preference: highest-confidence candidate whose cleaned text matches the
        barcode-like alphanumeric pattern; otherwise the highest-confidence
        candidate overall. Ties on confidence break by majority text agreement
        across engines (i.e. if two engines agree, that text wins over a lone
        higher-confidence outlier within 5 points).
        """
        normalized = [(c, re.sub(r"\s+", "", c.text).upper()) for c in candidates]

        pattern_matches = [(c, t) for c, t in normalized if _BARCODE_LIKE_PATTERN.match(t)]
        pool = pattern_matches or normalized

        # Majority agreement check
        from collections import Counter
        text_counts = Counter(t for _, t in pool)
        majority_text, majority_n = text_counts.most_common(1)[0]

        best_c, best_t = max(pool, key=lambda ct: ct[0].confidence)
        if majority_n > 1 and majority_text != best_t:
            agreeing = [c for c, t in pool if t == majority_text]
            best_among_agreeing = max(agreeing, key=lambda c: c.confidence)
            if best_among_agreeing.confidence >= best_c.confidence - 5.0:
                return best_among_agreeing.text.strip().upper(), best_among_agreeing.confidence, best_among_agreeing.engine

        return best_c.text.strip().upper(), best_c.confidence, best_c.engine
