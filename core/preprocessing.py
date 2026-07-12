"""
core/preprocessing.py
======================
Configurable OpenCV preprocessing pipeline.

Every step is independently toggleable via `PreprocessConfig` (see
config.py). `assess_quality()` scores an image on sharpness, contrast
and noise so `dynamic_pipeline()` can decide *which* steps a given
image actually needs instead of always running the full, slow chain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from config import PreprocessConfig
from logger import get_logger

log = get_logger(__name__)


@dataclass
class QualityReport:
    sharpness: float           # variance of Laplacian; higher = sharper
    contrast: float            # std-dev of pixel intensities
    brightness: float          # mean pixel intensity
    noise_estimate: float      # high-frequency energy estimate
    is_blurry: bool
    is_low_contrast: bool
    is_noisy: bool
    overall_score: float       # 0-100, higher = better quality


class ImageProcessor:
    """Encapsulates all image preprocessing operations as small, composable steps."""

    def __init__(self, config: Optional[PreprocessConfig] = None) -> None:
        self.config = config or PreprocessConfig()
        self._clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=(self.config.clahe_tile_grid, self.config.clahe_tile_grid),
        )

    # ── Loading ──────────────────────────────────────────────────────────

    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        try:
            img = cv2.imread(image_path)
            if img is None:
                log.error(f"cv2.imread returned None for: {image_path}")
                return None
            log.info(f"Image loaded: {image_path} shape={img.shape}")
            return img
        except Exception as exc:
            log.error(f"load_image error: {exc}")
            return None

    def load_from_pil(self, pil_image: Image.Image) -> np.ndarray:
        rgb = np.array(pil_image.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def load_from_bytes(self, data: bytes) -> Optional[np.ndarray]:
        try:
            arr = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                log.error("imdecode returned None - possibly corrupt data")
            return img
        except Exception as exc:
            log.error(f"load_from_bytes error: {exc}")
            return None

    # ── Quality assessment ───────────────────────────────────────────────

    def assess_quality(self, image: np.ndarray) -> QualityReport:
        """
        Score an image's suitability for OCR/barcode decoding.

        Used by `dynamic_pipeline()` to skip or add steps automatically,
        and surfaced in the UI so users understand *why* a scan failed.
        """
        gray = self.to_grayscale(image)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        contrast = float(gray.std())
        brightness = float(gray.mean())

        # Noise estimate: high-frequency energy after a median-filter residual.
        median = cv2.medianBlur(gray, 3)
        noise_estimate = float(np.mean(np.abs(gray.astype(np.float32) - median.astype(np.float32))))

        is_blurry = laplacian_var < 100.0
        is_low_contrast = contrast < 35.0
        is_noisy = noise_estimate > 8.0

        # Weighted 0-100 composite score (heuristic, tuned for label photos).
        score = 100.0
        if is_blurry:
            score -= 35
        if is_low_contrast:
            score -= 25
        if is_noisy:
            score -= 20
        if brightness < 40 or brightness > 220:
            score -= 10
        score = max(0.0, min(100.0, score))

        return QualityReport(
            sharpness=laplacian_var,
            contrast=contrast,
            brightness=brightness,
            noise_estimate=noise_estimate,
            is_blurry=is_blurry,
            is_low_contrast=is_low_contrast,
            is_noisy=is_noisy,
            overall_score=score,
        )

    # ── Individual steps ──────────────────────────────────────────────────

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def resize_image(self, image: np.ndarray, scale: float = 2.0, preserve_aspect: bool = True) -> np.ndarray:
        h, w = image.shape[:2]
        if preserve_aspect:
            new_w, new_h = int(round(w * scale)), int(round(h * scale))
        else:
            new_w, new_h = int(w * scale), int(h * scale)
        interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        return cv2.resize(image, (max(1, new_w), max(1, new_h)), interpolation=interp)

    def denoise(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)

    def bilateral_filter(self, image: np.ndarray) -> np.ndarray:
        """Edge-preserving smoothing — better than Gaussian for text/barcode edges."""
        cfg = self.config
        return cv2.bilateralFilter(image, cfg.bilateral_d, cfg.bilateral_sigma_color, cfg.bilateral_sigma_space)

    def gaussian_blur(self, image: np.ndarray) -> np.ndarray:
        k = self.config.gaussian_kernel | 1  # force odd
        return cv2.GaussianBlur(image, (k, k), 0)

    def apply_clahe(self, gray: np.ndarray) -> np.ndarray:
        return self._clahe.apply(gray)

    def adaptive_threshold(self, gray: np.ndarray) -> np.ndarray:
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    def otsu_threshold(self, gray: np.ndarray) -> np.ndarray:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def morphological_open(self, binary: np.ndarray) -> np.ndarray:
        k = self.config.morph_kernel_size
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    def morphological_close(self, binary: np.ndarray) -> np.ndarray:
        k = self.config.morph_kernel_size
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    def sharpen(self, image: np.ndarray) -> np.ndarray:
        """Unsharp-mask style sharpening kernel."""
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)

    # ── Geometric correction ─────────────────────────────────────────────

    def correct_rotation(self, image: np.ndarray) -> np.ndarray:
        """Deskew using the dominant Hough line angle. Skips if angle < 0.5deg."""
        try:
            gray = self.to_grayscale(image)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi / 180, 80)
            if lines is None:
                return image

            angles = []
            for rho, theta in lines[:, 0]:
                angle = (theta * 180 / np.pi) - 90
                if abs(angle) < 45:
                    angles.append(angle)
            if not angles:
                return image

            median_angle = float(np.median(angles))
            if abs(median_angle) < 0.5:
                return image

            h, w = image.shape[:2]
            matrix = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)
            rotated = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            log.debug(f"Rotation corrected by {median_angle:.2f} deg")
            return rotated
        except Exception as exc:
            log.error(f"correct_rotation error: {exc}")
            return image

    def correct_perspective(self, image: np.ndarray, quad: Optional[np.ndarray]) -> np.ndarray:
        """Warp a detected quadrangle (4 corner points) into a flat rectangle."""
        if quad is None:
            return image
        try:
            pts = quad.astype(np.float32)
            pad = 12
            width = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2]))) + pad * 2
            height = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3]))) + pad * 2
            width, height = max(width, 10), max(height, 10)
            dst = np.array(
                [[pad, pad], [pad, height - pad], [width - pad, height - pad], [width - pad, pad]],
                dtype=np.float32,
            )
            matrix = cv2.getPerspectiveTransform(pts, dst)
            return cv2.warpPerspective(image, matrix, (width, height))
        except Exception as exc:
            log.error(f"correct_perspective error: {exc}")
            return image

    # ── Region detection / cropping ──────────────────────────────────────

    def detect_barcode_bbox(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Morphological-gradient based barcode strip localisation."""
        try:
            gray = self.to_grayscale(image)
            gx = cv2.Sobel(gray, cv2.CV_32F, dx=1, dy=0, ksize=-1)
            gy = cv2.Sobel(gray, cv2.CV_32F, dx=0, dy=1, ksize=-1)
            gradient = cv2.convertScaleAbs(cv2.subtract(gx, gy))
            blurred = cv2.blur(gradient, (9, 9))
            _, thresh = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            eroded = cv2.erode(closed, None, iterations=4)
            dilated = cv2.dilate(eroded, None, iterations=4)
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None
            c = max(contours, key=cv2.contourArea)
            return cv2.boundingRect(c)
        except Exception as exc:
            log.error(f"detect_barcode_bbox error: {exc}")
            return None

    def crop_text_below_barcode(self, image: np.ndarray, barcode_bbox: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """
        Isolate the printed-text ROI so OCR never has to scan the whole label.

        Priority: caller-supplied barcode bbox -> morphological detection ->
        proportional bottom-30% fallback.
        """
        h, w = image.shape[:2]
        bbox = barcode_bbox or self.detect_barcode_bbox(image)
        if bbox is not None:
            bx, by, bw, bh = bbox
            y_start = by + bh + 2
            # Human-readable text under a barcode is typically a similar
            # height to the barcode's own bar height, not a quarter of the
            # whole image — using the latter pulls in unrelated rows above
            # the actual digits and corrupts OCR with extra characters.
            y_end = min(h, y_start + max(35, int(bh * 0.7)))
            region = image[y_start:y_end, :]
            if region.shape[0] > 8:
                log.debug("Text ROI cropped using barcode bbox")
                return region

        y_start = int(h * 0.65)
        log.debug("Text ROI: bottom-30% proportional fallback")
        return image[y_start:, :]

    def crop_barcode_region(self, image: np.ndarray, bbox: Optional[Tuple[int, int, int, int]], padding: int = 10) -> Optional[np.ndarray]:
        if bbox is None:
            return None
        x, y, bw, bh = bbox
        ih, iw = image.shape[:2]
        x1, y1 = max(0, x - padding), max(0, y - padding)
        x2, y2 = min(iw, x + bw + padding), min(ih, y + bh + padding)
        return image[y1:y2, x1:x2]

    # ── Full pipelines ────────────────────────────────────────────────────

    def dynamic_pipeline(self, image: np.ndarray) -> Tuple[np.ndarray, QualityReport]:
        """
        Run only the preprocessing steps the image actually needs, based on
        `assess_quality()`. Falls back to the full static pipeline if
        `config.dynamic_pipeline` is disabled.
        """
        quality = self.assess_quality(image)
        cfg = self.config

        if not cfg.dynamic_pipeline:
            return self.preprocess_for_ocr(image)[0], quality

        gray = self.to_grayscale(image)
        working = gray

        if cfg.resize_enabled:
            working = self.resize_image(working, cfg.resize_scale, cfg.preserve_aspect_ratio)

        if quality.is_noisy and cfg.denoise_enabled:
            working = self.denoise(working)
        elif cfg.bilateral_filter_enabled:
            working = self.bilateral_filter(working)

        if quality.is_blurry and cfg.sharpen_enabled:
            working = self.sharpen(working)

        if quality.is_low_contrast and cfg.clahe_enabled:
            working = self.apply_clahe(working)

        if cfg.rotation_correction_enabled:
            working = self.correct_rotation(working)

        log.info(
            f"Dynamic pipeline: quality={quality.overall_score:.0f} "
            f"blurry={quality.is_blurry} low_contrast={quality.is_low_contrast} noisy={quality.is_noisy}"
        )
        return working, quality

    def preprocess_for_ocr(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Full static pipeline tuned for OCR: gray -> resize -> denoise -> CLAHE -> threshold -> morph."""
        try:
            cfg = self.config
            gray = self.to_grayscale(image)
            resized = self.resize_image(gray, cfg.resize_scale) if cfg.resize_enabled else gray
            denoised = self.denoise(resized) if cfg.denoise_enabled else resized
            enhanced = self.apply_clahe(denoised) if cfg.clahe_enabled else denoised
            binary = self.adaptive_threshold(enhanced) if cfg.adaptive_threshold_enabled else self.otsu_threshold(enhanced)
            cleaned = binary
            if cfg.morph_open_enabled:
                cleaned = self.morphological_open(cleaned)
            if cfg.morph_close_enabled:
                cleaned = self.morphological_close(cleaned)
            return enhanced, cleaned
        except Exception as exc:
            log.error(f"preprocess_for_ocr error: {exc}")
            gray = self.to_grayscale(image)
            return gray, gray

    def draw_boxes(self, image: np.ndarray, results: list) -> np.ndarray:
        """Draw bounding boxes + labels from any (bbox, text, conf) style result list."""
        try:
            annotated = image.copy()
            if len(annotated.shape) == 2:
                annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)
            for item in results:
                if len(item) < 3:
                    continue
                bbox, text, conf = item[0], item[1], item[2]
                pts = np.array(bbox, dtype=np.int32)
                cv2.polylines(annotated, [pts], True, (0, 200, 80), 2)
                x, y = int(pts[0][0]), int(pts[0][1])
                cv2.putText(
                    annotated, f"{text} ({conf * 100:.0f}%)", (x, max(0, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 80), 2,
                )
            return annotated
        except Exception as exc:
            log.error(f"draw_boxes error: {exc}")
            return image
