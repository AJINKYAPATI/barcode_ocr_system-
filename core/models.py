"""
core/models.py
===============
Shared dataclasses and enums passed between services, the pipeline,
and the UI. Centralising these avoids the "dict soup" anti-pattern
(string keys with no schema) used in the original codebase.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class Status(str, Enum):
    SUCCESS = "SUCCESS"
    EMPTY = "EMPTY"
    FAILED = "FAILED"
    ERROR = "ERROR"
    SKIPPED = "SKIPPED"


class MatchStatus(str, Enum):
    MATCH = "MATCH"
    MISMATCH = "MISMATCH"
    UNKNOWN = "UNKNOWN"
    ERROR = "ERROR"


class BarcodeSymbology(str, Enum):
    CODE128 = "CODE_128"
    CODE39 = "CODE_39"
    EAN13 = "EAN_13"
    EAN8 = "EAN_8"
    UPC_A = "UPC_A"
    UPC_E = "UPC_E"
    QR_CODE = "QR_CODE"
    UNKNOWN = "UNKNOWN"


@dataclass
class BarcodeResult:
    value: str = ""
    symbology: BarcodeSymbology = BarcodeSymbology.UNKNOWN
    status: Status = Status.FAILED
    confidence: float = 0.0
    bbox: Optional[tuple] = None
    engine_used: str = ""
    processing_time: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["symbology"] = self.symbology.value
        d["status"] = self.status.value
        return d


@dataclass
class OCRCandidate:
    """A single engine's raw vote, before multi-engine arbitration."""

    engine: str
    text: str
    confidence: float  # 0-100
    processing_time: float = 0.0


@dataclass
class OCRResult:
    text: str = ""
    confidence: float = 0.0
    status: Status = Status.FAILED
    engine_used: str = ""
    candidates: List[OCRCandidate] = field(default_factory=list)
    processing_time: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass
class MatchResult:
    status: MatchStatus = MatchStatus.UNKNOWN
    similarity_percent: float = 0.0
    confidence_percent: float = 0.0
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass
class ScanResult:
    """The full result of processing a single image end to end."""

    image_number: str = ""
    filename: str = ""
    ocr: OCRResult = field(default_factory=OCRResult)
    barcode: BarcodeResult = field(default_factory=BarcodeResult)
    match: MatchResult = field(default_factory=MatchResult)
    total_processing_time: float = 0.0
    scan_time: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    error: Optional[str] = None

    def to_flat_dict(self) -> Dict[str, Any]:
        """Flattened representation used by report_service and the history table."""
        return {
            "image_number": self.image_number,
            "filename": self.filename,
            "ocr_text": self.ocr.text,
            "ocr_confidence": self.ocr.confidence,
            "ocr_engine": self.ocr.engine_used,
            "barcode_value": self.barcode.value,
            "barcode_type": self.barcode.symbology.value if isinstance(self.barcode.symbology, BarcodeSymbology) else self.barcode.symbology,
            "barcode_engine": self.barcode.engine_used,
            "match_status": self.match.status.value if isinstance(self.match.status, MatchStatus) else self.match.status,
            "similarity_percent": self.match.similarity_percent,
            "match_confidence": self.match.confidence_percent,
            "mismatch_reason": self.match.reason,
            "processing_time": self.total_processing_time,
            "scan_time": self.scan_time,
            "error": self.error or "",
        }
