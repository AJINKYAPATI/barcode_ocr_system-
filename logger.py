"""
logger.py
=========
Structured, rotating logging for the Barcode OCR System.

Replaces all `print()` usage across the codebase. Supports INFO,
WARNING, ERROR and DEBUG levels, writes to both console and a rotating
file under outputs/logs, and exposes a single module-level `logger`
singleton plus a `get_logger(name)` factory for per-module child loggers.
"""

from __future__ import annotations

import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_DIR = Path(__file__).resolve().parent / "outputs" / "logs"
_ROOT_LOGGER_NAME = "BarcodeOCR"
_initialized = False


def _build_root_logger(log_dir: Path = _LOG_DIR, level: str = "DEBUG") -> logging.Logger:
    global _initialized
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger(_ROOT_LOGGER_NAME)
    root.setLevel(getattr(logging, level.upper(), logging.DEBUG))

    if _initialized or root.handlers:
        return root

    log_filename = datetime.now().strftime("barcode_ocr_%Y%m%d_%H%M%S.log")
    log_filepath = log_dir / log_filename

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%H:%M:%S")
    )

    # Rotating file handler: keeps log files bounded (5 MB x 5 backups)
    file_handler = RotatingFileHandler(
        log_filepath, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        )
    )

    root.addHandler(console_handler)
    root.addHandler(file_handler)
    root.propagate = False
    _initialized = True
    root.info(f"Logger initialised -> {log_filepath}")
    return root


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child logger (e.g. `get_logger(__name__)`) sharing root handlers."""
    _build_root_logger()
    if name:
        return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")
    return logging.getLogger(_ROOT_LOGGER_NAME)


# Module-level singleton kept for backward compatibility with the original
# `from logger import logger` import style used throughout the project.
logger = _build_root_logger()
