import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.models import BarcodeResult, BarcodeSymbology, MatchResult, MatchStatus, OCRResult, ScanResult, Status
from services.report_service import ReportService


def _sample_result() -> ScanResult:
    return ScanResult(
        image_number="1",
        filename="1.png",
        ocr=OCRResult(text="ABC123", confidence=91.0, status=Status.SUCCESS, engine_used="tesseract"),
        barcode=BarcodeResult(value="ABC123", symbology=BarcodeSymbology.CODE128, status=Status.SUCCESS, confidence=90.0, engine_used="pyzbar"),
        match=MatchResult(status=MatchStatus.MATCH, similarity_percent=100.0, confidence_percent=95.0, reason="Exact match"),
        total_processing_time=0.42,
    )


def test_to_dataframe_has_expected_columns(tmp_path):
    svc = ReportService(output_dir=str(tmp_path))
    df = svc.to_dataframe([_sample_result()])
    assert "Barcode" in df.columns
    assert "OCR Text" in df.columns
    assert len(df) == 1


def test_save_to_csv(tmp_path):
    svc = ReportService(output_dir=str(tmp_path))
    assert svc.save_to_csv([_sample_result()]) is True
    assert Path(svc.csv_path).exists()


def test_save_to_excel(tmp_path):
    svc = ReportService(output_dir=str(tmp_path))
    assert svc.save_to_excel([_sample_result()]) is True
    assert Path(svc.xlsx_path).exists()


def test_save_all_json(tmp_path):
    svc = ReportService(output_dir=str(tmp_path))
    assert svc.save_all_json([_sample_result()]) is True


def test_get_csv_bytes_nonempty(tmp_path):
    svc = ReportService(output_dir=str(tmp_path))
    data = svc.get_csv_bytes([_sample_result()])
    assert b"ABC123" in data


def test_save_to_pdf_returns_path(tmp_path):
    svc = ReportService(output_dir=str(tmp_path))
    path = svc.save_to_pdf([_sample_result()], summary={"Total": 1})
    assert path is not None
    assert Path(path).exists()
