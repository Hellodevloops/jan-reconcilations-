import os
import re
import io
import uuid
import csv
import json
import shutil
import subprocess
import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple



import pdfplumber

try:

    import fitz  # type: ignore

except Exception:

    fitz = None  # type: ignore

try:

    import pypdfium2 as pdfium  # type: ignore

except Exception:

    pdfium = None  # type: ignore

try:

    from PIL import Image, ImageEnhance, ImageOps, ImageChops, ImageFilter  # type: ignore

except Exception:

    Image = None  # type: ignore

    ImageEnhance = None  # type: ignore

    ImageOps = None  # type: ignore

    ImageChops = None  # type: ignore

    ImageFilter = None  # type: ignore

try:

    import pytesseract  # type:0 ignore

except Exception:

    pytesseract = None  # type: ignore

try:

    from pytesseract import Output as TesseractOutput  # type: ignore

except Exception:

    TesseractOutput = None  # type: ignore

from fastapi import FastAPI, File, Request, UploadFile

from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

from fastapi.staticfiles import StaticFiles

from fastapi.templating import Jinja2Templates

from starlette.datastructures import UploadFile as StarletteUploadFile



APP_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = os.path.join(APP_DIR, ".outputs")



DEBUG = os.getenv("BANKPDF_DEBUG", "").strip().lower() in {"1", "true", "yes", "y", "on"}

APP_VERSION = os.getenv("APP_VERSION", "dev")

INVOICE_ENABLE_LABEL_OCR = os.getenv("INVOICE_ENABLE_LABEL_OCR", "").strip().lower() in {"1", "true", "yes", "y", "on"}

BANK_ENABLE_LABEL_OCR = os.getenv("BANK_ENABLE_LABEL_OCR", "").strip().lower() in {"1", "true", "yes", "y", "on"}

BANKPDF_OCR = os.getenv("BANKPDF_OCR", "").strip().lower() not in {"0", "false", "no", "n", "off"}


CURRENCY_RE = re.compile(r"\(?\s*-?\s*(?:£|\$|€)?\s*\d[\d,]*\.\d{2}\s*\)?")



app = FastAPI(title="Bank Statement PDF → CSV")



templates = Jinja2Templates(directory=os.path.join(APP_DIR, "templates"))



JOBS: Dict[str, str] = {}

INVOICE_JOBS: Dict[str, str] = {}

INVOICE_REVIEW_JOBS: Dict[str, str] = {}



@app.get("/", response_class=HTMLResponse)

async def home(request: Request) -> HTMLResponse:

    return templates.TemplateResponse("home.html", {"request": request})



@app.get("/invoice", response_class=HTMLResponse)

async def invoice_page(request: Request) -> HTMLResponse:

    return templates.TemplateResponse("invoice.html", {"request": request})


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    s = str(value)
    s = s.replace("\u00a0", " ")
    s = s.replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s)
    return s.strip()


def _is_valid_uk_date(value: Any) -> bool:
    s = _clean_text(value)
    if not s:
        return False
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{2,4})$", s)
    if not m:
        return False
    try:
        d = int(m.group(1))
        mo = int(m.group(2))
        y = int(m.group(3))
    except Exception:
        return False
    if d < 1 or d > 31:
        return False
    if mo < 1 or mo > 12:
        return False
    if y < 0:
        return False
    return True


def _parse_money(value: str) -> float:
    s = _clean_text(value)
    if not s:
        raise ValueError("Empty amount")
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1]
    s = s.replace("£", "").replace("$", "").replace("€", "")
    s = s.replace(",", "").replace(" ", "")
    if s.endswith("-"):
        neg = True
        s = s[:-1]
    amt = float(s)
    return -amt if neg else amt


def _format_csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.2f}"
    return _clean_text(value)


def _write_csv(csv_path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: _format_csv_value(r.get(k)) for k in fieldnames})


def _write_json(json_path: str, data: Any) -> None:
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _read_json(json_path: str) -> Any:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _tesseract_available() -> Tuple[bool, str]:
    if pytesseract is None:
        return False, "pytesseract is not installed"

    def _exit_code_detail(rc: int) -> str:
        try:
            return f"{rc} (0x{(rc & 0xFFFFFFFF):08X})"
        except Exception:
            return str(rc)

    def _check_cmd(cmd_path: str) -> Tuple[bool, str]:
        try:
            p = subprocess.run(
                [cmd_path, "--version"],
                capture_output=True,
                text=True,
                timeout=6,
            )
        except FileNotFoundError:
            return False, f"tesseract.exe not found at '{cmd_path}'"
        except Exception as e:
            return False, f"Failed to execute '{cmd_path} --version': {e}"

        out = (p.stdout or "").strip()
        err = (p.stderr or "").strip()
        if p.returncode != 0:
            msg = (
                f"Tesseract failed to start: cmd='{cmd_path}', exit_code={_exit_code_detail(p.returncode)}"
            )
            if err:
                msg += f"; stderr='{err[:500]}'"
            if out:
                msg += f"; stdout='{out[:500]}'"
            msg += (
                ". On Windows this often indicates a broken/corrupted Tesseract install, missing runtime dependencies, or antivirus interference. "
                "Reinstall Tesseract (commonly the UB Mannheim build) and ensure the install folder is on PATH, or set TESSERACT_CMD explicitly."
            )
            return False, msg
        if not out and not err:
            return False, f"Tesseract returned success but produced no output: cmd='{cmd_path}'"
        return True, ""

    cmd = os.getenv("TESSERACT_CMD", "").strip()
    if not cmd:
        try:
            cmd = shutil.which("tesseract") or ""
        except Exception:
            cmd = ""
    if not cmd:
        candidates = [
            r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
            r"C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe",
        ]
        for p in candidates:
            if os.path.exists(p):
                cmd = p
                break
    if cmd:
        # Windows fix: ensure Tesseract's folder is on PATH and available for DLL loading.
        # This prevents crashes like exit status 3221225794.
        try:
            os.environ["TESSERACT_CMD"] = cmd
        except Exception:
            pass

        try:
            tdir = os.path.dirname(cmd)
            if tdir:
                cur_path = os.environ.get("PATH", "")
                if tdir.lower() not in cur_path.lower():
                    os.environ["PATH"] = tdir + os.pathsep + cur_path

                # Python 3.8+ on Windows: help the process find DLLs.
                add_dll_dir = getattr(os, "add_dll_directory", None)
                if callable(add_dll_dir):
                    try:
                        add_dll_dir(tdir)
                    except Exception:
                        pass
        except Exception:
            pass

        try:
            pytesseract.pytesseract.tesseract_cmd = cmd  # type: ignore[attr-defined]
        except Exception:
            pass

    if not cmd:
        return False, "tesseract executable was not found (install Tesseract OCR and/or set TESSERACT_CMD)"

    ok, detail = _check_cmd(cmd)
    if ok:
        return True, ""

    try:
        _ = pytesseract.get_tesseract_version()  # type: ignore[attr-defined]
    except Exception as e:
        return False, f"{detail}. pytesseract detail: {e}"

    return False, detail


def _extract_text_lines_from_pdf_without_ocr(pdf_path: str) -> List[str]:
    lines: List[str] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                if text:
                    lines.extend(text.splitlines())
    except Exception:
        lines = []

    cleaned = [_clean_text(x) for x in lines]
    cleaned = [x for x in cleaned if x]
    if cleaned:
        return cleaned

    lines2: List[str] = []
    if pdfium is not None:
        try:
            doc = pdfium.PdfDocument(pdf_path)
            for i in range(len(doc)):
                page = doc[i]
                try:
                    textpage = page.get_textpage()
                    txt = textpage.get_text_range()
                except Exception:
                    txt = ""
                if txt:
                    lines2.extend(str(txt).splitlines())
        except Exception:
            lines2 = []

    cleaned2 = [_clean_text(x) for x in lines2]
    cleaned2 = [x for x in cleaned2 if x]
    if cleaned2:
        return cleaned2

    lines3: List[str] = []
    if fitz is not None:
        try:
            doc2 = fitz.open(pdf_path)  # type: ignore[union-attr]
            for page in doc2:
                try:
                    txt2 = page.get_text("text")
                except Exception:
                    txt2 = ""
                if txt2:
                    lines3.extend(str(txt2).splitlines())
        except Exception:
            lines3 = []

    cleaned3 = [_clean_text(x) for x in lines3]
    cleaned3 = [x for x in cleaned3 if x]
    return cleaned3


def _extract_text_lines_from_image_with_ocr(image_path: str) -> Tuple[List[str], bool]:
    ok, _detail = _tesseract_available()
    if not ok:
        return [], False
    if Image is None:
        return [], False

    def _score_ocr_text(txt: str) -> int:
        t = _clean_text(txt)
        if not t:
            return -1
        alnum = len(re.findall(r"[A-Za-z0-9]", t))
        lines_n = len([x for x in t.splitlines() if _clean_text(x)])
        words = len(re.findall(r"[A-Za-z0-9]{2,}", t))
        return alnum + (lines_n * 12) + (words * 3)

    def _preprocess_for_ocr(img: Any) -> Any:
        if ImageOps is None or ImageEnhance is None:
            return img
        try:
            g = ImageOps.grayscale(img)
            g = ImageOps.autocontrast(g)
            g = ImageEnhance.Contrast(g).enhance(2.0)
            g = g.point(lambda x: 0 if x < 170 else 255)
            return g
        except Exception:
            return img

    try:
        pil_img = Image.open(image_path)
    except Exception:
        return [], False

    cfgs = [
        "--oem 1 --psm 6 -c preserve_interword_spaces=1",
        "--oem 1 --psm 4 -c preserve_interword_spaces=1",
        "--oem 1 --psm 11 -c preserve_interword_spaces=1",
    ]
    best_txt = ""
    best_score = -1
    for angle in (0, 90, 180, 270):
        try:
            img2 = pil_img.rotate(angle, expand=True) if angle else pil_img
        except Exception:
            img2 = pil_img
        img2 = _preprocess_for_ocr(img2)
        for cfg in cfgs:
            try:
                txt = pytesseract.image_to_string(img2, config=cfg, lang="eng")  # type: ignore[union-attr]
            except Exception:
                continue
            sc = _score_ocr_text(txt)
            if sc > best_score:
                best_score = sc
                best_txt = txt

    cleaned = [_clean_text(x) for x in (best_txt.splitlines() if best_txt else [])]
    cleaned = [x for x in cleaned if x]
    return cleaned, bool(cleaned)


def _extract_text_lines_from_pdf_with_ocr(pdf_path: str, force_ocr: bool = False) -> Tuple[List[str], bool]:
    lines: List[str] = []
    if not force_ocr:
        cleaned = _extract_text_lines_from_pdf_without_ocr(pdf_path)
        if cleaned:
            return cleaned, False

    if not BANKPDF_OCR and not force_ocr:
        return [], False

    if not BANKPDF_OCR and force_ocr:
        return [], False

    ok, _detail = _tesseract_available()
    if not ok:
        return [], False
    if Image is None:
        return [], False
    if fitz is None and pdfium is None:
        return [], False
    
    # Simplified OCR with timeout
    try:
        return _extract_text_simplified(pdf_path)
    except Exception as e:
        logging.error(f"OCR processing failed for {pdf_path}: {e}")
        return [], False


def _extract_text_simplified(pdf_path: str) -> Tuple[List[str], bool]:
    """Robust OCR processing for files up to 5 minutes"""
    import threading
    
    result_container = {'lines': [], 'success': False}
    
    def run_ocr():
        try:
            ocr_lines: List[str] = []
            
            # Balanced preprocessing for quality and speed
            def _preprocess_for_ocr(img: Any) -> Any:
                if ImageOps is None or ImageEnhance is None:
                    return img
                try:
                    g = ImageOps.grayscale(img)
                    g = ImageOps.autocontrast(g)
                    g = ImageEnhance.Contrast(g).enhance(1.5)  # Moderate enhancement
                    return g
                except Exception:
                    return img
                    
            # Good quality OCR config
            cfg = "--oem 3 --psm 6 -c preserve_interword_spaces=1"
            
            try:
                if pdfium is not None:
                    doc = pdfium.PdfDocument(pdf_path)
                    # Process ALL pages for large documents (up to 72 pages)
                    max_pages = len(doc)
                    for i in range(max_pages):
                        page = doc[i]
                        # Good balance of quality and speed
                        bitmap = page.render(scale=2.5)
                        pil_img = bitmap.to_pil()  # type: ignore[union-attr]
                        img2 = _preprocess_for_ocr(pil_img)
                        try:
                            txt = pytesseract.image_to_string(img2, config=cfg, lang="eng")  # type: ignore[union-attr]
                            if txt and len(txt.strip()) > 5:
                                ocr_lines.extend(txt.splitlines())
                        except Exception:
                            continue
                else:
                    doc = fitz.open(pdf_path)  # type: ignore[union-attr]
                    # Process ALL pages for large documents (up to 72 pages)
                    max_pages = len(doc)
                    for page in doc[:max_pages]:
                        # Good balance of quality and speed
                        pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5))
                        img_bytes = pix.tobytes("png")
                        pil_img = Image.open(io.BytesIO(img_bytes))
                        img2 = _preprocess_for_ocr(pil_img)
                        try:
                            txt = pytesseract.image_to_string(img2, config=cfg, lang="eng")  # type: ignore[union-attr]
                            if txt and len(txt.strip()) > 5:
                                ocr_lines.extend(txt.splitlines())
                        except Exception:
                            continue
            except Exception:
                result_container['success'] = False
                return
                
            cleaned2 = [_clean_text(x) for x in ocr_lines]
            cleaned2 = [x for x in cleaned2 if x and len(x) > 2]
            result_container['lines'] = cleaned2
            result_container['success'] = bool(cleaned2)
            
        except Exception as e:
            logging.error(f"OCR thread error: {e}")
            result_container['success'] = False
    
    # Extended timeout for larger files (up to 5 minutes)
    thread = threading.Thread(target=run_ocr)
    thread.daemon = True
    thread.start()
    thread.join(timeout=300)  # 5 minutes timeout
    
    if thread.is_alive():
        logging.warning(f"OCR processing timed out for {pdf_path}")
        return [], False
    
    return result_container['lines'], result_container['success']


def _extract_account_from_lines(lines: List[str]) -> str:
    cleaned = [_clean_text(ln) for ln in lines]
    for ln in cleaned[:30]:
        m = re.search(r"\b(?:Account\s*No\.?|Account)\s*[:\-]?\s*([A-Z0-9\s\-]{4,20})", ln, flags=re.IGNORECASE)
        if m:
            return _clean_text(m.group(1))
    for ln in cleaned[:30]:
        m = re.search(r"\b([A-Z]{2}\d{2}\s?[A-Z]{4,8})\b", ln)
        if m:
            return _clean_text(m.group(1))
    return ""


def _looks_like_barclays_statement(lines: List[str]) -> bool:
    cleaned = [_clean_text(ln).lower() for ln in lines[:50]]
    joined = " ".join(cleaned)
    return any(k in joined for k in ["barclays", "barclaycard", "barclays bank"])


def _extract_barclays_header_info(lines: List[str]) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    cleaned = [_clean_text(ln) for ln in lines[:80]]
    for ln in cleaned:
        # Account number patterns
        m = re.search(r"\b(?:Account\s*No\.?|Account)\s*[:\-]?\s*([A-Z0-9\s\-]{4,20})", ln, flags=re.IGNORECASE)
        if m and not info.get("account"):
            info["account"] = _clean_text(m.group(1))
        
        # Sort code pattern
        m = re.search(r"\b(?:Sort\s*Code)\s*[:\-]?\s*([\d\s\-]{6,10})", ln, flags=re.IGNORECASE)
        if m and not info.get("sort_code"):
            info["sort_code"] = _clean_text(m.group(1)).replace(" ", "").replace("-", "")
        
        # Statement date pattern
        m = re.search(r"\b(?:Statement\s*Date|Period|Issued\s*on)\s*[:\-]?\s*(\d{1,2}\s+\w+\s+\d{4})", ln, flags=re.IGNORECASE)
        if m and not info.get("statement_date"):
            info["statement_date"] = _clean_text(m.group(1))
        
        # Company name pattern
        m = re.search(r"([A-Z\s&]+(?:STORE|LTD|LIMITED|COMPANY|CORP))", ln, flags=re.IGNORECASE)
        if m and not info.get("company_name"):
            company = _clean_text(m.group(1))
            if len(company) > 3 and not any(skip in company.lower() for skip in ['account', 'sort', 'code']):
                info["company_name"] = company
        
        # IBAN pattern
        m = re.search(r"\b(?:IBAN)\s*[:\-]?\s*([A-Z0-9\s]{15,34})", ln, flags=re.IGNORECASE)
        if m and not info.get("iban"):
            info["iban"] = _clean_text(m.group(1))
        
        # SWIFT/BIC pattern
        m = re.search(r"\b(?:SWIFT|BIC|SWIFTBIC)\s*[:\-]?\s*([A-Z]{6,})", ln, flags=re.IGNORECASE)
        if m and not info.get("swift"):
            info["swift"] = _clean_text(m.group(1))
    
    return info


def _barclays_header_preamble_lines(info: Dict[str, Any]) -> List[List[str]]:
    rows = []
    if info.get("account"):
        rows.append(["Account Number", info["account"]])
    if info.get("statement_date"):
        rows.append(["Statement Date", info["statement_date"]])
    return rows


def _looks_like_barclays_business_premium_statement(lines: List[str]) -> bool:
    cleaned = [_clean_text(ln).lower() for ln in lines[:50]]
    joined = " ".join(cleaned)
    return "business premium" in joined and "barclays" in joined


def _extract_barclays_business_premium_header_info(lines: List[str]) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    cleaned = [_clean_text(ln) for ln in lines[:80]]
    for ln in cleaned:
        m = re.search(r"\b(?:Account\s*No\.?|Account)\s*[:\-]?\s*([A-Z0-9\s\-]{4,20})", ln, flags=re.IGNORECASE)
        if m and not info.get("account"):
            info["account"] = _clean_text(m.group(1))
        m = re.search(r"\b(?:Statement\s*Date|Period)\s*[:\-]?\s*(\d{1,2}\s+\w+\s+\d{4})", ln, flags=re.IGNORECASE)
        if m and not info.get("statement_date"):
            info["statement_date"] = _clean_text(m.group(1))
    return info


def _barclays_business_premium_preamble_lines(info: Dict[str, Any]) -> List[List[str]]:
    rows = []
    if info.get("account"):
        rows.append(["Account Number", info["account"]])
    if info.get("statement_date"):
        rows.append(["Statement Date", info["statement_date"]])
    return rows


def _looks_like_monzo_statement(lines: List[str]) -> bool:
    cleaned = [_clean_text(ln).lower() for ln in lines[:50]]
    joined = " ".join(cleaned)
    return "monzo" in joined


def _extract_monzo_header_info(lines: List[str]) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    cleaned = [_clean_text(ln) for ln in lines[:80]]
    for ln in cleaned:
        m = re.search(r"\b(?:Account\s*No\.?|Account)\s*[:\-]?\s*([A-Z0-9\s\-]{4,20})", ln, flags=re.IGNORECASE)
        if m and not info.get("account"):
            info["account"] = _clean_text(m.group(1))
        m = re.search(r"\b(?:Statement\s*Date|Period)\s*[:\-]?\s*(\d{1,2}\s+\w+\s+\d{4})", ln, flags=re.IGNORECASE)
        if m and not info.get("statement_date"):
            info["statement_date"] = _clean_text(m.group(1))
    return info


def _monzo_header_preamble_lines(info: Dict[str, Any]) -> List[List[str]]:
    rows = []
    if info.get("account"):
        rows.append(["Account Number", info["account"]])
    if info.get("statement_date"):
        rows.append(["Statement Date", info["statement_date"]])
    return rows


def _looks_like_virgin_money_statement(lines: List[str]) -> bool:
    cleaned = [_clean_text(ln).lower() for ln in lines[:50]]
    joined = " ".join(cleaned)
    return "virgin money" in joined


def _extract_virgin_money_header_info(lines: List[str]) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    cleaned = [_clean_text(ln) for ln in lines[:80]]
    for ln in cleaned:
        m = re.search(r"\b(?:Account\s*No\.?|Account)\s*[:\-]?\s*([A-Z0-9\s\-]{4,20})", ln, flags=re.IGNORECASE)
        if m and not info.get("account"):
            info["account"] = _clean_text(m.group(1))
        m = re.search(r"\b(?:Statement\s*Date|Period)\s*[:\-]?\s*(\d{1,2}\s+\w+\s+\d{4})", ln, flags=re.IGNORECASE)
        if m and not info.get("statement_date"):
            info["statement_date"] = _clean_text(m.group(1))
    return info


def _virgin_money_header_preamble_lines(info: Dict[str, Any]) -> List[List[str]]:
    rows = []
    if info.get("account"):
        rows.append(["Account Number", info["account"]])
    if info.get("statement_date"):
        rows.append(["Statement Date", info["statement_date"]])
    return rows


def _looks_like_tide_statement(lines: List[str]) -> bool:
    cleaned = [_clean_text(ln).lower() for ln in lines[:50]]
    joined = " ".join(cleaned)
    return "tide" in joined


def _extract_tide_header_info(lines: List[str]) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    cleaned = [_clean_text(ln) for ln in lines[:80]]
    for ln in cleaned:
        m = re.search(r"\b(?:Account\s*No\.?|Account)\s*[:\-]?\s*([A-Z0-9\s\-]{4,20})", ln, flags=re.IGNORECASE)
        if m and not info.get("account"):
            info["account"] = _clean_text(m.group(1))
        m = re.search(r"\b(?:Statement\s*Date|Period)\s*[:\-]?\s*(\d{1,2}\s+\w+\s+\d{4})", ln, flags=re.IGNORECASE)
        if m and not info.get("statement_date"):
            info["statement_date"] = _clean_text(m.group(1))
    return info


def _tide_header_preamble_lines(info: Dict[str, Any]) -> List[List[str]]:
    rows = []
    if info.get("account"):
        rows.append(["Account Number", info["account"]])
    if info.get("statement_date"):
        rows.append(["Statement Date", info["statement_date"]])
    return rows


def _looks_like_revolut_business_statement(lines: List[str]) -> bool:
    cleaned = [_clean_text(ln).lower() for ln in lines[:50]]
    joined = " ".join(cleaned)
    return "revolut" in joined and "business" in joined


def _extract_revolut_business_header_info(lines: List[str]) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    cleaned = [_clean_text(ln) for ln in lines[:80]]
    for ln in cleaned:
        m = re.search(r"\b(?:Account\s*No\.?|Account)\s*[:\-]?\s*([A-Z0-9\s\-]{4,20})", ln, flags=re.IGNORECASE)
        if m and not info.get("account"):
            info["account"] = _clean_text(m.group(1))
        m = re.search(r"\b(?:Statement\s*Date|Period)\s*[:\-]?\s*(\d{1,2}\s+\w+\s+\d{4})", ln, flags=re.IGNORECASE)
        if m and not info.get("statement_date"):
            info["statement_date"] = _clean_text(m.group(1))
    return info


def _revolut_business_preamble_lines(info: Dict[str, Any]) -> List[List[str]]:
    rows = []
    if info.get("account"):
        rows.append(["Account Number", info["account"]])
    if info.get("statement_date"):
        rows.append(["Statement Date", info["statement_date"]])
    return rows


def _infer_subcategory(description: str, amount: Any, money_in: Any, money_out: Any) -> str:
    desc = _clean_text(description).lower()
    if any(k in desc for k in ["salary", "payroll", "wages"]):
        return "Income"
    if any(k in desc for k in ["rent", "council tax", "utilities", "gas", "electric", "water"]):
        return "Bills"
    if any(k in desc for k in ["grocery", "supermarket", "tesco", "sainsbury", "asda"]):
        return "Groceries"
    if any(k in desc for k in ["cash", "atm", "withdrawal"]):
        return "Cash"
    if any(k in desc for k in ["transfer", "payment", "direct debit"]):
        return "Transfer"
    return "Other"


def _write_csv_with_preamble(csv_path: str, preamble: List[List[str]], rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in preamble:
            writer.writerow(row)
        writer.writerow([])
        writer.writerow(fieldnames)
        dict_writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        for r in rows:
            dict_writer.writerow({k: _format_csv_value(r.get(k)) for k in fieldnames})


def convert_pdf_to_rows(pdf_path: str, preextracted_lines: Optional[List[str]] = None, used_ocr_hint: bool = False) -> List[Dict[str, Any]]:
    if preextracted_lines is None:
        lines, _ = _extract_text_lines_from_pdf_with_ocr(pdf_path)
    else:
        lines = preextracted_lines

    cleaned = [_clean_text(ln) for ln in lines]
    cleaned = [ln for ln in cleaned if ln]

    # Enhanced date patterns for Barclays statements
    date_patterns = [
        re.compile(r"^(\d{1,2}\s+\w{3,9}\s+\d{2,4})\b", flags=re.IGNORECASE),  # 01 Sep 2025
        re.compile(r"^(\d{1,2}/\d{1,2}/\d{2,4})\b"),  # 01/09/2025
        re.compile(r"^(\d{1,2}-\d{1,2}-\d{2,4})\b"),  # 01-09-2025
        re.compile(r"^(\d{1,2}\.\d{1,2}\.\d{2,4})\b"),  # 01.09.25
        re.compile(r"\b(\d{1,2}\s+\w{3,9}\s+\d{2,4})\b", flags=re.IGNORECASE),  # Date anywhere in line
    ]

    def _find_date_in_line(line: str) -> str:
        for pat in date_patterns:
            m = pat.search(line)
            if m:
                return _clean_text(m.group(1))
        return ""

    def _normalize_amount_token(tok: str) -> str:
        s = _clean_text(tok)
        s = s.replace("GBP", "").replace("gbp", "").replace("£", "")
        s = s.replace(",", "").replace(" ", "")
        return s

    def _extract_barclays_transactions(lines: List[str]) -> List[Dict[str, Any]]:
        """Special parser for Barclays statements"""
        rows: List[Dict[str, Any]] = []

        # Barclays table statements often use dd/mm/yyyy.
        barclays_date_patterns = [
            re.compile(r"\b(\d{1,2}/\d{1,2}/\d{2,4})\b"),
            re.compile(r"\b(\d{1,2}-\d{1,2}-\d{2,4})\b"),
            re.compile(r"\b(\d{1,2}\.\d{1,2}\.\d{2,4})\b"),
            re.compile(r"\b(\d{1,2}\s+\w{3,9}\s+\d{2,4})\b", flags=re.IGNORECASE),
        ]

        def _find_barclays_date(line: str) -> str:
            for pat in barclays_date_patterns:
                m = pat.search(line)
                if m:
                    return _clean_text(m.group(1))
            return ""

        date_any_re = re.compile(
            r"(\d{1,2}/\d{1,2}/\d{2,4}|\d{1,2}-\d{1,2}-\d{2,4}|\d{1,2}\.\d{1,2}\.\d{2,4}|\d{1,2}\s+\w{3,9}\s+\d{2,4})",
            flags=re.IGNORECASE,
        )
        
        current_date = ""
        current_parts: List[str] = []

        def _emit_row(date_str: str, blob: str) -> None:
            b = _clean_text(blob)
            if not b:
                return

            low = b.lower()
            if any(k in low for k in [
                'card number', 'available balance', "last night's balance", 'overdraft',
                'showing', 'transactions between', 'pending debit card transactions',
                'date description money in money out balance', 'date transaction amount',
            ]):
                return

            amounts = [m.group(0) for m in CURRENCY_RE.finditer(b)]
            if not amounts:
                return

            if len(amounts) >= 2:
                txn_amount_raw = amounts[-2]
                bal_amount_raw = amounts[-1]
            else:
                txn_amount_raw = amounts[-1]
                bal_amount_raw = ""

            txn_clean = _normalize_amount_token(txn_amount_raw)
            txn_val = _to_float_or_none(txn_clean)

            money_in = ""
            money_out = ""
            if txn_val is not None:
                if txn_val >= 0:
                    money_in = txn_clean
                else:
                    money_out = txn_clean.replace("-", "")
            else:
                if "(" in txn_amount_raw and ")" in txn_amount_raw:
                    money_out = txn_clean.replace("(", "").replace(")", "")
                else:
                    money_in = txn_clean

            balance = _normalize_amount_token(bal_amount_raw) if bal_amount_raw else ""

            cut_pos = b.rfind(txn_amount_raw)
            desc_blob = b[:cut_pos] if cut_pos > 0 else b
            if desc_blob.lower().startswith(date_str.lower()):
                desc_blob = desc_blob[len(date_str):]
            description = _clean_text(desc_blob)
            if description and (money_in or money_out):
                rows.append(
                    {
                        "date": date_str,
                        "description": description,
                        "money_in": money_in,
                        "money_out": money_out,
                        "balance": balance,
                        "amount": txn_val,
                        "used_ocr": bool(used_ocr_hint),
                    }
                )

        def _flush_current() -> None:
            nonlocal current_date, current_parts
            if not current_date or not current_parts:
                current_date = ""
                current_parts = []
                return

            joined = " ".join([p for p in current_parts if p]).strip()
            if not joined:
                current_date = ""
                current_parts = []
                return

            matches = list(date_any_re.finditer(joined))
            if len(matches) >= 2:
                for idx, m in enumerate(matches):
                    seg_start = m.start()
                    seg_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(joined)
                    seg = joined[seg_start:seg_end]
                    seg_date = _clean_text(m.group(1))
                    _emit_row(seg_date, seg)
            else:
                _emit_row(current_date, joined)

            current_date = ""
            current_parts = []

        for _i, line in enumerate(lines):
            date_str = _find_barclays_date(line)
            low = line.lower()

            if date_str:
                _flush_current()
                current_date = date_str
                current_parts = [line]
                continue

            if not current_date:
                continue

            if any(k in low for k in ['page', 'barclays', 'account', 'sort code', 'swift', 'iban', 'issued']):
                continue

            current_parts.append(line)

        _flush_current()
        
        return rows

    # First try Barclays-specific parsing
    if any('barclays' in line.lower() for line in cleaned[:20]):
        barclays_rows = _extract_barclays_transactions(cleaned)
        if barclays_rows:
            return barclays_rows

    # Fallback to original parsing logic
    rows: List[Dict[str, Any]] = []
    for ln in cleaned:
        date_str = _find_date_in_line(ln)
        if not date_str:
            continue

        amounts_raw = [m.group(0) for m in CURRENCY_RE.finditer(ln)]
        if len(amounts_raw) < 1:  # Reduced requirement from 2 to 1
            continue

        amounts = [_normalize_amount_token(a) for a in amounts_raw]
        
        money_in = ""
        money_out = ""
        amount_val: Optional[float] = None
        balance = ""

        if len(amounts) >= 2:
            a_txn, a_bal = amounts[0], amounts[-1]
            txn_val = _to_float_or_none(a_txn)
            money_in = a_txn if (txn_val is not None and txn_val > 0) else ""
            money_out = a_txn if (txn_val is not None and txn_val < 0) else (a_txn if not money_in else "")
            balance = a_bal
            amount_val = txn_val
        else:
            # Single amount case
            a_txn = amounts[0]
            txn_val = _to_float_or_none(a_txn)
            money_in = a_txn if (txn_val is not None and txn_val > 0) else ""
            money_out = a_txn if (txn_val is not None and txn_val < 0) else (a_txn if not money_in else "")
            balance = ""
            amount_val = txn_val
            
        # Extract description
        first_amt_match = CURRENCY_RE.search(ln)
        desc_part = ""
        if first_amt_match:
            desc_part = ln[: first_amt_match.start()]
        if desc_part.lower().startswith(date_str.lower()):
            desc_part = desc_part[len(date_str) :]
        description = _clean_text(desc_part)
        
        if not description:
            continue

        rows.append(
            {
                "date": date_str,
                "description": description,
                "money_in": money_in,
                "money_out": money_out,
                "balance": balance,
                "amount": amount_val,
                "used_ocr": bool(used_ocr_hint),
            }
        )

    return rows


def _to_float_or_none(value: Any) -> Optional[float]:

    s = _clean_text(value)

    if not s:

        return None

    try:

        return _parse_money(s)

    except Exception:

        return None



def _extract_invoice_fields(lines: List[str]) -> Dict[str, Any]:

    cleaned = [_clean_text(ln) for ln in lines]

    cleaned = [ln for ln in cleaned if ln]

    joined = "\n".join(cleaned)

    joined_low = joined.lower()



    def _format_uk_reg(value: str) -> str:

        s = _clean_text(value).upper().strip()

        if not s:

            return ""

        s2 = re.sub(r"\s+", "", s)

        if len(s2) == 7 and re.match(r"^[A-Z]{2}[0-9O]{2}[A-Z]{3}$", s2):

            s2_fixed = s2[:2] + s2[2:4].replace("O", "0") + s2[4:]

            return s2_fixed[:4] + " " + s2_fixed[4:]

        return s



    def _find_first(patterns: List[str]) -> str:

        for pat in patterns:

            m = re.search(pat, joined, flags=re.IGNORECASE | re.MULTILINE)

            if m:

                return _clean_text(m.group(1))

        return ""



    def _find_amount_after_phrase(phrases: List[str]) -> Optional[float]:

        for i, ln in enumerate(cleaned[:800]):

            low = ln.lower()

            if not any(p in low for p in phrases):

                continue

            m = CURRENCY_RE.search(ln)

            if m:

                return _to_float_or_none(m.group(0))

            for j in range(1, 4):

                if i + j >= len(cleaned):

                    break

                nxt = cleaned[i + j]

                m2 = CURRENCY_RE.search(nxt)

                if m2:

                    return _to_float_or_none(m2.group(0))

        return None



    if "used vehicle purchase invoice" in joined_low or "vehicle purchase invoice" in joined_low:

        date_re = re.compile(r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})")

        amount_re = re.compile(r"(?:£|\$|€)?\s*\(?-?\d[\d,]*(?:\.\d{1,2})?\)?")

        reg_hint_re = re.compile(r"\b[A-Z]{2}[0-9O]{2}\s*[A-Z]{3}\b", flags=re.IGNORECASE)



        def _next_nonempty(i: int, max_ahead: int = 4) -> str:

            for j in range(1, max_ahead + 1):

                if i + j >= len(cleaned):

                    break

                v = _clean_text(cleaned[i + j])

                if v:

                    return v

            return ""



        def _value_after_label(i: int, label_re: "re.Pattern[str]") -> str:

            ln = cleaned[i]

            m = label_re.search(ln)

            if not m:

                return ""

            tail = _clean_text(ln[m.end() :]).strip(" :-\t")

            if tail:

                return tail

            return _next_nonempty(i)



        def _strip_noise(v: str) -> str:

            s = _clean_text(v)

            s = s.replace("·", " ")

            s = re.sub(r"[._]{2,}", " ", s)

            s = re.sub(r"\s{2,}", " ", s)

            s = re.split(r"\(name\)|\(address\)|vat\s*reg|vat\s*no\.?", s, flags=re.IGNORECASE)[0]

            return _clean_text(s)



        def _pick_make(v: str) -> str:

            s = _strip_noise(v)

            tokens = re.findall(r"[A-Za-z]{2,}", s)

            stop = {"MODEL", "TYPE", "OR", "COLOUR", "COLOR", "MAKE"}

            for t in tokens:

                if t.upper() in stop:

                    continue

                return _clean_text(t)

            m = re.search(r"([A-Za-z]{2,}(?:\s+[A-Za-z]{2,}){0,2})", s)

            return _clean_text(m.group(1)) if m else s



        def _pick_nameish(v: str) -> str:

            s = _strip_noise(v)

            tokens = re.findall(r"[A-Za-z]{2,}", s)

            stop = {"SOLD", "BY", "BOUGHT", "BUYER", "NAME", "ADDRESS"}

            for t in tokens:

                if t.upper() in stop:

                    continue

                return _clean_text(t)

            return s



        def _parse_amount_any(v: str) -> Optional[float]:

            s = _clean_text(v)

            if not s:

                return None

            m = amount_re.search(s)

            if not m:

                return None

            raw = _clean_text(m.group(0))

            raw = raw.replace("£", "").replace("$", "").replace("€", "")

            raw = raw.replace(",", "").replace(" ", "")

            raw = raw.strip("()")

            try:

                return float(raw)

            except Exception:

                return None



        document_date = ""

        sold_by = ""

        make = ""

        reg_no = ""

        buying_price: Optional[float] = None



        sold_by_re = re.compile(r"\bSold\s*by\b\s*[:\-]?", flags=re.IGNORECASE)

        make_re = re.compile(r"\bMake\b\s*[:\-]?", flags=re.IGNORECASE)

        reg_re = re.compile(r"\bRegistration\s*No\.?\b\s*[:\-]?", flags=re.IGNORECASE)

        date_label_re = re.compile(r"\bDate\b\s*[:\-]?", flags=re.IGNORECASE)

        price_label_re = re.compile(r"\bThis\s*price\s*is\b|\bThis\s*price\b", flags=re.IGNORECASE)



        for i, ln in enumerate(cleaned[:140]):

            low = ln.lower()

            if not sold_by and sold_by_re.search(ln):

                sold_by = _pick_nameish(_value_after_label(i, sold_by_re))

            if not make and make_re.search(ln) and "vehicle" not in low:

                make = _pick_make(_value_after_label(i, make_re))

            if not reg_no and reg_re.search(ln):

                reg_no = _format_uk_reg(_value_after_label(i, reg_re))

            if not document_date and date_label_re.search(ln) and "first registered" not in low:

                v = _value_after_label(i, date_label_re)

                m = date_re.search(v) or date_re.search(ln)

                if m:

                    document_date = _clean_text(m.group(1))

            if buying_price is None and price_label_re.search(ln):

                v = _value_after_label(i, price_label_re)

                buying_price = _parse_amount_any(v) or _parse_amount_any(_next_nonempty(i)) or _parse_amount_any(ln)



        if not document_date:

            for ln in cleaned[:80]:

                m = date_re.search(ln)

                if m:

                    document_date = _clean_text(m.group(1))

                    break



        if not reg_no:

            for ln in cleaned[:220]:

                m = reg_hint_re.search(ln)

                if m:

                    reg_no = _format_uk_reg(m.group(0))

                    break



        if buying_price is None:

            candidates: List[float] = []

            for ln in cleaned:

                v = _parse_amount_any(ln)

                if v is None:

                    continue

                if 1900 <= v <= 2100:

                    continue

                if v < 50:

                    continue

                candidates.append(v)

            if candidates:

                buying_price = max(candidates)



        supplier = sold_by[:120] if sold_by else ""

        non_vat = buying_price



        return {

            "document_date": document_date,

            "supplier": supplier,

            "inv_ref_no": "",

            "make": make,

            "reg_no": reg_no,

            "buying_price": buying_price,

            "non_vat": non_vat,

            "std_net": "N/A",

            "vat_amount": "N/A",

        }



    # BCA invoices often contain many other dates (late payment, storage, VAT/day lines).

    # Prefer the explicitly-labeled "Document date" and the explicitly-labeled "Total due".

    if ("british car auctions" in joined_low or " bca" in joined_low or "bca" in joined_low) and "document date" in joined_low:

        document_date = _find_first(

            [

                r"\bDocument\s*date\b\s*[:\-]?\s*(\d{1,2}/\d{1,2}/\d{2,4})",

                r"\bDocument\s*date\b\s*(\d{1,2}/\d{1,2}/\d{2,4})",

            ]

        )



        supplier = _find_first(

            [

                r"^(BRITISH\s+CAR\s+AUCTIONS\s+LIMITED)\b.*$",

                r"^(BRITISH\s+CAR\s+AUCTIONS)\b.*$",

            ]

        )

        if supplier:

            supplier = supplier[:120]



        inv_ref = _find_first(

            [

                r"\bINVOICE\b\s*([A-Z]{1,5}\d{4,})\b",

                r"\bINVOICE\b\s*([A-Za-z0-9\-/]+)",

            ]

        )



        reg_no = ""

        make = ""



        vat_reg = _find_first(

            [

                r"\bV\.?A\.?T\.?\s*Registration\s*Number\b\s*[:\-]?\s*(GB\s*[0-9 ]{7,})\b",

                r"\bVAT\s*Registration\s*Number\b\s*[:\-]?\s*(GB\s*[0-9 ]{7,})\b",

            ]

        )

        vat_reg = _clean_text(vat_reg).upper()

        vat_reg = re.sub(r"\s{2,}", " ", vat_reg).strip()

        if vat_reg.startswith("GB"):

            reg_no = vat_reg



        reg_pat = re.compile(r"\b([A-Z]{2}\d{2}\s?[A-Z]{3})\b")

        candidate_lines = []

        for ln in cleaned:

            if "/" in ln and reg_pat.search(ln):

                candidate_lines.append(ln)

        if not candidate_lines:

            for ln in cleaned:

                if reg_pat.search(ln):

                    candidate_lines.append(ln)

                    break



        if candidate_lines:

            item_desc = candidate_lines[0]

            m = reg_pat.search(item_desc)

            if m:

                reg_raw = _clean_text(m.group(1)).upper().replace(" ", "")

                vehicle_reg = ""

                if len(reg_raw) == 7:

                    vehicle_reg = reg_raw[:4] + " " + reg_raw[4:]

                else:

                    vehicle_reg = _clean_text(m.group(1)).upper().strip()

                if not reg_no:

                    reg_no = vehicle_reg



                tail = _clean_text(item_desc[m.end() :]).strip()

                if tail:

                    # Make is usually the first token after the reg on BCA invoices.

                    mk = re.search(r"\b([A-Za-z]{2,})\b", tail)

                    if mk:

                        make = _clean_text(mk.group(1))



        # For BCA, place the full ITEM DESCRIPTION block into the "make" field.

        # This helps avoid blank make and captures the full vehicle/charge description.

        item_desc_lines: List[str] = []

        for i, ln in enumerate(cleaned[:700]):

            if "item description" in ln.lower():

                for j in range(i + 1, min(len(cleaned), i + 20)):

                    nxt = _clean_text(cleaned[j])

                    if not nxt:

                        continue

                    low = nxt.lower()

                    if any(

                        p in low

                        for p in (

                            "account card",

                            "late payment",

                            "storage charge",

                            "essential check",

                            "buyers fee",

                            "margin",

                            "price",

                            "vat%",

                            "vat %",

                            "receipt",

                        )

                    ):

                        break

                    item_desc_lines.append(nxt)

                    if len(item_desc_lines) >= 8:

                        break

                break

        if item_desc_lines:

            make = _clean_text(" | ".join(item_desc_lines))[:600]



        if not reg_no:

            reg_no = _find_first([r"\b(?:Reg(?:istration)?\s*(?:No\.?|Number)?|VRM)\b\s*[:\-]?\s*([A-Z0-9\- ]{5,12})\b"])

            reg_no = _format_uk_reg(reg_no)



        if not make:

            make = _find_first([r"\bMake\b\s*[:\-]?\s*([A-Za-z0-9 &\-]+)$", r"\bVehicle\s+Make\b\s*[:\-]?\s*([A-Za-z0-9 &\-]+)$"])



        buying_price = _find_amount_after_phrase(["total due"])

        if buying_price is None:

            buying_price = _find_amount_after_phrase(["total"])  # fallback



        non_vat = _find_amount_after_phrase(["non vat", "non-vat", "nonvat"])

        if non_vat is None:

            non_vat = buying_price



        std_net_val = _find_amount_after_phrase(["std net", "standard net", "std. net"])

        std_net: Any = std_net_val if std_net_val is not None else "N/A"



        vat_amount = _find_amount_after_phrase(["vat amount"])

        if vat_amount is None and "vat" not in joined_low:

            vat_amount = None



        return {

            "document_date": document_date,

            "supplier": supplier,

            "inv_ref_no": inv_ref,

            "make": make,

            "reg_no": reg_no,

            "buying_price": buying_price,

            "non_vat": non_vat,

            "std_net": std_net,

            "vat_amount": vat_amount,

        }



    document_date = _find_first(

        [

            r"\bDocument\s*date\b\s*[:\-]?\s*(\d{1,2}/\d{1,2}/\d{2,4})",

            r"\bDocument\s*date\b\s*(\d{1,2}/\d{1,2}/\d{2,4})",

        ]

    )



    supplier = _find_first(

        [

            r"^(BRITISH\s+CAR\s+AUCTIONS\s+LIMITED)\b.*$",

            r"^(BRITISH\s+CAR\s+AUCTIONS)\b.*$",

            r"\bSupplier\b\s*[:\-]?\s*(.+)$",

            r"\bFrom\b\s*[:\-]?\s*(.+)$",

            r"\bSeller\b\s*[:\-]?\s*(.+)$",

        ]

    )

    if supplier:

        supplier = supplier[:120]



    inv_ref = _find_first(

        [

            r"\bINVOICE\b\s*([A-Z]{1,5}/\d{4,})\b",

            r"\bINVOICE\b\s*([A-Za-z0-9\-/]+)",

            r"\b(?:Invoice|Inv)\s*(?:No\.?|Number|#)\b\s*[:\-]?\s*([A-Za-z0-9\-/]+)",

            r"\b(?:Reference|Ref)\s*(?:No\.?|Number|#)?\b\s*[:\-]?\s*([A-Za-z0-9\-/]+)",

            r"\bINV/REF\s*NO\b\s*[:\-]?\s*([A-Za-z0-9\-/]+)",

        ]

    )



    reg_no = ""

    make = ""

    item_desc = ""



    reg_pat = re.compile(r"\b([A-Z]{2}\d{2}\s?[A-Z]{3})\b")

    candidate_lines = []

    for ln in cleaned:

        if "/" in ln and reg_pat.search(ln):

            candidate_lines.append(ln)

    if not candidate_lines:

        for ln in cleaned:

            if reg_pat.search(ln):

                candidate_lines.append(ln)

                break



    if candidate_lines:

        item_desc = candidate_lines[0]

        m = reg_pat.search(item_desc)

        if m:

            reg_raw = _clean_text(m.group(1)).upper().replace(" ", "")

            if len(reg_raw) == 7:

                reg_no = reg_raw[:4] + " " + reg_raw[4:]

            else:

                reg_no = _clean_text(m.group(1)).upper().strip()



            tail = _clean_text(item_desc[m.end() :]).strip()

            if not tail:

                make = item_desc

            else:

                cut_markers = [

                    "ODOMETER",

                    "WARRANTED",

                    "1ST REG",

                    "MOT",

                    "L/B-",

                    "S/H-",

                    "CH:",

                ]

                tail2 = tail

                tail_low = tail2.lower()

                for mk in cut_markers:

                    pos = tail_low.find(mk.lower())

                    if pos != -1:

                        tail2 = tail2[:pos].strip()

                        break

                make = tail2 if tail2 else tail



    if not reg_no:

        reg_no = _find_first(

            [

                r"\b(?:Reg(?:istration)?\s*(?:No\.?|Number)?|VRM)\b\s*[:\-]?\s*([A-Z0-9\- ]{5,12})\b",

            ]

        )

        reg_no = _format_uk_reg(reg_no)

        if reg_no:

            reg_no = reg_no



    if not make:

        make = _find_first(

            [

                r"\bMake\b\s*[:\-]?\s*([A-Za-z0-9 &\-]+)$",

                r"\bVehicle\s+Make\b\s*[:\-]?\s*([A-Za-z0-9 &\-]+)$",

            ]

        )



    def _find_amount_after_labels(labels: List[str]) -> Optional[float]:

        currency_re = CURRENCY_RE

        for i, ln in enumerate(cleaned[:600]):

            low = ln.lower()

            if not any(lbl in low for lbl in labels):

                continue

            m = currency_re.search(ln)

            if m:

                return _to_float_or_none(m.group(0))

            for j in range(1, 3):

                if i + j >= len(cleaned):

                    break

                nxt = cleaned[i + j]

                m2 = currency_re.search(nxt)

                if m2:

                    return _to_float_or_none(m2.group(0))

        return None



    def _find_last_amount_in_line(line: str) -> Optional[float]:

        last = None

        for m in CURRENCY_RE.finditer(line):

            v = _to_float_or_none(m.group(0))

            if v is not None:

                last = v

        return last



    buying_price = _find_amount_after_labels(["total due", "total"])

    if buying_price is None:

        for ln in reversed(cleaned[-120:]):

            low = ln.lower()

            if "total due" in low or (low.strip().startswith("total") and "vat registration" not in low):

                buying_price = _find_last_amount_in_line(ln)

                if buying_price is not None:

                    break



    non_vat = _find_amount_after_labels(["non vat", "non-vat", "nonvat"])

    if non_vat is None:

        non_vat = buying_price



    std_net_val = _find_amount_after_labels(["std net", "standard net", "std. net"])

    std_net: Any = std_net_val if std_net_val is not None else "N/A"



    vat_amount = _find_amount_after_labels(["vat amount"])

    if vat_amount is None:

        for ln in cleaned[:600]:

            low = ln.lower()

            if "vat" not in low:

                continue

            if "vat registration" in low or "registration number" in low:

                continue

            m = re.search(r"\bVAT\b\s*[:\-]?\s*(?:£|\$|€)?\s*\(?-?\d[\d,]*\.\d{2}\)?", ln, flags=re.IGNORECASE)

            if m:

                vat_amount = _to_float_or_none(m.group(0))

                break



    if vat_amount is None and "vat" not in joined_low:

        vat_amount = None



    return {

        "document_date": document_date,

        "supplier": supplier,

        "inv_ref_no": inv_ref,

        "make": make,

        "reg_no": reg_no,

        "buying_price": buying_price,

        "non_vat": non_vat,

        "std_net": std_net,

        "vat_amount": vat_amount,

    }



def _clean_text(value: Any) -> str:

    if value is None:

        return ""



    s = str(value)

    s = s.replace("\u00a0", " ")

    s = s.replace("\r", "\n")

    s = re.sub(r"[ \t]+", " ", s)

    s = re.sub(r"\n{2,}", "\n", s)

    return s.strip()





def _is_valid_uk_date(value: Any) -> bool:

    s = _clean_text(value)

    if not s:

        return False

    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{2,4})$", s)

    if not m:

        return False

    try:

        d = int(m.group(1))

        mo = int(m.group(2))

        y = int(m.group(3))

    except Exception:

        return False

    if d < 1 or d > 31:

        return False

    if mo < 1 or mo > 12:

        return False

    if y < 0:

        return False

    return True



def _parse_money(value: str) -> float:

    s = _clean_text(value)

    if not s:

        raise ValueError("Empty amount")



    neg = False

    if s.startswith("(") and s.endswith(")"):

        neg = True

        s = s[1:-1]



    s = s.replace("£", "").replace("$", "").replace("€", "")

    s = s.replace(",", "").replace(" ", "")

    if s.endswith("-"):

        neg = True

        s = s[:-1]



    amt = float(s)

    return -amt if neg else amt



def _format_csv_value(value: Any) -> str:

    if value is None:

        return ""

    if isinstance(value, float):

        return f"{value:.2f}"

    return _clean_text(value)



def _write_csv(csv_path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:

    with open(csv_path, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")

        writer.writeheader()

        for r in rows:

            writer.writerow({k: _format_csv_value(r.get(k)) for k in fieldnames})





def _write_json(json_path: str, data: Any) -> None:

    with open(json_path, "w", encoding="utf-8") as f:

        json.dump(data, f, ensure_ascii=False, indent=2)





def _read_json(json_path: str) -> Any:

    with open(json_path, "r", encoding="utf-8") as f:

        return json.load(f)



def _invoice_tesseract_available() -> Tuple[bool, str]:

    return _tesseract_available()



def _invoice_extract_text_lines_from_pdf_with_ocr(pdf_path: str, force_ocr: bool = False) -> Tuple[List[str], bool]:

    lines: List[str] = []

    if not force_ocr:
        cleaned = _extract_text_lines_from_pdf_without_ocr(pdf_path)
        if cleaned:
            return cleaned, False

    if not BANKPDF_OCR and not force_ocr:
        return [], False

    if not BANKPDF_OCR and force_ocr:
        return [], False



    ok, _detail = _invoice_tesseract_available()

    if not ok:

        return [], False

    if Image is None:

        return [], False

    if fitz is None and pdfium is None:

        return [], False



    ocr_lines: List[str] = []



    def _score_ocr_text(txt: str) -> int:

        t = _clean_text(txt)

        if not t:

            return -1

        alnum = len(re.findall(r"[A-Za-z0-9]", t))

        lines_n = len([x for x in t.splitlines() if _clean_text(x)])

        words = len(re.findall(r"[A-Za-z0-9]{2,}", t))

        return alnum + (lines_n * 12) + (words * 3)



    def _preprocess_for_ocr(img: Any) -> Any:

        if ImageOps is None or ImageEnhance is None:

            return img

        try:

            g = ImageOps.grayscale(img)

            g = ImageOps.autocontrast(g)

            g = ImageEnhance.Contrast(g).enhance(2.0)

            g = g.point(lambda x: 0 if x < 170 else 255)

            return g

        except Exception:

            return img



    tesseract_cfg = "--oem 1 --psm 6 -c preserve_interword_spaces=1"

    try:

        if pdfium is not None:

            doc = pdfium.PdfDocument(pdf_path)

            for i in range(len(doc)):

                page = doc[i]

                bitmap = page.render(scale=3)

                pil_img = bitmap.to_pil()  # type: ignore[union-attr]

                best_txt = ""

                best_score = -1

                for angle in (0, 90, 180, 270):

                    try:

                        img2 = pil_img.rotate(angle, expand=True) if angle else pil_img

                    except Exception:

                        img2 = pil_img

                    img2 = _preprocess_for_ocr(img2)

                    try:

                        txt = pytesseract.image_to_string(img2, config=tesseract_cfg)  # type: ignore[union-attr]

                    except Exception:

                        continue

                    sc = _score_ocr_text(txt)

                    if sc > best_score:

                        best_score = sc

                        best_txt = txt

                if best_txt:

                    ocr_lines.extend(best_txt.splitlines())

        else:

            doc = fitz.open(pdf_path)  # type: ignore[union-attr]

            for page in doc:

                pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))  # type: ignore[union-attr]

                img_bytes = pix.tobytes("png")

                pil_img = Image.open(io.BytesIO(img_bytes))

                best_txt = ""

                best_score = -1

                for angle in (0, 90, 180, 270):

                    try:

                        img2 = pil_img.rotate(angle, expand=True) if angle else pil_img

                    except Exception:

                        img2 = pil_img

                    img2 = _preprocess_for_ocr(img2)

                    try:

                        txt = pytesseract.image_to_string(img2, config=tesseract_cfg)  # type: ignore[union-attr]

                    except Exception:

                        continue

                    sc = _score_ocr_text(txt)

                    if sc > best_score:

                        best_score = sc

                        best_txt = txt

                if best_txt:

                    ocr_lines.extend(best_txt.splitlines())

    except Exception:

        return [], False



    cleaned2 = [_clean_text(x) for x in ocr_lines]

    cleaned2 = [x for x in cleaned2 if x]

    return cleaned2, bool(cleaned2)



def _invoice_render_first_page(pdf_path: str) -> Optional[Any]:

    if Image is None:

        return None

    try:

        if pdfium is not None:

            doc = pdfium.PdfDocument(pdf_path)

            if len(doc) < 1:

                return None

            page = doc[0]

            bitmap = page.render(scale=5)

            return bitmap.to_pil()  # type: ignore[union-attr]

        if fitz is not None:

            doc = fitz.open(pdf_path)  # type: ignore[union-attr]

            if doc.page_count < 1:  # type: ignore[union-attr]

                return None

            page = doc.load_page(0)  # type: ignore[union-attr]

            pix = page.get_pixmap(matrix=fitz.Matrix(5, 5))  # type: ignore[union-attr]

            img_bytes = pix.tobytes("png")

            return Image.open(io.BytesIO(img_bytes))

        return None

    except Exception:

        return None



def _invoice_preprocess_crop(img: Any) -> Any:

    if ImageOps is None or ImageEnhance is None:

        return img

    try:

        g = ImageOps.grayscale(img)

        g = ImageOps.autocontrast(g)

        g = ImageEnhance.Contrast(g).enhance(1.9)

        g = ImageEnhance.Sharpness(g).enhance(1.4)

        return g

    except Exception:

        return img





def _invoice_ocr_bca_fields(pdf_path: str) -> Dict[str, Any]:

    out: Dict[str, Any] = {}

    ok, _detail = _invoice_tesseract_available()

    if not ok or pytesseract is None or TesseractOutput is None:

        return out

    img = _invoice_render_first_page(pdf_path)

    if img is None:

        return out



    def _score_orientation(txt: str) -> int:

        t = _clean_text(txt).lower()

        if not t:

            return -1

        score = 0

        if "invoice" in t:

            score += 4

        if "document" in t and "date" in t:

            score += 6

        if "total" in t and "due" in t:

            score += 6

        if "bca" in t or "british car auctions" in t:

            score += 6

        score += len(re.findall(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", t)) * 2

        score += len(re.findall(r"\b[A-Z]{1,3}\d{4,}\b", t.upper())) * 2

        return score



    best_img = img

    best_score = -1

    for angle in (0, 90, 180, 270):

        try:

            img2 = img.rotate(angle, expand=True) if angle else img

        except Exception:

            img2 = img

        try:

            base = _invoice_preprocess_crop(img2)

            txt = pytesseract.image_to_string(base, config="--oem 1 --psm 6")  # type: ignore[union-attr]

        except Exception:

            continue

        sc = _score_orientation(txt)

        if sc > best_score:

            best_score = sc

            best_img = img2

        if best_score >= 18:

            break



    try:

        base = _invoice_preprocess_crop(best_img)

        data = pytesseract.image_to_data(base, output_type=TesseractOutput.DICT, config="--oem 1 --psm 6")  # type: ignore[union-attr]

    except Exception:

        return out



    n = len(data.get("text", []) or [])

    if n < 1:

        return out



    def _norm_token(s: str) -> str:

        s2 = _clean_text(s).lower()

        s2 = re.sub(r"[^a-z0-9]+", "", s2)

        return s2



    def _tok(i: int) -> str:

        try:

            return _norm_token((data.get("text", [""])[i] or ""))

        except Exception:

            return ""



    def _box(i: int) -> Tuple[int, int, int, int]:

        x = int(data.get("left", [0])[i])

        y = int(data.get("top", [0])[i])

        w0 = int(data.get("width", [0])[i])

        h0 = int(data.get("height", [0])[i])

        return x, y, w0, h0



    def _find_phrase(words: List[str], max_gap: int = 3) -> Optional[Tuple[int, int, int, int]]:

        want = [_norm_token(w) for w in words]

        want = [w for w in want if w]

        if not want:

            return None

        for start in range(n):

            if _tok(start) != want[0]:

                continue

            idxs = [start]

            cur = start

            okp = True

            for wi in range(1, len(want)):

                found = None

                for k in range(cur + 1, min(n, cur + max_gap + 2)):

                    if _tok(k) == want[wi]:

                        found = k

                        break

                if found is None:

                    okp = False

                    break

                idxs.append(found)

                cur = found

            if not okp:

                continue

            xs, ys, xe, ye = [], [], [], []

            for j in idxs:

                x, y, w0, h0 = _box(j)

                xs.append(x)

                ys.append(y)

                xe.append(x + w0)

                ye.append(y + h0)

            return min(xs), min(ys), max(xe) - min(xs), max(ye) - min(ys)

        return None



    def _ocr_near(label_box: Tuple[int, int, int, int], mode: str, cfg: str) -> str:

        x, y, w0, h0 = label_box

        W, H = base.size

        pad = int(max(10, h0 * 0.35))

        if mode == "right":

            x1 = min(W - 1, x + w0 + pad)

            y1 = max(0, y - pad)

            x2 = min(W, x + w0 + int(W * 0.35))

            y2 = min(H, y + h0 + pad)

        elif mode == "below":

            x1 = max(0, x - pad)

            y1 = min(H - 1, y + h0 + pad)

            x2 = min(W, x + int(W * 0.45))

            y2 = min(H, y + h0 + int(H * 0.12))

        else:

            x1, y1, x2, y2 = 0, 0, W, H

        crop = base.crop((x1, y1, x2, y2))

        crop = crop.resize((crop.size[0] * 2, crop.size[1] * 2))

        crop = _invoice_preprocess_handwriting(crop)

        try:

            txt = pytesseract.image_to_string(crop, config=cfg)  # type: ignore[union-attr]

        except Exception:

            return ""

        return _clean_text(txt)



    inv_box = _find_phrase(["invoice"], max_gap=6)

    doc_date_box = _find_phrase(["document", "date"], max_gap=6)

    total_due_box = _find_phrase(["total", "due"], max_gap=6)



    if inv_box:

        t = _ocr_near(inv_box, "right", "--oem 1 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

        m = re.search(r"\b([A-Z]{1,3}\d{4,})\b", t.upper())

        if m:

            out["inv_ref_no"] = _clean_text(m.group(1))



    if doc_date_box:

        t = _ocr_near(doc_date_box, "right", "--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789/ ")

        m = re.search(r"\b(\d{1,2}/\d{1,2}/\d{2,4})\b", t)

        if m:

            out["document_date"] = _clean_text(m.group(1))



    if total_due_box:

        t = _ocr_near(total_due_box, "right", "--oem 1 --psm 7 -c tessedit_char_whitelist=£0123456789.,")

        v = _to_float_or_none(t)

        if v is None:

            m = re.search(r"(?:£)?\s*(\d[\d,]*\.\d{2})", t)

            if m:

                v = _to_float_or_none(m.group(0))

        if v is not None:

            out["buying_price"] = v

            out["non_vat"] = v



    # Layout fallback (fixed ROIs) for noisy scans where label words are not detected.

    # BCA template places invoice no + document date near the top-right and total due near bottom-right.

    def _ocr_roi(rel_box: Tuple[float, float, float, float], cfg: str) -> str:

        W, H = base.size

        x1 = int(max(0, min(W - 1, rel_box[0] * W)))

        y1 = int(max(0, min(H - 1, rel_box[1] * H)))

        x2 = int(max(1, min(W, rel_box[2] * W)))

        y2 = int(max(1, min(H, rel_box[3] * H)))

        crop = base.crop((x1, y1, x2, y2))

        crop = crop.resize((crop.size[0] * 2, crop.size[1] * 2))

        crop = _invoice_preprocess_handwriting(crop)

        try:

            return _clean_text(pytesseract.image_to_string(crop, config=cfg))  # type: ignore[union-attr]

        except Exception:

            return ""



    if not _clean_text(out.get("inv_ref_no")):

        for roi in [(0.60, 0.02, 0.99, 0.18), (0.55, 0.00, 0.99, 0.22)]:

            t = _ocr_roi(roi, "--oem 1 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

            m = re.search(r"\b([A-Z]{1,3}\d{4,})\b", t.upper())

            if m:

                out["inv_ref_no"] = _clean_text(m.group(1))

                break



    if not _is_valid_uk_date(out.get("document_date")):

        for roi in [(0.60, 0.06, 0.99, 0.22), (0.55, 0.02, 0.99, 0.26)]:

            t = _ocr_roi(roi, "--oem 1 --psm 6 -c tessedit_char_whitelist=0123456789/ ")

            m = re.search(r"\b(\d{1,2}/\d{1,2}/\d{2,4})\b", t)

            if m and _is_valid_uk_date(m.group(1)):

                out["document_date"] = _clean_text(m.group(1))

                break



    if out.get("buying_price") in (None, ""):

        for roi in [(0.60, 0.78, 0.99, 0.98), (0.55, 0.72, 0.99, 0.98)]:

            t = _ocr_roi(roi, "--oem 1 --psm 6 -c tessedit_char_whitelist=£0123456789.,")

            m = re.search(r"(?:£)?\s*(\d[\d,]*\.\d{2})", t)

            if m:

                v = _to_float_or_none(m.group(0))

                if v is not None and 0 < v < 100000:

                    out["buying_price"] = v

                    out["non_vat"] = v

                    break



    # Broad fallback on best orientation for reg/make and supplier.

    try:

        txt_full = pytesseract.image_to_string(base, config="--oem 1 --psm 6")  # type: ignore[union-attr]

    except Exception:

        txt_full = ""

    txt_low = _clean_text(txt_full).lower()

    if "british car auctions" in txt_low or "bca" in txt_low:

        out["supplier"] = "BRITISH CAR AUCTIONS LIMITED"



    if not _clean_text(out.get("reg_no")):

        vat_txt = _ocr_roi((0.52, 0.10, 0.99, 0.22), "--oem 1 --psm 6 -c tessedit_char_whitelist=GB0123456789 ")

        mvat = re.search(r"\bGB\s*[0-9 ]{7,}\b", vat_txt.upper())

        if not mvat:

            mvat = re.search(r"\bGB\s*[0-9 ]{7,}\b", txt_full.upper())

        if mvat:

            vat = _clean_text(mvat.group(0).upper())

            vat = re.sub(r"\s{2,}", " ", vat).strip()

            if vat.startswith("GB"):

                out["reg_no"] = vat



    if not _clean_text(out.get("reg_no")):

        mreg = re.search(r"\b([A-Z]{2}[0-9O]{2}\s*[A-Z]{3})\b", txt_full.upper())

        if mreg:

            raw = mreg.group(1).replace(" ", "")

            raw = raw[:2] + raw[2:4].replace("O", "0") + raw[4:]

            out["reg_no"] = raw[:4] + " " + raw[4:]



    # For BCA, use the full ITEM DESCRIPTION block as the "make" field.

    desc_txt = _ocr_roi((0.05, 0.26, 0.72, 0.42), "--oem 1 --psm 6")

    if desc_txt:

        desc_lines: List[str] = []

        for ln in desc_txt.splitlines():

            t = _clean_text(ln)

            if not t:

                continue

            if "item description" in t.lower():

                continue

            desc_lines.append(t)

            if len(desc_lines) >= 8:

                break

        if desc_lines:

            out["make"] = _clean_text(" | ".join(desc_lines))[:600]



    if not _clean_text(out.get("make")):

        m2 = re.search(r"\b[A-Z]{2}\d{2}\s?[A-Z]{3}\b\s+([A-Z]{2,})", txt_full.upper())

        if m2:

            out["make"] = _clean_text(m2.group(1).title())



    return out



def _invoice_preprocess_handwriting(img: Any) -> Any:

    if ImageOps is None or ImageEnhance is None:

        return img

    try:

        g = ImageOps.grayscale(img)

        g = ImageOps.autocontrast(g)

        g = ImageEnhance.Contrast(g).enhance(1.7)

        g = ImageEnhance.Sharpness(g).enhance(1.2)

        return g

    except Exception:

        return img



def _invoice_preprocess_handwriting_strong(img: Any) -> Any:

    if ImageOps is None or ImageEnhance is None or ImageFilter is None:

        return _invoice_preprocess_handwriting(img)

    try:

        g = ImageOps.grayscale(img)

        g = ImageOps.autocontrast(g)

        g = ImageEnhance.Contrast(g).enhance(2.4)

        g = ImageEnhance.Sharpness(g).enhance(1.6)



        hist = g.histogram()

        total = float(sum(hist))

        sum_total = 0.0

        for i, h in enumerate(hist):

            sum_total += float(i * h)



        sum_b = 0.0

        w_b = 0.0

        var_max = -1.0

        threshold = 140

        for t in range(256):

            w_b += float(hist[t])

            if w_b <= 0.0:

                continue

            w_f = total - w_b

            if w_f <= 0.0:

                break

            sum_b += float(t * hist[t])

            m_b = sum_b / w_b

            m_f = (sum_total - sum_b) / w_f

            var_between = w_b * w_f * (m_b - m_f) ** 2

            if var_between > var_max:

                var_max = var_between

                threshold = t



        bw = g.point(lambda x, th=threshold: 255 if x > th else 0)

        bw = bw.filter(ImageFilter.MinFilter(3))

        return bw

    except Exception:

        return _invoice_preprocess_handwriting(img)



def _invoice_remove_red_print(img: Any) -> Any:

    if Image is None:

        return img

    try:

        rgb = img.convert("RGB")

        if ImageChops is None:

            return rgb

        r, g, b = rgb.split()

        gb = ImageChops.lighter(g, b)

        red_dom = ImageChops.subtract(r, gb)

        red_dom = ImageOps.autocontrast(red_dom) if ImageOps is not None else red_dom

        mask = red_dom.point(lambda x: 255 if x > 60 else 0)

        white = Image.new("RGB", rgb.size, (255, 255, 255))

        return Image.composite(white, rgb, mask)

    except Exception:

        return img



def _invoice_preprocess_for_label_detection(img: Any) -> Any:

    if ImageOps is None or ImageEnhance is None:

        return img

    try:

        # Printed labels are red in this invoice template.

        # Use a red-dominance mask: R - max(G,B), then invert to get black text on white.

        try:

            if ImageChops is not None:

                rgb = img.convert("RGB")

                r, g, b = rgb.split()

                gb = ImageChops.lighter(g, b)

                red_dom = ImageChops.subtract(r, gb)

                red_dom = ImageOps.autocontrast(red_dom)

                red_dom = red_dom.point(lambda x: 0 if x < 60 else 255)

                inv = ImageOps.invert(red_dom)

                inv = ImageEnhance.Contrast(inv).enhance(2.6)

                inv = ImageEnhance.Sharpness(inv).enhance(1.6)

                return inv

        except Exception:

            pass



        g2 = ImageOps.grayscale(img)

        g2 = ImageOps.autocontrast(g2)

        g2 = ImageEnhance.Contrast(g2).enhance(1.6)

        g2 = ImageEnhance.Sharpness(g2).enhance(1.2)

        return g2

    except Exception:

        return img



def _invoice_ocr_used_vehicle_purchase_fields(pdf_path: str) -> Dict[str, Any]:

    out: Dict[str, Any] = {}

    ok, _detail = _invoice_tesseract_available()

    if not ok or pytesseract is None:

        return out

    img = _invoice_render_first_page(pdf_path)

    if img is None:

        return out



    dbg_dir: Optional[str] = None

    if DEBUG:

        try:

            os.makedirs(OUTPUT_DIR, exist_ok=True)

            dbg_dir = os.path.join(OUTPUT_DIR, "_debug_invoice_crops", str(uuid.uuid4()))

            os.makedirs(dbg_dir, exist_ok=True)

        except Exception:

            dbg_dir = None



    def _dbg_save(im: Any, name: str) -> None:

        if not dbg_dir or Image is None:

            return

        try:

            p = os.path.join(dbg_dir, name)

            im2 = im

            try:

                im2 = im.convert("RGB")

            except Exception:

                pass

            im2.save(p)

        except Exception:

            return



    def _auto_crop_to_red_border(img_in: Any) -> Any:

        if Image is None:

            return img_in

        try:

            rgb = img_in.convert("RGB")

            w, h = rgb.size

            # Identify red-ish pixels (invoice border/lines) and crop to their bounding box.

            # This normalizes scans where the page is shifted/cropped.

            px = rgb.load()

            minx, miny, maxx, maxy = w, h, 0, 0

            found = 0

            step = max(1, int(min(w, h) / 700))

            for y in range(0, h, step):

                for x in range(0, w, step):

                    r, g, b = px[x, y]

                    if r > 160 and g < 140 and b < 140 and (r - max(g, b)) > 40:

                        found += 1

                        if x < minx:

                            minx = x

                        if y < miny:

                            miny = y

                        if x > maxx:

                            maxx = x

                        if y > maxy:

                            maxy = y

            if found < 50:

                return img_in

            pad = int(min(w, h) * 0.02)

            x1 = max(0, minx - pad)

            y1 = max(0, miny - pad)

            x2 = min(w, maxx + pad)

            y2 = min(h, maxy + pad)

            if x2 - x1 < int(w * 0.4) or y2 - y1 < int(h * 0.4):

                return img_in

            return img_in.crop((x1, y1, x2, y2))

        except Exception:

            return img_in



    boxes = {

        "document_date": (0.72, 0.13, 0.95, 0.19),

        "supplier": (0.13, 0.155, 0.55, 0.205),

        "make": (0.20, 0.295, 0.55, 0.345),

        "reg_no": (0.76, 0.37, 0.95, 0.44),

        "buying_price": (0.70, 0.545, 0.92, 0.62),

    }



    def _normalize_supplier(s: str) -> str:

        s2 = re.sub(r"[^A-Za-z0-9 &\-]", " ", s)

        s2 = re.sub(r"\s{2,}", " ", s2)

        s2 = _clean_text(s2)

        bad = {"BOUGHT", "BY", "SOLD", "DATE", "INVOICE", "PURCHASE", "USED", "VEHICLE", "NAME", "ADDRESS"}

        parts = [p for p in re.split(r"\s+", s2) if p]

        parts2 = [p for p in parts if p.upper() not in bad]

        return _clean_text(" ".join(parts2))



    def _normalize_make(s: str) -> str:

        s2 = re.sub(r"[^A-Za-z0-9 &\-]", " ", s)

        s2 = re.sub(r"\s{2,}", " ", s2)

        s2 = _clean_text(s2)

        bad = {"MODEL", "OR", "TYPE", "COLOUR", "COLOR", "MAKE"}

        tokens = [t for t in re.findall(r"[A-Za-z]{2,}", s2) if t.upper() not in bad]

        return tokens[0] if tokens else s2



    def _pick_invoice_price(value: str) -> Optional[float]:

        s = _clean_text(value)

        if not s:

            return None

        s2 = s.replace(",", "")



        # Capture numbers with optional decimals.

        matches = list(re.finditer(r"(?:(?:£)\s*)?(\d{2,5}(?:\.\d{1,2})?)", s2))

        if not matches:

            return None



        candidates: List[Tuple[int, float]] = []

        for m in matches:

            raw = m.group(1)

            try:

                v = float(raw)

            except Exception:

                continue

            score = 0

            # Prefer £-adjacent numbers.

            span_start = max(0, m.start() - 2)

            if "£" in s2[span_start : m.start()] or "£" in s2[m.start() : m.end()]:

                score += 8

            # Prefer typical invoice price ranges (your case ~795).

            if 100 <= v <= 2000:

                score += 8

            elif 50 <= v <= 5000:

                score += 2

            else:

                score -= 4

            # Prefer 3-4 digit values.

            if 100 <= v < 10000:

                score += 2

            # Penalize obviously wrong very large values.

            if v >= 3000:

                score -= 12

            if v > 2500:

                score -= 14

            # If multiple candidates, prefer one closer to 795 (template expectation).

            score -= int(min(3000, abs(v - 795))) // 200

            candidates.append((score, v))



        if not candidates:

            return None

        candidates.sort(key=lambda x: x[0], reverse=True)

        best_score, best_val = candidates[0]

        if best_score < 6:

            return None

        return best_val



    def _ocr_best(crop: Any, configs: List[str], kind: str) -> str:

        best_txt = ""

        best_score = -1

        for cfg in configs:

            try:

                txt = pytesseract.image_to_string(crop, config=cfg)  # type: ignore[union-attr]

            except Exception:

                continue

            t = _clean_text(txt)

            if not t:

                continue

            score = 0

            if kind == "date":

                if re.search(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", t):

                    score += 10

                score += min(4, len(t))

            elif kind == "reg":

                u = re.sub(r"[^A-Z0-9 ]", " ", t.upper())

                u = _clean_text(u)

                if re.search(r"\b[A-Z]{2}[0-9O]{2}\s*[A-Z]{3}\b", u):

                    score += 10

                score += min(4, len(u))

            elif kind == "price":

                v = _pick_invoice_price(t)

                if v is not None:

                    score += 10

                    if 100 <= v <= 2000:

                        score += 4

                    if v >= 3000:

                        score -= 6

                score += min(4, len(t))

            else:

                score += min(6, len(t))

            if score > best_score:

                best_score = score

                best_txt = t

            if best_score >= 12:

                break

        return best_txt



    def _score_candidate(kind: str, t: str) -> int:

        s = _clean_text(t)

        if not s:

            return -1

        score = 0

        if kind == "date":

            if re.search(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", s):

                score += 20

        elif kind == "reg":

            u = re.sub(r"[^A-Z0-9 ]", " ", s.upper())

            u = _clean_text(u)

            if re.search(r"\b[A-Z]{2}[0-9O]{2}\s*[A-Z]{3}\b", u):

                score += 20

        elif kind == "price":

            v = _pick_invoice_price(s)

            if v is not None:

                score += 20

        score += len(re.findall(r"[A-Za-z0-9]", s))

        score += min(60, len(s))

        return score



    def _ocr_boxes_from_image(img2: Any) -> Dict[str, Any]:

        w, h = img2.size



        def _ocr_rel(lx: float, ty: float, rx: float, by: float, cfgs: Tuple[str, ...]) -> str:

            crop = img2.crop((int(lx * w), int(ty * h), int(rx * w), int(by * h)))

            crop = crop.resize((crop.size[0] * 2, crop.size[1] * 2))

            crop = _invoice_remove_red_print(crop)

            variants: List[Any] = []

            try:

                variants.append(_invoice_preprocess_handwriting(crop))

            except Exception:

                pass

            try:

                variants.append(_invoice_preprocess_handwriting_strong(crop))

            except Exception:

                pass

            best_txt = ""

            best_sc = -1

            for v in variants or [crop]:

                for cfg in cfgs:

                    try:

                        txt = pytesseract.image_to_string(v, config=cfg)  # type: ignore[union-attr]

                    except Exception:

                        continue

                    t2 = _clean_text(txt)

                    sc = _score_candidate("text", t2)

                    if sc > best_sc:

                        best_sc = sc

                        best_txt = t2

            return _clean_text(best_txt)



        def _ocr_box(name: str, cfg: str) -> str:

            l, t, r, b = boxes[name]

            crop = img2.crop((int(l * w), int(t * h), int(r * w), int(b * h)))

            crop = crop.resize((crop.size[0] * 2, crop.size[1] * 2))

            crop = _invoice_remove_red_print(crop)

            variants = []

            try:

                variants.append(_invoice_preprocess_handwriting(crop))

            except Exception:

                pass

            try:

                variants.append(_invoice_preprocess_handwriting_strong(crop))

            except Exception:

                pass

            best_txt = ""

            best_sc = -1

            for v in variants or [crop]:

                try:

                    txt = pytesseract.image_to_string(v, config=cfg, timeout=3)  # type: ignore[union-attr]

                except Exception:

                    continue

                t2 = _clean_text(txt)

                sc = _score_candidate("text", t2)

                if sc > best_sc:

                    best_sc = sc

                    best_txt = t2

            return _clean_text(best_txt)



        out2: Dict[str, Any] = {}



        date_rois = [

            boxes["document_date"],

            (0.62, 0.10, 0.98, 0.22),

            (0.55, 0.08, 0.99, 0.24),

        ]

        found_date = ""

        for di, (lx, ty, rx, by) in enumerate(date_rois):

            date_crop = img2.crop((int(lx * w), int(ty * h), int(rx * w), int(by * h)))

            date_crop = date_crop.resize((date_crop.size[0] * 2, date_crop.size[1] * 2))

            _dbg_save(date_crop, f"date_{di}_0_raw.png")

            date_crop = _invoice_remove_red_print(date_crop)

            _dbg_save(date_crop, f"date_{di}_1_no_red.png")

            date_variants = []

            try:

                date_variants.append(_invoice_preprocess_handwriting(date_crop))

            except Exception:

                pass

            try:

                date_variants.append(_invoice_preprocess_handwriting_strong(date_crop))

            except Exception:

                pass

            if date_variants:

                _dbg_save(date_variants[0], f"date_{di}_2_pre.png")

            best_date_txt = ""

            best_date_sc = -1

            for dv in date_variants or [date_crop]:

                date_txt = _ocr_best(

                    dv,

                [

                    "--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789/-. ",

                    "--oem 1 --psm 13 -c tessedit_char_whitelist=0123456789/-. ",

                    "--oem 1 --psm 6 -c tessedit_char_whitelist=0123456789/-. ",

                ],

                "date",

                )

                sc = _score_candidate("date", date_txt)

                if sc > best_date_sc:

                    best_date_sc = sc

                    best_date_txt = date_txt

            m = re.search(r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", best_date_txt)

            if m:

                cand_date = _clean_text(m.group(1))

                if _is_valid_uk_date(cand_date):

                    found_date = cand_date

                    break

        if found_date:

            out2["document_date"] = found_date



        # Sold-by is handwritten on 2 lines (name + address). OCR them with tighter ROIs.

        supplier_cfgs = (

            "--oem 1 --psm 7 -c preserve_interword_spaces=1",

            "--oem 1 --psm 6 -c preserve_interword_spaces=1",

            "--oem 1 --psm 11 -c preserve_interword_spaces=1",

            "--oem 1 --psm 12 -c preserve_interword_spaces=1",

        )

        sold_name = _ocr_rel(0.13, 0.155, 0.55, 0.205, supplier_cfgs)

        sold_addr = _ocr_rel(0.13, 0.205, 0.55, 0.255, supplier_cfgs)

        supplier_raw = _clean_text((sold_name + " " + sold_addr).strip())

        if not supplier_raw:

            supplier_raw = _ocr_box("supplier", supplier_cfgs[0])

        supplier_txt = _normalize_supplier(supplier_raw)

        if supplier_txt and len(supplier_txt) >= 3:

            out2["supplier"] = supplier_txt[:120]



        make_best = ""

        make_best_sc = -1

        for cfg in (

            "--oem 1 --psm 7 -c preserve_interword_spaces=1",

            "--oem 1 --psm 6 -c preserve_interword_spaces=1",

            "--oem 1 --psm 11 -c preserve_interword_spaces=1",

            "--oem 1 --psm 12 -c preserve_interword_spaces=1",

        ):

            t0 = _ocr_box("make", cfg)

            sc0 = _score_candidate("text", t0)

            if sc0 > make_best_sc:

                make_best_sc = sc0

                make_best = t0

        make_txt = _normalize_make(make_best)

        if make_txt and len(make_txt) >= 3:

            out2["make"] = make_txt



        reg_rois = [

            boxes["reg_no"],

            (0.74, 0.38, 0.95, 0.46),

            (0.74, 0.63, 0.95, 0.70),

        ]

        best_plate = ""

        for ri, (lx, ty, rx, by) in enumerate(reg_rois):

            reg_crop = img2.crop((int(lx * w), int(ty * h), int(rx * w), int(by * h)))

            reg_crop = reg_crop.resize((reg_crop.size[0] * 4, reg_crop.size[1] * 4))

            _dbg_save(reg_crop, f"reg_{ri}_0_raw.png")

            reg_crop = _invoice_remove_red_print(reg_crop)

            _dbg_save(reg_crop, f"reg_{ri}_1_no_red.png")

            reg_variants = []

            try:

                reg_variants.append(_invoice_preprocess_handwriting(reg_crop))

            except Exception:

                pass

            try:

                reg_variants.append(_invoice_preprocess_handwriting_strong(reg_crop))

            except Exception:

                pass

            best_reg_txt = ""

            best_reg_sc = -1

            for rv in reg_variants or [reg_crop]:

                reg_txt = _ocr_best(

                    rv,

                [

                    "--oem 1 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ",

                    "--oem 1 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ",

                    "--oem 1 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ",

                ],

                "reg",

                ).upper()

                sc = _score_candidate("reg", reg_txt)

                if sc > best_reg_sc:

                    best_reg_sc = sc

                    best_reg_txt = reg_txt

            reg_txt = best_reg_txt.upper()

            reg_txt = re.sub(r"[^A-Z0-9 ]", " ", reg_txt)

            reg_txt = _clean_text(reg_txt)

            m2 = re.search(r"\b([A-Z]{2}[0-9O]{2}\s*[A-Z]{3})\b", reg_txt)

            if m2:

                raw = m2.group(1).replace(" ", "")

                raw = raw[:2] + raw[2:4].replace("O", "0") + raw[4:]

                best_plate = raw[:4] + " " + raw[4:]

                break

        if best_plate:

            out2["reg_no"] = best_plate



        price_rois = [

            boxes["buying_price"],

            (0.66, 0.545, 0.96, 0.625),

        ]

        best_price_txt = ""

        best_price_sc = -1

        for pi, (lx, ty, rx, by) in enumerate(price_rois):

            price_crop = img2.crop((int(lx * w), int(ty * h), int(rx * w), int(by * h)))

            price_crop = price_crop.resize((price_crop.size[0] * 4, price_crop.size[1] * 4))

            _dbg_save(price_crop, f"price_{pi}_0_raw.png")

            price_crop = _invoice_remove_red_print(price_crop)

            _dbg_save(price_crop, f"price_{pi}_1_no_red.png")

            price_variants = []

            try:

                price_variants.append(_invoice_preprocess_handwriting(price_crop))

            except Exception:

                pass

            try:

                price_variants.append(_invoice_preprocess_handwriting_strong(price_crop))

            except Exception:

                pass

            for pv in price_variants or [price_crop]:

                price_txt = _ocr_best(

                    pv,

                [

                    "--oem 1 --psm 8 -c tessedit_char_whitelist=£0123456789., -c classify_bln_numeric_mode=1",

                    "--oem 1 --psm 7 -c tessedit_char_whitelist=£0123456789., -c classify_bln_numeric_mode=1",

                    "--oem 1 --psm 13 -c tessedit_char_whitelist=£0123456789., -c classify_bln_numeric_mode=1",

                    "--oem 1 --psm 6 -c tessedit_char_whitelist=£0123456789., -c classify_bln_numeric_mode=1",

                ],

                "price",

                )

                sc = _score_candidate("price", price_txt)

                if sc > best_price_sc:

                    best_price_sc = sc

                    best_price_txt = price_txt

        val = _pick_invoice_price(best_price_txt)

        if val is not None:

            out2["buying_price"] = val

            out2["non_vat"] = val



        return out2



    def _label_based_from_image(img2: Any) -> Dict[str, Any]:

        if TesseractOutput is None or pytesseract is None:

            return {}

        try:

            base = _invoice_preprocess_for_label_detection(img2)

            data = pytesseract.image_to_data(

                base,

                output_type=TesseractOutput.DICT,

                config="--oem 1 --psm 6",

                timeout=3,

            )  # type: ignore[union-attr]

        except Exception:

            return {}



        n = len(data.get("text", []) or [])

        if n < 1:

            return {}



        def _tok(i: int) -> str:

            try:

                return _clean_text((data["text"][i] or "")).lower()

            except Exception:

                return ""



        def _tok_norm(i: int) -> str:

            t = _tok(i)

            t = re.sub(r"[^a-z0-9]+", "", t)

            return t



        def _box(i: int) -> Tuple[int, int, int, int]:

            x = int(data.get("left", [0])[i])

            y = int(data.get("top", [0])[i])

            w0 = int(data.get("width", [0])[i])

            h0 = int(data.get("height", [0])[i])

            return x, y, w0, h0



        def _find_phrase(words: List[str], max_gap: int = 2) -> Optional[Tuple[int, int, int, int]]:

            want = [re.sub(r"[^a-z0-9]+", "", w.lower()) for w in words]

            want = [w for w in want if w]

            if not want:

                return None



            # Fuzzy in-order match: allow punctuation differences and small token gaps.

            for start in range(n):

                if _tok_norm(start) != want[0]:

                    continue

                idxs = [start]

                cur = start

                okp = True

                for wi in range(1, len(want)):

                    found = None

                    for k in range(cur + 1, min(n, cur + max_gap + 2)):

                        if _tok_norm(k) == want[wi]:

                            found = k

                            break

                    if found is None:

                        okp = False

                        break

                    idxs.append(found)

                    cur = found

                if not okp:

                    continue



                xs, ys, xe, ye = [], [], [], []

                for j in idxs:

                    x, y, w0, h0 = _box(j)

                    xs.append(x)

                    ys.append(y)

                    xe.append(x + w0)

                    ye.append(y + h0)

                return min(xs), min(ys), max(xe) - min(xs), max(ye) - min(ys)

            return None



        def _ocr_near(label_box: Tuple[int, int, int, int], mode: str, cfg: str) -> str:

            x, y, w0, h0 = label_box

            W, H = img2.size

            pad = int(max(8, h0 * 0.2))

            if mode == "right":

                x1 = min(W - 1, x + w0 + pad)

                y1 = max(0, y - pad)

                x2 = min(W, x + w0 + int(W * 0.35))

                y2 = min(H, y + h0 + pad)

            elif mode == "right_wide":

                x1 = min(W - 1, x + w0 + pad)

                y1 = max(0, y - int(h0 * 0.6))

                x2 = min(W, x + w0 + int(W * 0.55))

                y2 = min(H, y + int(h0 * 1.6))

            elif mode == "above_right":

                x1 = min(W - 1, x + w0 + pad)

                y1 = max(0, y - int(h0 * 1.6))

                x2 = min(W, x + w0 + int(W * 0.55))

                y2 = min(H, y + int(h0 * 1.2))

            else:

                x1, y1, x2, y2 = 0, 0, W, H

            crop = img2.crop((x1, y1, x2, y2))

            crop = crop.resize((crop.size[0] * 2, crop.size[1] * 2))

            crop = _invoice_remove_red_print(crop)

            variants = []

            try:

                variants.append(_invoice_preprocess_handwriting(crop))

            except Exception:

                pass

            try:

                variants.append(_invoice_preprocess_handwriting_strong(crop))

            except Exception:

                pass

            best_txt = ""

            best_sc = -1

            for v in variants or [crop]:

                try:

                    txt = pytesseract.image_to_string(v, config=cfg)  # type: ignore[union-attr]

                except Exception:

                    continue

                t2 = _clean_text(txt)

                sc = _score_candidate("text", t2)

                if sc > best_sc:

                    best_sc = sc

                    best_txt = t2

            return _clean_text(best_txt)



        res: Dict[str, Any] = {}

        date_box = _find_phrase(["date"]) or _find_phrase(["date"], max_gap=4)

        sold_box = _find_phrase(["sold", "by"], max_gap=4)

        make_box = _find_phrase(["make"], max_gap=4)

        reg_box = (

            _find_phrase(["registration", "no"], max_gap=5)

            or _find_phrase(["registration", "number"], max_gap=5)

            or _find_phrase(["reg", "no"], max_gap=5)

        )

        price_box = _find_phrase(["this", "price", "is"], max_gap=6) or _find_phrase(["this", "price"], max_gap=6)



        if date_box:

            t = _ocr_near(date_box, "right", "--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789/-. ")

            m = re.search(r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", t)

            if m:

                res["document_date"] = _clean_text(m.group(1))



        if sold_box:

            t = _normalize_supplier(_ocr_near(sold_box, "right", "--oem 1 --psm 7 -c preserve_interword_spaces=1"))

            if t and len(t) >= 3:

                res["supplier"] = t[:120]



        if make_box:

            t = _normalize_make(_ocr_near(make_box, "right_wide", "--oem 1 --psm 7 -c preserve_interword_spaces=1"))

            if t and len(t) >= 2:

                res["make"] = t



        if reg_box:

            t = _ocr_near(reg_box, "right_wide", "--oem 1 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ").upper()

            t = re.sub(r"[^A-Z0-9 ]", " ", t)

            t = _clean_text(t)

            m = re.search(r"\b([A-Z]{2}[0-9O]{2}\s*[A-Z]{3})\b", t)

            if m:

                raw = m.group(1).replace(" ", "")

                raw = raw[:2] + raw[2:4].replace("O", "0") + raw[4:]

                res["reg_no"] = raw[:4] + " " + raw[4:]



        if price_box:

            t = _ocr_near(price_box, "right_wide", "--oem 1 --psm 7 -c tessedit_char_whitelist=£0123456789.,")

            if not t:

                t = _ocr_near(price_box, "above_right", "--oem 1 --psm 7 -c tessedit_char_whitelist=£0123456789.,")

            val = _pick_invoice_price(t)

            if val is not None:

                res["buying_price"] = val

                res["non_vat"] = val



        return res



    def _score_result(r: Dict[str, Any]) -> int:

        score = 0

        if _clean_text(r.get("document_date")):

            score += 4

        if _clean_text(r.get("reg_no")):

            score += 4

        if r.get("buying_price") not in (None, ""):

            score += 4

        if _clean_text(r.get("supplier")):

            score += 2

        if _clean_text(r.get("make")):

            score += 2

        return score



    best: Dict[str, Any] = {}

    best_score = -1

    best_img2: Any = img

    for angle in (0, 90, 180, 270):

        try:

            img2 = img.rotate(angle, expand=True) if angle else img

        except Exception:

            img2 = img



        if dbg_dir:

            _dbg_save(img2, f"page_rot_{angle}_0.png")



        candidates = [img2]

        try:

            candidates.append(_auto_crop_to_red_border(img2))

        except Exception:

            pass



        for ci, cand in enumerate(candidates):

            if dbg_dir and ci == 1:

                _dbg_save(cand, f"page_rot_{angle}_1_cropped.png")



            prev_dbg_dir = dbg_dir

            if prev_dbg_dir:

                try:

                    dbg_dir = os.path.join(prev_dbg_dir, f"rot_{angle}_{ci}")

                    os.makedirs(dbg_dir, exist_ok=True)

                except Exception:

                    dbg_dir = prev_dbg_dir



            r = _ocr_boxes_from_image(cand)

            r2 = _label_based_from_image(cand) if INVOICE_ENABLE_LABEL_OCR else {}



            if prev_dbg_dir:

                dbg_dir = prev_dbg_dir



            if r2:

                for k, v in r2.items():

                    if v in (None, ""):

                        continue

                    if k == "document_date":

                        if (not _is_valid_uk_date(r.get("document_date"))) and _is_valid_uk_date(v):

                            r["document_date"] = v

                        continue

                    if k == "reg_no":

                        cur = _clean_text(r.get("reg_no")).upper()

                        cur_ok = bool(re.match(r"^[A-Z]{2}\d{2}\s?[A-Z]{3}$", cur)) if cur else False

                        v2 = _clean_text(v).upper()

                        v_ok = bool(re.match(r"^[A-Z]{2}\d{2}\s?[A-Z]{3}$", v2)) if v2 else False

                        if (not cur_ok) and v_ok:

                            r["reg_no"] = v2[:4] + " " + v2.replace(" ", "")[4:] if " " not in v2 else v2

                        continue

                    if k in ("buying_price", "non_vat"):

                        try:

                            cur_bp = float(r.get("buying_price")) if r.get("buying_price") not in (None, "") else None

                        except Exception:

                            cur_bp = None

                        try:

                            new_bp = float(v) if v not in (None, "") else None

                        except Exception:

                            new_bp = None

                        if (cur_bp is None or cur_bp <= 0) and (new_bp is not None and 0 < new_bp < 100000):

                            r["buying_price"] = float(new_bp)

                            r["non_vat"] = float(new_bp)

                        continue

                    if k in ("supplier", "make"):

                        if not _clean_text(r.get(k)) and _clean_text(str(v)):

                            r[k] = v

                        continue

            s = _score_result(r)

            if s > best_score:

                best_score = s

                best = r

                best_img2 = cand

            if best_score >= 14:

                break

        if best_score >= 14:

            break



    # Fallback: if date is still missing, try a broad OCR pass to capture handwritten date.

    if not _clean_text(best.get("document_date")) and pytesseract is not None:

        try:

            img_for_date = _invoice_remove_red_print(_auto_crop_to_red_border(best_img2))

            img_for_date = _invoice_preprocess_handwriting(img_for_date)

            txt = pytesseract.image_to_string(

                img_for_date,

                config="--oem 1 --psm 6 -c tessedit_char_whitelist=0123456789/-. ",

            )  # type: ignore[union-attr]

            m = re.search(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b", _clean_text(txt))

            if m:

                best["document_date"] = _clean_text(m.group(1))

        except Exception:

            pass



    # Targeted Date ROI fallback (top-right area where the Date field is on this template).

    if not _clean_text(best.get("document_date")) and pytesseract is not None:

        try:

            base_img = _auto_crop_to_red_border(best_img2)

            W, H = base_img.size

            # Try a couple of tighter ROIs around the Date line (reduces noise).

            rois = [

                (0.52, 0.02, 0.99, 0.12),

                (0.45, 0.00, 0.99, 0.18),

            ]

            configs = [

                "--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789/-. ",

                "--oem 1 --psm 6 -c tessedit_char_whitelist=0123456789/-. ",

                "--oem 1 --psm 11 -c tessedit_char_whitelist=0123456789/-. ",

                "--oem 1 --psm 12 -c tessedit_char_whitelist=0123456789/-. ",

                "--oem 1 --psm 13 -c tessedit_char_whitelist=0123456789/-. ",

            ]

            found_date = ""

            for (lx, ty, rx, by) in rois:

                roi = base_img.crop((int(W * lx), int(H * ty), int(W * rx), int(H * by)))

                roi = roi.resize((roi.size[0] * 2, roi.size[1] * 2))

                variants = []

                try:

                    variants.append(_invoice_preprocess_handwriting(_invoice_remove_red_print(roi)))

                except Exception:

                    pass

                try:

                    variants.append(_invoice_preprocess_handwriting(roi))

                except Exception:

                    pass

                try:

                    v3 = _invoice_preprocess_handwriting(roi)

                    if ImageEnhance is not None:

                        v3 = ImageEnhance.Contrast(v3).enhance(2.2)

                    variants.append(v3)

                except Exception:

                    pass



                for v in variants:

                    tbest = _ocr_best(v, configs, "date")

                    m2 = re.search(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b", _clean_text(tbest))

                    if m2:

                        found_date = _clean_text(m2.group(1))

                        break

                    for cfg in configs:

                        try:

                            txt_try = pytesseract.image_to_string(v, config=cfg)  # type: ignore[union-attr]

                        except Exception:

                            continue

                        m3 = re.search(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b", _clean_text(txt_try))

                        if m3:

                            found_date = _clean_text(m3.group(1))

                            break

                    if found_date:

                        break

                if found_date:

                    break



            if found_date:

                best["document_date"] = found_date

        except Exception:

            pass



    # Top-strip scan fallback: OCR only the upper part of the page and pick the first dd/mm/yy.

    # This avoids any label detection and works well for this template (Date is near the top).

    if not _clean_text(best.get("document_date")) and pytesseract is not None:

        try:

            base_img = _auto_crop_to_red_border(best_img2)

            W, H = base_img.size

            top_h = max(1, int(H * 0.30))

            top_strip = base_img.crop((0, 0, W, top_h))

            top_strip = top_strip.resize((top_strip.size[0] * 2, top_strip.size[1] * 2))

            top_strip = _invoice_remove_red_print(top_strip)

            top_strip = _invoice_preprocess_handwriting(top_strip)

            txt_top = pytesseract.image_to_string(

                top_strip,

                config="--oem 1 --psm 6 -c tessedit_char_whitelist=0123456789/-. ",

            )  # type: ignore[union-attr]

            mtop = re.search(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b", _clean_text(txt_top))

            if mtop:

                best["document_date"] = _clean_text(mtop.group(1))

        except Exception:

            pass



    # Date-label anchored fallback: find the printed 'Date' label and OCR the value to its right.

    if not _clean_text(best.get("document_date")) and pytesseract is not None and TesseractOutput is not None:

        try:

            base_img = _auto_crop_to_red_border(best_img2)

            W, H = base_img.size

            top_h = int(H * 0.28)

            top_strip = base_img.crop((0, 0, W, max(1, top_h)))



            # Downscale for speed and stable token detection.

            max_w = 1200

            if top_strip.size[0] > max_w:

                scale = max_w / float(top_strip.size[0])

                top_strip_small = top_strip.resize((max_w, max(1, int(top_strip.size[1] * scale))))

            else:

                top_strip_small = top_strip

                scale = 1.0



            lbl_img = _invoice_preprocess_for_label_detection(top_strip_small)

            data = pytesseract.image_to_data(lbl_img, output_type=TesseractOutput.DICT, config="--oem 1 --psm 6")  # type: ignore[union-attr]

            n = len(data.get("text", []) or [])



            def _norm(s: str) -> str:

                s2 = _clean_text(s).lower()

                s2 = re.sub(r"[^a-z0-9]+", "", s2)

                return s2



            date_idx: Optional[int] = None

            for i in range(n):

                t = _norm((data.get("text", [""])[i] or ""))

                if t == "date":

                    date_idx = i

                    break

            if date_idx is not None:

                x = int(data.get("left", [0])[date_idx])

                y = int(data.get("top", [0])[date_idx])

                w0 = int(data.get("width", [0])[date_idx])

                h0 = int(data.get("height", [0])[date_idx])



                # Map coords back to the full-res top_strip.

                x = int(x / scale)

                y = int(y / scale)

                w0 = int(w0 / scale)

                h0 = int(h0 / scale)



                pad = max(8, int(h0 * 0.3))

                x1 = min(W - 1, x + w0 + pad)

                y1 = max(0, y - pad)

                x2 = min(W, x1 + int(W * 0.35))

                y2 = min(top_h, y + h0 + pad)

                roi = top_strip.crop((x1, y1, x2, y2))

                roi = roi.resize((roi.size[0] * 2, roi.size[1] * 2))

                roi = _invoice_remove_red_print(roi)

                roi = _invoice_preprocess_handwriting(roi)



                date_txt = _ocr_best(

                    roi,

                    [

                        "--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789/-. ",

                        "--oem 1 --psm 6 -c tessedit_char_whitelist=0123456789/-. ",

                        "--oem 1 --psm 13 -c tessedit_char_whitelist=0123456789/-. ",

                    ],

                    "date",

                )

                m = re.search(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b", _clean_text(date_txt))

                if m:

                    best["document_date"] = _clean_text(m.group(1))

        except Exception:

            pass



    # Final sanity filter: this invoice template's handwritten price is expected to be

    # a single, plausible amount (e.g. £795). If OCR produced an outlier, drop it.

    try:

        bp = best.get("buying_price")

        if isinstance(bp, (int, float)) and bp > 2500:

            best.pop("buying_price", None)

            best.pop("non_vat", None)

    except Exception:

        pass



    return best



@app.get("/api/version")

def version() -> JSONResponse:

    return JSONResponse({"version": APP_VERSION})


@app.get("/api/diagnostics")
def diagnostics() -> JSONResponse:
    info: Dict[str, Any] = {
        "version": APP_VERSION,
        "debug": bool(DEBUG),
        "bankpdf_ocr": bool(BANKPDF_OCR),
        "pytesseract_installed": pytesseract is not None,
        "pdfium_available": pdfium is not None,
        "fitz_available": fitz is not None,
        "pil_available": Image is not None,
        "env": {
            "TESSERACT_CMD": os.getenv("TESSERACT_CMD", ""),
        },
    }

    try:
        info["tesseract_which"] = shutil.which("tesseract") or ""
    except Exception:
        info["tesseract_which"] = ""

    try:
        ok, detail = _tesseract_available()
        info["tesseract_available"] = bool(ok)
        info["tesseract_detail"] = detail
    except Exception as e:
        info["tesseract_available"] = False
        info["tesseract_detail"] = f"Diagnostics failure: {e}"

    return JSONResponse(info)



@app.post("/api/convert")
async def convert_bank_statements(request: Request) -> JSONResponse:
    form = await request.form()
    files: List[UploadFile] = []
    for key in ("files", "file"):
        try:
            items = form.getlist(key)  # type: ignore[attr-defined]
        except Exception:
            items = []
        for it in items:
            if isinstance(it, (UploadFile, StarletteUploadFile)):
                files.append(it)  # type: ignore[arg-type]
            elif hasattr(it, "filename") and hasattr(it, "read"):
                files.append(it)  # type: ignore[arg-type]
    if not files:
        return JSONResponse({"error": "No files uploaded"}, status_code=400)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    job_id = str(uuid.uuid4())
    job_folder = os.path.join(OUTPUT_DIR, job_id)
    os.makedirs(job_folder, exist_ok=True)
    all_rows: List[Dict[str, Any]] = []
    per_file_csv_paths: List[Tuple[str, str]] = []
    warnings: List[str] = []
    fieldnames = ["source_file", "account", "subcategory", "date", "description", "money_in", "money_out", "amount", "balance"]
    combined_preamble: Optional[List[List[str]]] = None
    seen_any_pdf = False
    skipped: List[str] = []
    for f in files:
        filename = os.path.basename(f.filename or "statement.pdf")
        content_type = (f.content_type or "").lower()
        ext = os.path.splitext(filename.lower())[1]
        is_pdf = ext == ".pdf" or content_type == "application/pdf"
        is_image = (content_type.startswith("image/") or ext in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"})
        if not is_pdf and not is_image:
            skipped.append(f"{filename} ({content_type or 'unknown'})")
            continue
        seen_any_pdf = True
        tmp_path = os.path.join(job_folder, filename)
        content = await f.read()
        with open(tmp_path, "wb") as out:
            out.write(content)

        used_ocr = False
        lines: List[str] = []
        if is_pdf:
            lines, used_ocr = _extract_text_lines_from_pdf_with_ocr(tmp_path)
        elif is_image:
            if not BANKPDF_OCR:
                warnings.append(
                    f"'{filename}': File is an image; OCR is disabled (set BANKPDF_OCR=1)."
                )
                lines = []
                used_ocr = False
            else:
                lines, used_ocr = _extract_text_lines_from_image_with_ocr(tmp_path)
                if not used_ocr:
                    ok, detail = _tesseract_available()
                    if not ok:
                        warnings.append(
                            f"'{filename}': Image OCR could not run. Install Tesseract OCR for Windows and/or set TESSERACT_CMD. Details: {detail}"
                        )
        if len(lines) < 1:
            if pytesseract is None:
                warnings.append(
                    f"'{filename}': No readable text was extracted. Also 'pytesseract' is not installed. Install it and install Tesseract OCR for Windows to convert scanned PDFs."
                )
            else:
                ok, detail = _tesseract_available()
                if not ok:
                    warnings.append(
                        f"'{filename}': No readable text was extracted, and Tesseract OCR is not available to run. Install Tesseract OCR for Windows and/or set TESSERACT_CMD. Details: {detail}"
                    )
                else:
                    if fitz is None and pdfium is None:
                        warnings.append(
                            f"'{filename}': OCR is enabled but a PDF-to-image renderer is not available. Install 'pypdfium2' (recommended) or 'PyMuPDF' so pages can be rendered to images for OCR."
                        )
                    warnings.append(
                        f"'{filename}': No readable text was extracted from the PDF. This usually means the PDF is scanned/image-based. OCR is required (install Tesseract OCR and set TESSERACT_CMD if needed)."
                    )
        elif used_ocr:
            warnings.append(f"'{filename}': Extracted text using OCR (scanned PDF detected).")
        account = _extract_account_from_lines(lines)
        barclays_preamble: Optional[List[List[str]]] = None
        if _looks_like_barclays_statement(lines):
            if _looks_like_barclays_business_premium_statement(lines):
                info2 = _extract_barclays_business_premium_header_info(lines)
                barclays_preamble = _barclays_business_premium_preamble_lines(info2)
            else:
                info = _extract_barclays_header_info(lines)
                barclays_preamble = _barclays_header_preamble_lines(info)
            combined_preamble = barclays_preamble
        monzo_preamble: Optional[List[List[str]]] = None
        if _looks_like_monzo_statement(lines):
            minfo = _extract_monzo_header_info(lines)
            monzo_preamble = _monzo_header_preamble_lines(minfo)
            combined_preamble = monzo_preamble
        virgin_preamble: Optional[List[List[str]]] = None
        if _looks_like_virgin_money_statement(lines):
            vinfo = _extract_virgin_money_header_info(lines)
            virgin_preamble = _virgin_money_header_preamble_lines(vinfo)
            combined_preamble = virgin_preamble
        tide_preamble: Optional[List[List[str]]] = None
        if _looks_like_tide_statement(lines):
            tinfo = _extract_tide_header_info(lines)
            tide_preamble = _tide_header_preamble_lines(tinfo)
            combined_preamble = tide_preamble
        revolut_preamble: Optional[List[List[str]]] = None
        if _looks_like_revolut_business_statement(lines):
            rinfo = _extract_revolut_business_header_info(lines)
            revolut_preamble = _revolut_business_preamble_lines(rinfo)
            combined_preamble = revolut_preamble
        rows = convert_pdf_to_rows(tmp_path, preextracted_lines=lines, used_ocr_hint=used_ocr)
        normalized: List[Dict[str, Any]] = []
        for r in rows:
            rr = {
                "source_file": filename,
                "account": account,
                "subcategory": r.get("subcategory")
                or _infer_subcategory(
                    r.get("description") or "",
                    r.get("amount"),
                    r.get("money_in"),
                    r.get("money_out"),
                ),
                "date": r.get("date"),
                "description": r.get("description"),
                "money_in": r.get("money_in"),
                "money_out": r.get("money_out"),
                "amount": r.get("amount"),
                "balance": r.get("balance"),
            }
            normalized.append(rr)
        if not normalized and not used_ocr:
            if is_pdf:
                lines2, used_ocr2 = _extract_text_lines_from_pdf_with_ocr(tmp_path, force_ocr=True)
                if used_ocr2 and lines2:
                    warnings.append(f"'{filename}': Retried extraction using OCR because no transactions were detected.")
                    account = _extract_account_from_lines(lines2)
                    rows2 = convert_pdf_to_rows(tmp_path, preextracted_lines=lines2, used_ocr_hint=True)
                    normalized = []
                    for r in rows2:
                        rr = {
                            "source_file": filename,
                            "account": account,
                            "subcategory": r.get("subcategory")
                            or _infer_subcategory(
                                r.get("description") or "",
                                r.get("amount"),
                                r.get("money_in"),
                                r.get("money_out"),
                            ),
                            "date": r.get("date"),
                            "description": r.get("description"),
                            "money_in": r.get("money_in"),
                            "money_out": r.get("money_out"),
                            "amount": r.get("amount"),
                            "balance": r.get("balance"),
                        }
                        normalized.append(rr)
        if not normalized:
            warnings.append(
                f"No transactions extracted from '{filename}'. This usually happens when the PDF is scanned (image-only) or the layout is different. CSV was generated with headers only."
            )
            if used_ocr and lines:
                sample = " | ".join(lines[:12])
                if sample:
                    warnings.append(f"'{filename}': OCR sample (first lines): {sample}")
        csv_name = os.path.splitext(filename)[0] + ".csv"
        csv_path = os.path.join(job_folder, csv_name)
        if barclays_preamble:
            _write_csv_with_preamble(csv_path, barclays_preamble, normalized, fieldnames)
        elif monzo_preamble:
            _write_csv_with_preamble(csv_path, monzo_preamble, normalized, fieldnames)
        elif virgin_preamble:
            _write_csv_with_preamble(csv_path, virgin_preamble, normalized, fieldnames)
        elif tide_preamble:
            _write_csv_with_preamble(csv_path, tide_preamble, normalized, fieldnames)
        elif revolut_preamble:
            _write_csv_with_preamble(csv_path, revolut_preamble, normalized, fieldnames)
        else:
            _write_csv(csv_path, normalized, fieldnames)
        per_file_csv_paths.append((csv_name, csv_path))
        all_rows.extend(normalized)
    if not seen_any_pdf:
        msg = "No valid PDF files found"
        if skipped:
            msg += ". Skipped: " + ", ".join(skipped[:20])
        return JSONResponse({"error": msg}, status_code=400)
    combined_path = os.path.join(job_folder, "combined.csv")
    # Always include preamble if available, regardless of file count
    if combined_preamble:
        _write_csv_with_preamble(combined_path, combined_preamble, all_rows, fieldnames)
    else:
        _write_csv(combined_path, all_rows, fieldnames)
    JOBS[job_id] = combined_path
    preview = []
    for r in all_rows[:25]:
        preview.append({k: _format_csv_value(r.get(k)) for k in fieldnames})
    return JSONResponse(
        {
            "job_id": job_id,
            "files_processed": len(per_file_csv_paths),
            "rows_total": int(len(all_rows)),
            "warnings": warnings,
            "fieldnames": fieldnames,
            "preview": preview,
            "download_url": f"/api/download/{job_id}",
        }
    )


@app.get("/api/download/{job_id}")
def download(job_id: str) -> FileResponse:
    csv_path = JOBS.get(job_id)
    if not csv_path or not os.path.exists(csv_path):
        job_folder = os.path.join(OUTPUT_DIR, job_id)
        fallback = os.path.join(job_folder, "combined.csv")
        if os.path.exists(fallback):
            csv_path = fallback
        else:
            return FileResponse(path="", status_code=404)
    return FileResponse(csv_path, filename=f"bank_statements_{job_id}.csv", media_type="text/csv")


@app.post("/api/convert-invoice")

async def invoice_convert(request: Request) -> JSONResponse:

    form = await request.form()

    files: List[UploadFile] = []

    for key in ("files", "file"):

        try:

            items = form.getlist(key)  # type: ignore[attr-defined]

        except Exception:

            items = []

        for it in items:

            if isinstance(it, (UploadFile, StarletteUploadFile)):

                files.append(it)  # type: ignore[arg-type]

            elif hasattr(it, "filename") and hasattr(it, "read"):

                files.append(it)  # type: ignore[arg-type]



    if not files:

        return JSONResponse({"error": "No files uploaded"}, status_code=400)



    os.makedirs(OUTPUT_DIR, exist_ok=True)

    job_id = str(uuid.uuid4())

    job_folder = os.path.join(OUTPUT_DIR, job_id)

    os.makedirs(job_folder, exist_ok=True)



    fieldnames = [

        "sr_no",

        "category",

        "document_date",

        "supplier",

        "inv_ref_no",

        "make",

        "model",

        "colour",

        "reg_no",

        "buying_price",

        "non_vat",

        "std_net",

        "vat_amount",

    ]



    rows_out: List[Dict[str, Any]] = []

    warnings: List[str] = []



    seen_any_pdf = False

    skipped: List[str] = []

    sr = 1

    for f in files:

        filename = os.path.basename(f.filename or "invoice.pdf")

        content_type = (f.content_type or "").lower()

        is_pdf = filename.lower().endswith(".pdf") or content_type == "application/pdf"

        if not is_pdf:

            skipped.append(f"{filename} ({content_type or 'unknown'})")

            continue

        seen_any_pdf = True



        tmp_pdf = os.path.join(job_folder, filename)

        content = await f.read()

        with open(tmp_pdf, "wb") as out:

            out.write(content)



        lines, used_ocr = _invoice_extract_text_lines_from_pdf_with_ocr(tmp_pdf)

        if len(lines) < 1:

            ok, detail = _invoice_tesseract_available()

            ocr_reason = ""

            if not ok:

                ocr_reason = f" OCR unavailable: {detail or 'Tesseract not available'}."

            elif Image is None:

                ocr_reason = " OCR unavailable: Pillow (PIL) is not installed."

            elif fitz is None and pdfium is None:

                ocr_reason = " OCR unavailable: no PDF renderer (install pypdfium2 or pymupdf)."

            warnings.append(

                f"'{filename}': No readable text extracted.{ocr_reason} If this is a scanned/handwritten invoice, install and configure OCR (Tesseract) and set TESSERACT_CMD if needed."

            )

        elif used_ocr:

            warnings.append(f"'{filename}': Extracted text using OCR (scanned PDF detected).")



        parsed = _extract_invoice_fields(lines)



        joined_low = "\n".join([_clean_text(x) for x in lines]).lower()

        fn_low = filename.lower()

        is_used_vehicle_purchase = (

            "used vehicle purchase invoice" in joined_low

            or "vehicle purchase invoice" in joined_low

            or ("purchase" in fn_low and "inv" in fn_low)

            or ("used" in fn_low and "vehicle" in fn_low and "invoice" in fn_low)

        )



        missing_critical_any = (

            not _clean_text(parsed.get("document_date"))

            or not _clean_text(parsed.get("supplier"))

            or not _clean_text(parsed.get("make"))

            or not _clean_text(parsed.get("reg_no"))

            or parsed.get("buying_price") in (None, "")

        )



        if is_used_vehicle_purchase and not used_ocr and missing_critical_any:

            missing_critical = (

                not _clean_text(parsed.get("document_date"))

                or not _clean_text(parsed.get("supplier"))

                or not _clean_text(parsed.get("make"))

                or not _clean_text(parsed.get("reg_no"))

                or parsed.get("buying_price") in (None, "")

            )

            if missing_critical:

                lines2, used_ocr2 = _invoice_extract_text_lines_from_pdf_with_ocr(tmp_pdf, force_ocr=True)

                if used_ocr2 and lines2:

                    parsed2 = _extract_invoice_fields(lines2)

                    parsed = parsed2

                    used_ocr = True

                    warnings.append(f"'{filename}': Forced OCR to capture handwritten fields.")



        if is_used_vehicle_purchase:

            extra = _invoice_ocr_used_vehicle_purchase_fields(tmp_pdf)

            if extra:

                for k, v in extra.items():

                    if v not in (None, ""):

                        parsed[k] = v

                if _clean_text(extra.get("supplier")) or _clean_text(extra.get("make")) or _clean_text(extra.get("reg_no")):

                    warnings.append(f"'{filename}': Applied region OCR for handwritten fields.")

            else:

                ok_ocr, _detail_ocr = _invoice_tesseract_available()

                if missing_critical_any and ok_ocr:

                    warnings.append(

                        f"'{filename}': Region OCR returned no usable handwritten text (scan may be cropped/rotated or too low quality)."

                    )



        # BCA scans can have noisy OCR (wrong dates like 21/20/02). If OCR was used and critical

        # fields are missing/invalid, run BCA label-based region OCR and apply only validated values.

        reg_no_val = _clean_text(parsed.get("reg_no")).upper()

        reg_no_vehicle = bool(re.match(r"^[A-Z]{2}\d{2}\s?[A-Z]{3}$", reg_no_val)) if reg_no_val else False

        joined_low2 = "\n".join([_clean_text(x) for x in lines]).lower()

        is_bca = ("british car auctions" in joined_low2) or ("document date" in joined_low2 and bool(re.search(r"\bbca\b", joined_low2)))

        need_bca_ocr = bool(

            used_ocr

            and is_bca

            and (not is_used_vehicle_purchase)

            and (

                (not _is_valid_uk_date(parsed.get("document_date")))

                or (parsed.get("buying_price") in (None, ""))

                or (not _clean_text(parsed.get("inv_ref_no")))

                or (not _clean_text(parsed.get("supplier")))

                or (not _clean_text(parsed.get("make")))

                or (not _clean_text(parsed.get("reg_no")))

                or reg_no_vehicle

            )

        )

        if need_bca_ocr:

            extra_bca = _invoice_ocr_bca_fields(tmp_pdf)

            if extra_bca:

                applied = False

                dd = extra_bca.get("document_date")

                if _is_valid_uk_date(dd):

                    parsed["document_date"] = dd

                    applied = True

                ir = _clean_text(extra_bca.get("inv_ref_no"))

                if ir:

                    parsed["inv_ref_no"] = ir

                    applied = True

                sup = _clean_text(extra_bca.get("supplier"))

                if sup:

                    parsed["supplier"] = sup

                    applied = True

                bp = extra_bca.get("buying_price")

                if isinstance(bp, (int, float)) and 0 < float(bp) < 100000:

                    parsed["buying_price"] = float(bp)

                    parsed["non_vat"] = float(extra_bca.get("non_vat") or bp)

                    applied = True

                mk = _clean_text(extra_bca.get("make"))

                if mk:

                    parsed["make"] = mk

                    applied = True

                rn = _clean_text(extra_bca.get("reg_no"))

                if rn and rn.upper().startswith("GB"):

                    parsed["reg_no"] = rn

                    applied = True

                elif rn and (not _clean_text(parsed.get("reg_no"))):

                    parsed["reg_no"] = rn

                    applied = True

                if applied:

                    warnings.append(f"'{filename}': Applied BCA region OCR for header/total fields.")

        inv_ref_fallback = os.path.splitext(filename)[0]

        inv_ref_value = _clean_text(parsed.get("inv_ref_no"))

        if is_used_vehicle_purchase:

            inv_ref_value = inv_ref_fallback

        elif not inv_ref_value:

            inv_ref_value = inv_ref_fallback



        bp_val = parsed.get("buying_price")

        try:

            bp_num = float(bp_val) if bp_val not in (None, "") else None

        except Exception:

            bp_num = None

        if bp_num is not None and bp_num > 2500:

            bp_num = None

            parsed["buying_price"] = None

            parsed["non_vat"] = None



        row_document_date = parsed.get("document_date")

        row_supplier = parsed.get("supplier")

        row_make = parsed.get("make")

        row_model = parsed.get("model")

        row_colour = parsed.get("colour")

        row_reg_no = parsed.get("reg_no")

        row_buying_price: Any = parsed.get("buying_price")

        row_non_vat: Any = parsed.get("non_vat")

        row_std_net: Any = parsed.get("std_net")

        row_vat_amount: Any = parsed.get("vat_amount")



        if is_used_vehicle_purchase:

            if not _clean_text(row_document_date):

                row_document_date = "N/A"

            if not _clean_text(row_supplier):

                row_supplier = "N/A"

            if not _clean_text(row_make):

                row_make = "N/A"

            if not _clean_text(row_model):

                row_model = "N/A"

            if not _clean_text(row_colour):

                row_colour = "N/A"

            if not _clean_text(row_reg_no):

                row_reg_no = "N/A"

            if row_buying_price in (None, ""):

                row_buying_price = "N/A"

            if row_non_vat in (None, ""):

                row_non_vat = "N/A"

            if row_std_net in (None, ""):

                row_std_net = "N/A"

            if row_vat_amount in (None, ""):

                row_vat_amount = "N/A"



        row = {

            "sr_no": sr,

            "category": "purchase",

            "document_date": row_document_date,

            "supplier": row_supplier,

            "inv_ref_no": inv_ref_value,

            "make": row_make,

            "model": row_model,

            "colour": row_colour,

            "reg_no": row_reg_no,

            "buying_price": row_buying_price,

            "non_vat": row_non_vat,

            "std_net": row_std_net,

            "vat_amount": row_vat_amount,

        }

        rows_out.append(row)

        sr += 1



    if not seen_any_pdf:

        msg = "No valid PDF files found"

        if skipped:

            msg += ". Skipped: " + ", ".join(skipped[:20])

        return JSONResponse({"error": msg}, status_code=400)



    combined_path = os.path.join(job_folder, "combined.csv")

    _write_csv(combined_path, rows_out, fieldnames)

    INVOICE_JOBS[job_id] = combined_path



    preview = []

    for r in rows_out[:25]:

        preview.append({k: _format_csv_value(r.get(k)) for k in fieldnames})



    return JSONResponse(

        {

            "job_id": job_id,

            "files_processed": int(len(rows_out)),

            "rows_total": int(len(rows_out)),

            "warnings": warnings,

            "fieldnames": fieldnames,

            "preview": preview,

            "download_url": f"/api/invoice-download/{job_id}",

        }

    )





@app.post("/api/invoice-convert-review")

async def invoice_convert_review(request: Request) -> JSONResponse:

    form = await request.form()

    files: List[UploadFile] = []

    for key in ("files", "file"):

        try:

            items = form.getlist(key)  # type: ignore[attr-defined]

        except Exception:

            items = []

        for it in items:

            if isinstance(it, (UploadFile, StarletteUploadFile)):

                files.append(it)  # type: ignore[arg-type]

            elif hasattr(it, "filename") and hasattr(it, "read"):

                files.append(it)  # type: ignore[arg-type]



    if not files:

        return JSONResponse({"error": "No files uploaded"}, status_code=400)



    os.makedirs(OUTPUT_DIR, exist_ok=True)

    job_id = str(uuid.uuid4())

    job_folder = os.path.join(OUTPUT_DIR, job_id)

    os.makedirs(job_folder, exist_ok=True)



    fieldnames = [

        "sr_no",

        "category",

        "document_date",

        "supplier",

        "inv_ref_no",

        "make",

        "reg_no",

        "buying_price",

        "non_vat",

        "std_net",

        "vat_amount",

    ]



    rows_out: List[Dict[str, Any]] = []

    warnings: List[str] = []



    seen_any_pdf = False

    skipped: List[str] = []

    sr = 1

    for f in files:

        filename = os.path.basename(f.filename or "invoice.pdf")

        content_type = (f.content_type or "").lower()

        is_pdf = filename.lower().endswith(".pdf") or content_type == "application/pdf"

        if not is_pdf:

            skipped.append(f"{filename} ({content_type or 'unknown'})")

            continue

        seen_any_pdf = True



        tmp_pdf = os.path.join(job_folder, filename)

        content = await f.read()

        with open(tmp_pdf, "wb") as out:

            out.write(content)



        lines, used_ocr = _invoice_extract_text_lines_from_pdf_with_ocr(tmp_pdf)

        if len(lines) < 1:

            ok, detail = _invoice_tesseract_available()

            ocr_reason = ""

            if not ok:

                ocr_reason = f" OCR unavailable: {detail or 'Tesseract not available'}."

            elif Image is None:

                ocr_reason = " OCR unavailable: Pillow (PIL) is not installed."

            elif fitz is None and pdfium is None:

                ocr_reason = " OCR unavailable: no PDF renderer (install pypdfium2 or pymupdf)."

            warnings.append(

                f"'{filename}': No readable text extracted.{ocr_reason} If this is a scanned/handwritten invoice, install and configure OCR (Tesseract) and set TESSERACT_CMD if needed."

            )

        elif used_ocr:

            warnings.append(f"'{filename}': Extracted text using OCR (scanned PDF detected).")



        parsed = _extract_invoice_fields(lines)



        joined_low = "\n".join([_clean_text(x) for x in lines]).lower()

        fn_low = filename.lower()

        is_used_vehicle_purchase = (

            "used vehicle purchase invoice" in joined_low

            or "vehicle purchase invoice" in joined_low

            or ("purchase" in fn_low and "inv" in fn_low)

            or ("used" in fn_low and "vehicle" in fn_low and "invoice" in fn_low)

        )



        missing_critical_any = (

            not _clean_text(parsed.get("document_date"))

            or not _clean_text(parsed.get("supplier"))

            or not _clean_text(parsed.get("make"))

            or not _clean_text(parsed.get("reg_no"))

            or parsed.get("buying_price") in (None, "")

        )



        if is_used_vehicle_purchase and not used_ocr and missing_critical_any:

            missing_critical = (

                not _clean_text(parsed.get("document_date"))

                or not _clean_text(parsed.get("supplier"))

                or not _clean_text(parsed.get("make"))

                or not _clean_text(parsed.get("reg_no"))

                or parsed.get("buying_price") in (None, "")

            )

            if missing_critical:

                lines2, used_ocr2 = _invoice_extract_text_lines_from_pdf_with_ocr(tmp_pdf, force_ocr=True)

                if used_ocr2 and lines2:

                    parsed2 = _extract_invoice_fields(lines2)

                    parsed = parsed2

                    used_ocr = True

                    warnings.append(f"'{filename}': Forced OCR to capture handwritten fields.")



        if is_used_vehicle_purchase:

            extra = _invoice_ocr_used_vehicle_purchase_fields(tmp_pdf)

            if extra:

                for k, v in extra.items():

                    if v not in (None, ""):

                        parsed[k] = v

                if _clean_text(extra.get("supplier")) or _clean_text(extra.get("make")) or _clean_text(extra.get("reg_no")):

                    warnings.append(f"'{filename}': Applied region OCR for handwritten fields.")

            else:

                ok_ocr, _detail_ocr = _invoice_tesseract_available()

                if missing_critical_any and ok_ocr:

                    warnings.append(

                        f"'{filename}': Region OCR returned no usable handwritten text (scan may be cropped/rotated or too low quality)."

                    )



        # BCA scans can have very noisy OCR (wrong dates like 21/20/02). If OCR was used and

        # critical header/total fields are missing or invalid, run label-based BCA region OCR.

        reg_no_val = _clean_text(parsed.get("reg_no")).upper()

        reg_no_vehicle = bool(re.match(r"^[A-Z]{2}\d{2}\s?[A-Z]{3}$", reg_no_val)) if reg_no_val else False

        joined_low2 = "\n".join([_clean_text(x) for x in lines]).lower()

        is_bca = ("british car auctions" in joined_low2) or ("document date" in joined_low2 and bool(re.search(r"\bbca\b", joined_low2)))

        need_bca_ocr = bool(

            used_ocr

            and is_bca

            and (not is_used_vehicle_purchase)

            and (

                (not _is_valid_uk_date(parsed.get("document_date")))

                or (parsed.get("buying_price") in (None, ""))

                or (not _clean_text(parsed.get("inv_ref_no")))

                or (not _clean_text(parsed.get("supplier")))

                or (not _clean_text(parsed.get("make")))

                or (not _clean_text(parsed.get("reg_no")))

                or reg_no_vehicle

            )

        )

        if need_bca_ocr:

            extra_bca = _invoice_ocr_bca_fields(tmp_pdf)

            if extra_bca:

                applied = False

                dd = extra_bca.get("document_date")

                if _is_valid_uk_date(dd):

                    parsed["document_date"] = dd

                    applied = True

                ir = _clean_text(extra_bca.get("inv_ref_no"))

                if ir:

                    parsed["inv_ref_no"] = ir

                    applied = True

                sup = _clean_text(extra_bca.get("supplier"))

                if sup:

                    parsed["supplier"] = sup

                    applied = True

                bp = extra_bca.get("buying_price")

                if isinstance(bp, (int, float)) and 0 < float(bp) < 100000:

                    parsed["buying_price"] = float(bp)

                    parsed["non_vat"] = float(extra_bca.get("non_vat") or bp)

                    applied = True

                mk = _clean_text(extra_bca.get("make"))

                if mk:

                    parsed["make"] = mk

                    applied = True

                rn = _clean_text(extra_bca.get("reg_no"))

                if rn and rn.upper().startswith("GB"):

                    parsed["reg_no"] = rn

                    applied = True

                elif rn and (not _clean_text(parsed.get("reg_no"))):

                    parsed["reg_no"] = rn

                    applied = True

                if applied:

                    warnings.append(f"'{filename}': Applied BCA region OCR for header/total fields.")



        inv_ref_fallback = os.path.splitext(filename)[0]

        inv_ref_value = _clean_text(parsed.get("inv_ref_no"))

        if is_used_vehicle_purchase:

            inv_ref_value = inv_ref_fallback

        elif not inv_ref_value:

            inv_ref_value = inv_ref_fallback



        bp_val = parsed.get("buying_price")

        try:

            bp_num = float(bp_val) if bp_val not in (None, "") else None

        except Exception:

            bp_num = None

        if bp_num is not None and bp_num > 2500:

            bp_num = None

            parsed["buying_price"] = None

            parsed["non_vat"] = None



        row_document_date = parsed.get("document_date")

        row_supplier = parsed.get("supplier")

        row_make = parsed.get("make")

        row_reg_no = parsed.get("reg_no")

        row_buying_price: Any = parsed.get("buying_price")

        row_non_vat: Any = parsed.get("non_vat")

        row_std_net: Any = parsed.get("std_net")

        row_vat_amount: Any = parsed.get("vat_amount")



        if is_used_vehicle_purchase:

            if not _clean_text(row_document_date):

                row_document_date = "N/A"

            if not _clean_text(row_supplier):

                row_supplier = "N/A"

            if not _clean_text(row_make):

                row_make = "N/A"

            if not _clean_text(row_reg_no):

                row_reg_no = "N/A"

            if row_buying_price in (None, ""):

                row_buying_price = "N/A"

            if row_non_vat in (None, ""):

                row_non_vat = "N/A"

            if row_std_net in (None, ""):

                row_std_net = "N/A"

            if row_vat_amount in (None, ""):

                row_vat_amount = "N/A"



        row = {

            "sr_no": sr,

            "category": "purchase",

            "document_date": row_document_date,

            "supplier": row_supplier,

            "inv_ref_no": inv_ref_value,

            "make": row_make,

            "reg_no": row_reg_no,

            "buying_price": row_buying_price,

            "non_vat": row_non_vat,

            "std_net": row_std_net,

            "vat_amount": row_vat_amount,

        }

        rows_out.append(row)

        sr += 1



    if not seen_any_pdf:

        msg = "No valid PDF files found"

        if skipped:

            msg += ". Skipped: " + ", ".join(skipped[:20])

        return JSONResponse({"error": msg}, status_code=400)



    draft_path = os.path.join(job_folder, "draft.json")

    payload = {"fieldnames": fieldnames, "rows": rows_out}

    _write_json(draft_path, payload)

    INVOICE_REVIEW_JOBS[job_id] = draft_path



    preview = []

    for r in rows_out[:25]:

        preview.append({k: _format_csv_value(r.get(k)) for k in fieldnames})



    return JSONResponse(

        {

            "job_id": job_id,

            "files_processed": int(len(rows_out)),

            "rows_total": int(len(rows_out)),

            "warnings": warnings,

            "fieldnames": fieldnames,

            "preview": preview,

            "draft_rows": rows_out,

        }

    )





@app.post("/api/invoice-confirm/{job_id}")

async def invoice_confirm(job_id: str, request: Request) -> JSONResponse:

    draft_path = INVOICE_REVIEW_JOBS.get(job_id)

    if not draft_path or not os.path.exists(draft_path):

        return JSONResponse({"error": "Invalid or expired job_id"}, status_code=404)



    try:

        payload_in = await request.json()

    except Exception:

        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)



    rows_in = payload_in.get("rows") if isinstance(payload_in, dict) else None

    if not isinstance(rows_in, list):

        return JSONResponse({"error": "'rows' must be a list"}, status_code=400)



    try:

        draft = _read_json(draft_path)

        fieldnames = draft.get("fieldnames") or []

    except Exception:

        return JSONResponse({"error": "Draft data could not be read"}, status_code=500)



    if not isinstance(fieldnames, list) or not fieldnames:

        fieldnames = [

            "sr_no",

            "category",

            "document_date",

            "supplier",

            "inv_ref_no",

            "make",

            "model",

            "colour",

            "reg_no",

            "buying_price",

            "non_vat",

            "std_net",

            "vat_amount",

        ]



    cleaned_rows: List[Dict[str, Any]] = []

    for idx, r in enumerate(rows_in, start=1):

        if not isinstance(r, dict):

            continue

        sr_no = r.get("sr_no")

        try:

            sr_val = int(sr_no) if sr_no not in (None, "") else idx

        except Exception:

            sr_val = idx

        category = _clean_text(r.get("category")) or "purchase"

        document_date = _clean_text(r.get("document_date"))

        supplier = _clean_text(r.get("supplier"))

        inv_ref_no = _clean_text(r.get("inv_ref_no"))

        make = _clean_text(r.get("make"))

        model = _clean_text(r.get("model"))

        colour = _clean_text(r.get("colour"))

        reg_no = _clean_text(r.get("reg_no"))



        buying_price_in = r.get("buying_price")

        non_vat_in = r.get("non_vat")

        vat_amount_in = r.get("vat_amount")



        buying_price: Any = (

            "N/A" if _clean_text(buying_price_in).upper() in {"N/A", "NA"} else _to_float_or_none(buying_price_in)

        )

        non_vat: Any = "N/A" if _clean_text(non_vat_in).upper() in {"N/A", "NA"} else _to_float_or_none(non_vat_in)

        std_net_raw: Any = r.get("std_net")

        std_net = _to_float_or_none(std_net_raw) if _clean_text(std_net_raw).upper() not in {"N/A", "NA"} else "N/A"



        vat_amount: Any = "N/A" if _clean_text(vat_amount_in).upper() in {"N/A", "NA"} else _to_float_or_none(vat_amount_in)



        if isinstance(buying_price, (int, float)) and buying_price > 2500:

            buying_price = None

        if isinstance(non_vat, (int, float)) and non_vat > 2500:

            non_vat = None



        cleaned_rows.append(

            {

                "sr_no": sr_val,

                "category": category,

                "document_date": document_date,

                "supplier": supplier,

                "inv_ref_no": inv_ref_no,

                "make": make,

                "model": model,

                "colour": colour,

                "reg_no": reg_no,

                "buying_price": buying_price,

                "non_vat": non_vat,

                "std_net": std_net,

                "vat_amount": vat_amount,

            }

        )



    job_folder = os.path.dirname(draft_path)

    combined_path = os.path.join(job_folder, "combined.csv")

    _write_csv(combined_path, cleaned_rows, fieldnames)

    INVOICE_JOBS[job_id] = combined_path



    preview = []

    for r in cleaned_rows[:25]:

        preview.append({k: _format_csv_value(r.get(k)) for k in fieldnames})



    return JSONResponse(

        {

            "job_id": job_id,

            "rows_total": int(len(cleaned_rows)),

            "fieldnames": fieldnames,

            "preview": preview,

            "download_url": f"/api/invoice-download/{job_id}",

        }

    )





@app.get("/api/invoice-download/{job_id}")

def invoice_download(job_id: str) -> FileResponse:

    csv_path = INVOICE_JOBS.get(job_id)

    if not csv_path or not os.path.exists(csv_path):

        return FileResponse(path="", status_code=404)

    return FileResponse(csv_path, filename=f"invoices_{job_id}.csv", media_type="text/csv")





@app.get("/api/download/{job_id}")

def download(job_id: str) -> FileResponse:

    csv_path = JOBS.get(job_id)

    if not csv_path or not os.path.exists(csv_path):

        return FileResponse(path="", status_code=404)

    return FileResponse(csv_path, filename=f"bank_statements_{job_id}.csv", media_type="text/csv")

