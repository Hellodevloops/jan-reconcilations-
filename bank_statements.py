import os

from typing import Any, Dict, List, Optional, Tuple


async def convert_bank_statements(
    request: Any,
    *,
    output_dir: str,
    jobs: Dict[str, str],
    UploadFileT: Any,
    StarletteUploadFileT: Any,
    clean_text: Any,
    to_float_or_none: Any,
    infer_subcategory: Any,
    extract_text_lines_from_pdf_with_ocr: Any,
    extract_text_lines_from_image_with_ocr: Any,
    tesseract_available: Any,
    convert_pdf_to_rows: Any,
    extract_account_from_lines: Any,
    looks_like_barclays_statement: Any,
    looks_like_barclays_business_premium_statement: Any,
    extract_barclays_business_premium_header_info: Any,
    barclays_business_premium_preamble_lines: Any,
    extract_barclays_header_info: Any,
    barclays_header_preamble_lines: Any,
    looks_like_monzo_statement: Any,
    extract_monzo_header_info: Any,
    monzo_header_preamble_lines: Any,
    looks_like_virgin_money_statement: Any,
    extract_virgin_money_header_info: Any,
    virgin_money_header_preamble_lines: Any,
    looks_like_tide_statement: Any,
    extract_tide_header_info: Any,
    tide_header_preamble_lines: Any,
    looks_like_revolut_business_statement: Any,
    extract_revolut_business_header_info: Any,
    revolut_business_preamble_lines: Any,
    write_csv: Any,
    write_csv_with_preamble: Any,
    write_barclays_csv_with_pending: Any,
    format_csv_value: Any,
    JSONResponseT: Any,
    BANKPDF_OCR: bool,
    pytesseract_installed: bool,
    fitz_available: bool,
    pdfium_available: bool,
    uuid4: Any,
) -> Any:
    form = await request.form()

    files: List[Any] = []
    for key in ("files", "file"):
        try:
            items = form.getlist(key)  # type: ignore[attr-defined]
        except Exception:
            items = []
        for it in items:
            if isinstance(it, (UploadFileT, StarletteUploadFileT)):
                files.append(it)  # type: ignore[arg-type]
            elif hasattr(it, "filename") and hasattr(it, "read"):
                files.append(it)  # type: ignore[arg-type]

    if not files:
        return JSONResponseT({"error": "No files uploaded"}, status_code=400)

    os.makedirs(output_dir, exist_ok=True)
    job_id = str(uuid4())
    job_folder = os.path.join(output_dir, job_id)
    os.makedirs(job_folder, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    all_pending_rows: List[Dict[str, Any]] = []
    per_file_csv_paths: List[Tuple[str, str]] = []
    warnings: List[str] = []

    fieldnames = ["source_file", "account", "subcategory", "date", "description", "money_in", "money_out", "amount", "balance"]

    combined_preamble: Optional[List[List[str]]] = None
    combined_preamble_is_barclays = False
    seen_any_file = False
    skipped: List[str] = []

    def _read_csv_bytes_to_rows(content_bytes: bytes) -> List[Dict[str, Any]]:
        try:
            text = content_bytes.decode("utf-8-sig", errors="replace")
        except Exception:
            try:
                text = content_bytes.decode("latin-1", errors="replace")
            except Exception:
                return []

        sample = text[:2000]
        try:
            import csv

            dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"])
        except Exception:
            import csv

            dialect = csv.excel

        import csv
        import io

        f = io.StringIO(text)
        try:
            reader = csv.DictReader(f, dialect=dialect)
        except Exception:
            f.seek(0)
            reader = csv.DictReader(f)

        def _pick(d: Dict[str, Any], keys: List[str]) -> Any:
            for k in keys:
                for kk, vv in d.items():
                    if clean_text(kk).lower() == k:
                        return vv
            return ""

        out: List[Dict[str, Any]] = []
        for row in reader:
            if not row:
                continue
            date_v = _pick(row, ["date", "transaction date", "txn date"])
            desc_v = _pick(row, ["description", "transaction", "details", "narrative"])
            mi_v = _pick(row, ["money_in", "money in", "credit", "in"])
            mo_v = _pick(row, ["money_out", "money out", "debit", "out"])
            bal_v = _pick(row, ["balance", "running balance"])
            amt_v = _pick(row, ["amount", "transaction amount", "txn amount"])
            sub_v = _pick(row, ["subcategory", "category", "type"])
            acc_v = _pick(row, ["account", "account number"])

            rr: Dict[str, Any] = {
                "source_file": "",
                "account": clean_text(acc_v) or "",
                "subcategory": clean_text(sub_v) or "",
                "date": clean_text(date_v) or "",
                "description": clean_text(desc_v) or "",
                "money_in": clean_text(mi_v) or "",
                "money_out": clean_text(mo_v) or "",
                "balance": clean_text(bal_v) or "",
            }

            if amt_v not in (None, ""):
                rr["amount"] = to_float_or_none(amt_v)
            else:
                rr["amount"] = to_float_or_none(rr.get("money_in") or rr.get("money_out") or "")

            if not any(clean_text(rr.get(k)) for k in ["date", "description", "money_in", "money_out", "balance"]):
                continue

            out.append(rr)

        return out

    for f in files:
        filename = os.path.basename(getattr(f, "filename", None) or "statement.pdf")
        content_type = (getattr(f, "content_type", None) or "").lower()
        ext = os.path.splitext(filename.lower())[1]
        is_pdf = ext == ".pdf" or content_type == "application/pdf"
        is_image = content_type.startswith("image/") or ext in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
        is_csv = ext == ".csv" or content_type in {"text/csv", "application/csv", "application/vnd.ms-excel"}

        if not is_pdf and not is_image and not is_csv:
            skipped.append(f"{filename} ({content_type or 'unknown'})")
            continue

        seen_any_file = True
        tmp_path = os.path.join(job_folder, filename)
        content = await f.read()
        with open(tmp_path, "wb") as out:
            out.write(content)

        if is_csv:
            csv_rows = _read_csv_bytes_to_rows(content)
            normalized: List[Dict[str, Any]] = []
            for r in csv_rows:
                rr = {
                    "source_file": filename,
                    "account": r.get("account") or "",
                    "subcategory": r.get("subcategory")
                    or infer_subcategory(
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

            csv_name = os.path.splitext(filename)[0] + ".csv"
            csv_path = os.path.join(job_folder, csv_name)
            write_csv(csv_path, normalized, fieldnames)
            per_file_csv_paths.append((csv_name, csv_path))
            all_rows.extend(normalized)
            continue

        used_ocr = False
        lines: List[str] = []
        if is_pdf:
            lines, used_ocr = extract_text_lines_from_pdf_with_ocr(tmp_path)
        elif is_image:
            if not BANKPDF_OCR:
                warnings.append(f"'{filename}': File is an image; OCR is disabled (set BANKPDF_OCR=1).")
                lines = []
                used_ocr = False
            else:
                lines, used_ocr = extract_text_lines_from_image_with_ocr(tmp_path)
                if not used_ocr:
                    ok, detail = tesseract_available()
                    if not ok:
                        warnings.append(
                            f"'{filename}': Image OCR could not run. Install Tesseract OCR for Windows and/or set TESSERACT_CMD. Details: {detail}"
                        )

        if len(lines) < 1:
            if not pytesseract_installed:
                warnings.append(
                    f"'{filename}': No readable text was extracted. Also 'pytesseract' is not installed. Install it and install Tesseract OCR for Windows to convert scanned PDFs."
                )
            else:
                ok, detail = tesseract_available()
                if not ok:
                    warnings.append(
                        f"'{filename}': No readable text was extracted, and Tesseract OCR is not available to run. Install Tesseract OCR for Windows and/or set TESSERACT_CMD. Details: {detail}"
                    )
                else:
                    if (not fitz_available) and (not pdfium_available):
                        warnings.append(
                            f"'{filename}': OCR is enabled but a PDF-to-image renderer is not available. Install 'pypdfium2' (recommended) or 'PyMuPDF' so pages can be rendered to images for OCR."
                        )
                    warnings.append(
                        f"'{filename}': No readable text was extracted from the PDF. This usually means the PDF is scanned/image-based. OCR is required (install Tesseract OCR and set TESSERACT_CMD if needed)."
                    )
        elif used_ocr:
            warnings.append(f"'{filename}': Extracted text using OCR (scanned PDF detected).")

        account = extract_account_from_lines(lines)

        barclays_preamble: Optional[List[List[str]]] = None
        if looks_like_barclays_statement(lines):
            if looks_like_barclays_business_premium_statement(lines):
                info2 = extract_barclays_business_premium_header_info(lines)
                barclays_preamble = barclays_business_premium_preamble_lines(info2)
            else:
                info = extract_barclays_header_info(lines)
                if info.get("account"):
                    account = str(info.get("account") or "")
                barclays_preamble = barclays_header_preamble_lines(info)
            combined_preamble = barclays_preamble
            combined_preamble_is_barclays = True

        monzo_preamble: Optional[List[List[str]]] = None
        if looks_like_monzo_statement(lines):
            minfo = extract_monzo_header_info(lines)
            monzo_preamble = monzo_header_preamble_lines(minfo)
            combined_preamble = monzo_preamble
            combined_preamble_is_barclays = False

        virgin_preamble: Optional[List[List[str]]] = None
        if looks_like_virgin_money_statement(lines):
            vinfo = extract_virgin_money_header_info(lines)
            virgin_preamble = virgin_money_header_preamble_lines(vinfo)
            combined_preamble = virgin_preamble
            combined_preamble_is_barclays = False

        tide_preamble: Optional[List[List[str]]] = None
        if looks_like_tide_statement(lines):
            tinfo = extract_tide_header_info(lines)
            tide_preamble = tide_header_preamble_lines(tinfo)
            combined_preamble = tide_preamble
            combined_preamble_is_barclays = False

        revolut_preamble: Optional[List[List[str]]] = None
        if looks_like_revolut_business_statement(lines):
            rinfo = extract_revolut_business_header_info(lines)
            revolut_preamble = revolut_business_preamble_lines(rinfo)
            combined_preamble = revolut_preamble
            combined_preamble_is_barclays = False

        rows = convert_pdf_to_rows(tmp_path, preextracted_lines=lines, used_ocr_hint=used_ocr)

        normalized2: List[Dict[str, Any]] = []
        for r in rows:
            rr2 = {
                "__section": r.get("__section"),
                "source_file": filename,
                "account": account,
                "subcategory": r.get("subcategory")
                or infer_subcategory(
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
            normalized2.append(rr2)

        pending_norm = [r for r in normalized2 if r.get("__section") == "barclays_pending_debit_card"]
        main_norm = [r for r in normalized2 if r.get("__section") != "barclays_pending_debit_card"]

        if not normalized2 and not used_ocr and is_pdf:
            lines2, used_ocr2 = extract_text_lines_from_pdf_with_ocr(tmp_path, force_ocr=True)
            if used_ocr2 and lines2:
                warnings.append(f"'{filename}': Retried extraction using OCR because no transactions were detected.")
                account2 = extract_account_from_lines(lines2)
                rows2 = convert_pdf_to_rows(tmp_path, preextracted_lines=lines2, used_ocr_hint=True)
                normalized3: List[Dict[str, Any]] = []
                for r2 in rows2:
                    rr3 = {
                        "source_file": filename,
                        "account": account2,
                        "subcategory": r2.get("subcategory")
                        or infer_subcategory(
                            r2.get("description") or "",
                            r2.get("amount"),
                            r2.get("money_in"),
                            r2.get("money_out"),
                        ),
                        "date": r2.get("date"),
                        "description": r2.get("description"),
                        "money_in": r2.get("money_in"),
                        "money_out": r2.get("money_out"),
                        "amount": r2.get("amount"),
                        "balance": r2.get("balance"),
                    }
                    normalized3.append(rr3)
                main_norm = normalized3
                pending_norm = []

        if not main_norm:
            warnings.append(
                f"No transactions extracted from '{filename}'. This usually happens when the PDF is scanned (image-only) or the layout is different. CSV was generated with headers only."
            )
            if used_ocr and lines:
                sample = " | ".join(lines[:12])
                if sample:
                    warnings.append(f"'{filename}': OCR sample (first lines): {sample}")

        csv_name2 = os.path.splitext(filename)[0] + ".csv"
        csv_path2 = os.path.join(job_folder, csv_name2)

        if barclays_preamble:
            write_barclays_csv_with_pending(csv_path2, barclays_preamble, pending_norm, main_norm, fieldnames)
        elif monzo_preamble:
            write_csv_with_preamble(csv_path2, monzo_preamble, main_norm, fieldnames)
        elif virgin_preamble:
            write_csv_with_preamble(csv_path2, virgin_preamble, main_norm, fieldnames)
        elif tide_preamble:
            write_csv_with_preamble(csv_path2, tide_preamble, main_norm, fieldnames)
        elif revolut_preamble:
            write_csv_with_preamble(csv_path2, revolut_preamble, main_norm, fieldnames)
        else:
            write_csv(csv_path2, main_norm, fieldnames)

        per_file_csv_paths.append((csv_name2, csv_path2))
        all_rows.extend(main_norm)
        all_pending_rows.extend(pending_norm)

    if not seen_any_file:
        msg = "No valid files found"
        if skipped:
            msg += ". Skipped: " + ", ".join(skipped[:20])
        return JSONResponseT({"error": msg}, status_code=400)

    combined_path = os.path.join(job_folder, "combined.csv")
    if combined_preamble and combined_preamble_is_barclays:
        write_barclays_csv_with_pending(combined_path, combined_preamble, all_pending_rows, all_rows, fieldnames)
    elif combined_preamble:
        write_csv_with_preamble(combined_path, combined_preamble, all_rows, fieldnames)
    else:
        write_csv(combined_path, all_rows, fieldnames)

    jobs[job_id] = combined_path

    preview = []
    for r in all_rows[:25]:
        preview.append({k: format_csv_value(r.get(k)) for k in fieldnames})

    return JSONResponseT(
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


def download_bank_statements(job_id: str, *, output_dir: str, jobs: Dict[str, str], FileResponseT: Any) -> Any:
    csv_path = jobs.get(job_id)
    if not csv_path or not os.path.exists(csv_path):
        job_folder = os.path.join(output_dir, job_id)
        fallback = os.path.join(job_folder, "combined.csv")
        if os.path.exists(fallback):
            csv_path = fallback
        else:
            return FileResponseT(path="", status_code=404)
    return FileResponseT(csv_path, filename=f"bank_statements_{job_id}.csv", media_type="text/csv")
