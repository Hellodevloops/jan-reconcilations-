import os

from typing import Any, Dict, List


async def invoice_convert(
    request: Any,
    *,
    output_dir: str,
    invoice_jobs: Dict[str, str],
    UploadFileT: Any,
    StarletteUploadFileT: Any,
    clean_text: Any,
    format_csv_value: Any,
    JSONResponseT: Any,
    uuid4: Any,
    invoice_pdf_page_count: Any,
    invoice_extract_text_lines_from_pdf_pages_with_ocr: Any,
    extract_invoice_fields: Any,
    invoice_row_from_parsed: Any,
    invoice_extract_text_lines_from_pdf_with_ocr: Any,
    invoice_tesseract_available: Any,
    Image_available: bool,
    fitz_available: bool,
    pdfium_available: bool,
    OCR_PROVIDER: str,
    deepseek_extract_used_vehicle_fields_from_pdf: Any,
    invoice_ocr_used_vehicle_purchase_fields: Any,
    is_valid_uk_date: Any,
    invoice_ocr_autotrader_costs_box: Any,
    invoice_ocr_bca_fields: Any,
    write_csv: Any,
    to_float_or_none: Any,
    re_mod: Any,
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
        filename = os.path.basename(getattr(f, "filename", None) or "invoice.pdf")
        content_type = (getattr(f, "content_type", None) or "").lower()
        is_pdf = filename.lower().endswith(".pdf") or content_type == "application/pdf"

        if not is_pdf:
            skipped.append(f"{filename} ({content_type or 'unknown'})")
            continue

        seen_any_pdf = True
        tmp_pdf = os.path.join(job_folder, filename)
        content = await f.read()
        with open(tmp_pdf, "wb") as out:
            out.write(content)

        page_count = invoice_pdf_page_count(tmp_pdf)
        if page_count > 1:
            pages_lines, used_ocr_pages = invoice_extract_text_lines_from_pdf_pages_with_ocr(tmp_pdf)
            if not pages_lines:
                warnings.append(f"'{filename}': Could not extract any text from PDF pages.")
            else:
                if used_ocr_pages:
                    warnings.append(f"'{filename}': Extracted text using OCR (scanned PDF detected).")
                for page_idx, page_lines in enumerate(pages_lines, start=1):
                    if not page_lines:
                        warnings.append(f"'{filename}': Page {page_idx}: No readable text extracted.")
                        continue
                    parsed_page = extract_invoice_fields(page_lines)
                    inv_ref_fallback_page = os.path.splitext(filename)[0] + f"_p{page_idx}"
                    inv_ref_value_page = clean_text(parsed_page.get("inv_ref_no")) or inv_ref_fallback_page
                    row = invoice_row_from_parsed(sr, parsed_page, inv_ref_value_page, used_vehicle_purchase=False)
                    rows_out.append(row)
                    sr += 1
            continue

        fn_low = filename.lower()
        quick_used_vehicle = bool(
            ("used vehicle" in fn_low and "invoice" in fn_low)
            or ("vehicle purchase" in fn_low and "invoice" in fn_low)
            or ("used_vehicle" in fn_low and "invoice" in fn_low)
        )

        parsed: Dict[str, Any] = {}
        used_ocr = False
        is_used_vehicle_purchase = False

        if quick_used_vehicle:
            try:
                extra0 = invoice_ocr_used_vehicle_purchase_fields(tmp_pdf)
            except Exception:
                extra0 = {}

            def _has_key_fields(x: Dict[str, Any]) -> bool:
                if not isinstance(x, dict):
                    return False
                dd = clean_text(x.get("document_date")).replace("-", "/")
                rn = clean_text(x.get("reg_no")).upper()
                has_date = is_valid_uk_date(dd)
                has_vrm = bool(re_mod.search(r"\b[A-Z]{2}[0-9O]{2}\s*[A-Z]{3}\b", rn))
                has_amount = x.get("buying_price") not in (None, "")
                return bool((has_date or has_vrm) and has_amount)

            if extra0 and _has_key_fields(extra0):
                parsed = {
                    "document_date": extra0.get("document_date") or "",
                    "supplier": extra0.get("supplier") or "",
                    "inv_ref_no": "",
                    "make": extra0.get("make") or "",
                    "model": extra0.get("model") or "",
                    "colour": extra0.get("colour") or "",
                    "reg_no": extra0.get("reg_no") or "",
                    "buying_price": extra0.get("buying_price"),
                    "non_vat": extra0.get("non_vat") if extra0.get("non_vat") not in (None, "") else extra0.get("buying_price"),
                    "std_net": "N/A",
                    "vat_amount": "N/A",
                }
                used_ocr = True
                is_used_vehicle_purchase = True
                warnings.append(f"'{filename}': Used fast region OCR for handwritten fields (skipped full-page OCR).")

            if (not parsed) and (OCR_PROVIDER == "deepseek"):
                try:
                    ds = deepseek_extract_used_vehicle_fields_from_pdf(tmp_pdf)
                except Exception:
                    ds = {}

                if ds and isinstance(ds, dict):
                    dd = clean_text(ds.get("document_date")).replace("-", "/")
                    rn = clean_text(ds.get("reg_no")).upper()
                    mreg = re_mod.search(r"\b([A-Z]{2}[0-9O]{2}\s*[A-Z]{3})\b", rn)
                    bp: Any = ds.get("buying_price")
                    try:
                        bp_num = float(bp) if bp not in (None, "") else None
                    except Exception:
                        bp_num = None

                    if is_valid_uk_date(dd) and mreg and (bp_num is not None and 0 < bp_num < 100000):
                        reg_raw = clean_text(mreg.group(1)).upper().replace(" ", "")
                        reg_raw = reg_raw[:2] + reg_raw[2:4].replace("O", "0") + reg_raw[4:]
                        parsed = {
                            "document_date": dd,
                            "supplier": clean_text(ds.get("supplier"))[:120],
                            "inv_ref_no": "",
                            "make": clean_text(ds.get("make"))[:80],
                            "model": clean_text(ds.get("model"))[:60],
                            "colour": clean_text(ds.get("colour"))[:40],
                            "reg_no": reg_raw[:4] + " " + reg_raw[4:],
                            "buying_price": float(bp_num),
                            "non_vat": float(bp_num),
                            "std_net": "N/A",
                            "vat_amount": "N/A",
                        }
                        used_ocr = True
                        is_used_vehicle_purchase = True
                        warnings.append(f"'{filename}': Used DeepSeek structured extraction for handwritten fields.")

        lines: List[str] = []
        if not is_used_vehicle_purchase:
            lines, used_ocr = invoice_extract_text_lines_from_pdf_with_ocr(tmp_pdf)

        if len(lines) < 1:
            ok, detail = invoice_tesseract_available()
            ocr_reason = ""
            if not ok:
                ocr_reason = f" OCR unavailable: {detail or 'Tesseract not available'}."
            elif not Image_available:
                ocr_reason = " OCR unavailable: Pillow (PIL) is not installed."
            elif (not fitz_available) and (not pdfium_available):
                ocr_reason = " OCR unavailable: no PDF renderer (install pypdfium2 or pymupdf)."
            warnings.append(
                f"'{filename}': No readable text extracted.{ocr_reason} If this is a scanned/handwritten invoice, install and configure OCR (Tesseract) and set TESSERACT_CMD if needed."
            )
        elif used_ocr:
            warnings.append(f"'{filename}': Extracted text using OCR (scanned PDF detected).")

        if not parsed:
            parsed = extract_invoice_fields(lines)

        joined_low = "\n".join([clean_text(x) for x in lines]).lower()
        is_autotrader = (
            ("autotrader" in joined_low or "auto trader" in joined_low)
            or ("autotrader" in fn_low or "auto_trader" in fn_low or "auto-trader" in fn_low)
        )
        missing_totals = (
            parsed.get("buying_price") in (None, "", "N/A")
            or parsed.get("non_vat") in (None, "", "N/A")
            or parsed.get("std_net") in (None, "", "N/A")
            or parsed.get("vat_amount") in (None, "", "N/A")
        )
        if missing_totals:
            if is_autotrader:
                warnings.append(f"'{filename}': Attempting AutoTrader totals OCR (totals missing).")
            else:
                warnings.append(f"'{filename}': Attempting totals-box OCR (totals missing; AutoTrader not detected from text).")
            try:
                extra_at = invoice_ocr_autotrader_costs_box(tmp_pdf)
            except Exception:
                extra_at = {}
            if extra_at:
                for k, v in extra_at.items():
                    if v not in (None, ""):
                        parsed[k] = v
                warnings.append(
                    f"'{filename}': Totals-box OCR extracted std_net={extra_at.get('std_net')}, vat_amount={extra_at.get('vat_amount')}, grand_total={extra_at.get('buying_price')}."
                )
            else:
                warnings.append(f"'{filename}': Totals-box OCR did not extract any amounts.")

        joined_low = "\n".join([clean_text(x) for x in lines]).lower()
        is_used_vehicle_purchase = (
            is_used_vehicle_purchase
            or ("used vehicle purchase invoice" in joined_low)
            or ("vehicle purchase invoice" in joined_low)
            or ("used" in fn_low and "vehicle" in fn_low and "invoice" in fn_low)
        )

        is_bca_like = ("british car auctions" in joined_low) or (
            "document date" in joined_low and bool(re_mod.search(r"\bbca\b", joined_low))
        )
        if is_bca_like and ("used vehicle purchase invoice" not in joined_low and "vehicle purchase invoice" not in joined_low):
            is_used_vehicle_purchase = False

        missing_critical_any = (
            (not clean_text(parsed.get("document_date")))
            or (not clean_text(parsed.get("supplier")))
            or (not clean_text(parsed.get("make")))
            or (not clean_text(parsed.get("reg_no")))
            or (parsed.get("buying_price") in (None, ""))
        )

        if is_used_vehicle_purchase and (not used_ocr) and missing_critical_any:
            lines2, used_ocr2 = invoice_extract_text_lines_from_pdf_with_ocr(tmp_pdf, force_ocr=True)
            if used_ocr2 and lines2:
                parsed2 = extract_invoice_fields(lines2)
                parsed = parsed2
                used_ocr = True
                warnings.append(f"'{filename}': Forced OCR to capture handwritten fields.")

        if is_used_vehicle_purchase:
            extra = invoice_ocr_used_vehicle_purchase_fields(tmp_pdf)
            if extra:
                ddx = clean_text(extra.get("document_date")).replace("-", "/")
                rnx = clean_text(extra.get("reg_no")).upper()
                ok_merge = bool(
                    (is_valid_uk_date(ddx) or re_mod.search(r"\b[A-Z]{2}[0-9O]{2}\s*[A-Z]{3}\b", rnx))
                    and (extra.get("buying_price") not in (None, ""))
                )
                if ok_merge:
                    for k, v in extra.items():
                        if v not in (None, ""):
                            parsed[k] = v
                    warnings.append(f"'{filename}': Applied region OCR for handwritten fields.")

        reg_no_val = clean_text(parsed.get("reg_no")).upper()
        reg_no_vehicle = bool(re_mod.match(r"^[A-Z]{2}\d{2}\s?[A-Z]{3}$", reg_no_val)) if reg_no_val else False
        joined_low2 = "\n".join([clean_text(x) for x in lines]).lower()
        is_bca = ("british car auctions" in joined_low2) or (
            "document date" in joined_low2 and bool(re_mod.search(r"\bbca\b", joined_low2))
        )
        need_bca_ocr = bool(
            used_ocr
            and is_bca
            and (not is_used_vehicle_purchase)
            and (
                (not is_valid_uk_date(parsed.get("document_date")))
                or (parsed.get("buying_price") in (None, ""))
                or (not clean_text(parsed.get("inv_ref_no")))
                or (not clean_text(parsed.get("supplier")))
                or (not clean_text(parsed.get("make")))
                or (not clean_text(parsed.get("reg_no")))
                or reg_no_vehicle
            )
        )
        if need_bca_ocr:
            extra_bca = invoice_ocr_bca_fields(tmp_pdf)
            if extra_bca:
                applied = False
                dd = extra_bca.get("document_date")
                if is_valid_uk_date(dd):
                    parsed["document_date"] = dd
                    applied = True
                ir = clean_text(extra_bca.get("inv_ref_no"))
                if ir:
                    parsed["inv_ref_no"] = ir
                    applied = True
                sup = clean_text(extra_bca.get("supplier"))
                if sup:
                    parsed["supplier"] = sup
                    applied = True
                bp = extra_bca.get("buying_price")
                if isinstance(bp, (int, float)) and 0 < float(bp) < 100000:
                    parsed["buying_price"] = float(bp)
                    parsed["non_vat"] = float(extra_bca.get("non_vat") or bp)
                    applied = True
                mk = clean_text(extra_bca.get("make"))
                if mk:
                    parsed["make"] = mk
                    applied = True
                rn = clean_text(extra_bca.get("reg_no"))
                if rn and rn.upper().startswith("GB"):
                    parsed["reg_no"] = rn
                    applied = True
                elif rn and (not clean_text(parsed.get("reg_no"))):
                    parsed["reg_no"] = rn
                    applied = True
                if applied:
                    warnings.append(f"'{filename}': Applied BCA region OCR for header/total fields.")

        inv_ref_fallback = os.path.splitext(filename)[0]
        inv_ref_value = clean_text(parsed.get("inv_ref_no"))
        if is_used_vehicle_purchase:
            inv_ref_value = inv_ref_fallback
        elif not inv_ref_value:
            inv_ref_value = inv_ref_fallback

        bp_val = parsed.get("buying_price")
        try:
            bp_num = float(bp_val) if bp_val not in (None, "") else None
        except Exception:
            bp_num = None

        if is_used_vehicle_purchase and (bp_num is not None and bp_num > 2500):
            bp_num = None
            parsed["buying_price"] = None
            parsed["non_vat"] = None

        row = invoice_row_from_parsed(sr, parsed, inv_ref_value, used_vehicle_purchase=is_used_vehicle_purchase)
        rows_out.append(row)
        sr += 1

    if not seen_any_pdf:
        msg = "No valid PDF files found"
        if skipped:
            msg += ". Skipped: " + ", ".join(skipped[:20])
        return JSONResponseT({"error": msg}, status_code=400)

    combined_path = os.path.join(job_folder, "combined.csv")
    write_csv(combined_path, rows_out, fieldnames)
    invoice_jobs[job_id] = combined_path

    preview = []
    for r in rows_out[:25]:
        preview.append({k: format_csv_value(r.get(k)) for k in fieldnames})

    return JSONResponseT(
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


async def invoice_convert_review(
    request: Any,
    *,
    output_dir: str,
    invoice_review_jobs: Dict[str, str],
    invoice_jobs: Dict[str, str],
    UploadFileT: Any,
    StarletteUploadFileT: Any,
    clean_text: Any,
    format_csv_value: Any,
    JSONResponseT: Any,
    uuid4: Any,
    invoice_pdf_page_count: Any,
    invoice_extract_text_lines_from_pdf_pages_with_ocr: Any,
    extract_invoice_fields: Any,
    invoice_row_from_parsed: Any,
    invoice_extract_text_lines_from_pdf_with_ocr: Any,
    invoice_tesseract_available: Any,
    Image_available: bool,
    fitz_available: bool,
    pdfium_available: bool,
    OCR_PROVIDER: str,
    deepseek_extract_used_vehicle_fields_from_pdf: Any,
    invoice_ocr_used_vehicle_purchase_fields: Any,
    is_valid_uk_date: Any,
    invoice_ocr_autotrader_costs_box: Any,
    invoice_ocr_bca_fields: Any,
    write_csv: Any,
    write_json: Any,
    to_float_or_none: Any,
    re_mod: Any,
) -> Any:
    # Currently, review endpoint in main.py mirrors /api/convert-invoice output but also writes draft.json.
    # To keep behavior stable, we call invoice_convert(), then persist draft.json using returned preview/fieldnames.
    # Note: We cannot regenerate exact internal row objects from preview, so we rebuild draft from the CSV file.

    resp = await invoice_convert(
        request,
        output_dir=output_dir,
        invoice_jobs=invoice_jobs,
        UploadFileT=UploadFileT,
        StarletteUploadFileT=StarletteUploadFileT,
        clean_text=clean_text,
        format_csv_value=format_csv_value,
        JSONResponseT=JSONResponseT,
        uuid4=uuid4,
        invoice_pdf_page_count=invoice_pdf_page_count,
        invoice_extract_text_lines_from_pdf_pages_with_ocr=invoice_extract_text_lines_from_pdf_pages_with_ocr,
        extract_invoice_fields=extract_invoice_fields,
        invoice_row_from_parsed=invoice_row_from_parsed,
        invoice_extract_text_lines_from_pdf_with_ocr=invoice_extract_text_lines_from_pdf_with_ocr,
        invoice_tesseract_available=invoice_tesseract_available,
        Image_available=Image_available,
        fitz_available=fitz_available,
        pdfium_available=pdfium_available,
        OCR_PROVIDER=OCR_PROVIDER,
        deepseek_extract_used_vehicle_fields_from_pdf=deepseek_extract_used_vehicle_fields_from_pdf,
        invoice_ocr_used_vehicle_purchase_fields=invoice_ocr_used_vehicle_purchase_fields,
        is_valid_uk_date=is_valid_uk_date,
        invoice_ocr_autotrader_costs_box=invoice_ocr_autotrader_costs_box,
        invoice_ocr_bca_fields=invoice_ocr_bca_fields,
        write_csv=write_csv,
        to_float_or_none=to_float_or_none,
        re_mod=re_mod,
    )

    # If invoice_convert returned error (non-200), just pass it through.
    try:
        status_code = getattr(resp, "status_code", 200)
    except Exception:
        status_code = 200
    if status_code >= 400:
        return resp

    try:
        data = resp.body
    except Exception:
        return resp

    # Parse response body JSON
    payload: Any = None
    try:
        import json

        payload = json.loads(data.decode("utf-8"))
    except Exception:
        payload = None

    job_id = payload.get("job_id") if isinstance(payload, dict) else None
    if not job_id:
        return resp

    job_folder = os.path.join(output_dir, str(job_id))
    csv_path = os.path.join(job_folder, "combined.csv")

    # Build draft as simple structure for UI review
    try:
        draft_path = os.path.join(job_folder, "draft.json")
        rows = []
        if os.path.exists(csv_path):
            import csv

            with open(csv_path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row:
                        rows.append(row)
        write_json(draft_path, {"fieldnames": payload.get("fieldnames") or [], "rows": rows})
        invoice_review_jobs[str(job_id)] = draft_path
    except Exception:
        pass

    # Return same payload but include draft_rows if available
    if isinstance(payload, dict) and "draft_rows" not in payload:
        payload["draft_rows"] = []
        try:
            payload["draft_rows"] = rows[:25]
        except Exception:
            payload["draft_rows"] = []
        return JSONResponseT(payload)

    return resp


async def invoice_confirm(
    job_id: str,
    request: Any,
    *,
    invoice_review_jobs: Dict[str, str],
    invoice_jobs: Dict[str, str],
    clean_text: Any,
    to_float_or_none: Any,
    read_json: Any,
    write_json: Any,
    write_csv: Any,
    format_csv_value: Any,
    JSONResponseT: Any,
) -> Any:
    draft_path = invoice_review_jobs.get(job_id)
    if not draft_path or not os.path.exists(draft_path):
        return JSONResponseT({"error": "Invalid or expired job_id"}, status_code=404)

    try:
        payload_in = await request.json()
    except Exception:
        return JSONResponseT({"error": "Invalid JSON body"}, status_code=400)

    rows_in = payload_in.get("rows") if isinstance(payload_in, dict) else None
    if not isinstance(rows_in, list):
        return JSONResponseT({"error": "'rows' must be a list"}, status_code=400)

    try:
        draft = read_json(draft_path)
        fieldnames = draft.get("fieldnames") or []
    except Exception:
        return JSONResponseT({"error": "Draft data could not be read"}, status_code=500)

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

        category = clean_text(r.get("category")) or "purchase"
        document_date = clean_text(r.get("document_date"))
        supplier = clean_text(r.get("supplier"))
        inv_ref_no = clean_text(r.get("inv_ref_no"))
        make = clean_text(r.get("make"))
        model = clean_text(r.get("model"))
        colour = clean_text(r.get("colour"))
        reg_no = clean_text(r.get("reg_no"))

        buying_price_in = r.get("buying_price")
        non_vat_in = r.get("non_vat")
        vat_amount_in = r.get("vat_amount")

        buying_price: Any = "N/A" if clean_text(buying_price_in).upper() in {"N/A", "NA"} else to_float_or_none(buying_price_in)
        non_vat: Any = "N/A" if clean_text(non_vat_in).upper() in {"N/A", "NA"} else to_float_or_none(non_vat_in)

        std_net_raw: Any = r.get("std_net")
        std_net = to_float_or_none(std_net_raw) if clean_text(std_net_raw).upper() not in {"N/A", "NA"} else "N/A"

        vat_amount: Any = "N/A" if clean_text(vat_amount_in).upper() in {"N/A", "NA"} else to_float_or_none(vat_amount_in)

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
    confirmed_path = os.path.join(job_folder, "confirmed.json")
    write_json(confirmed_path, {"fieldnames": fieldnames, "rows": cleaned_rows})

    combined_path = os.path.join(job_folder, "combined.csv")
    write_csv(combined_path, cleaned_rows, fieldnames)

    invoice_jobs[job_id] = combined_path

    preview = []
    for r in cleaned_rows[:25]:
        preview.append({k: format_csv_value(r.get(k)) for k in fieldnames})

    return JSONResponseT(
        {
            "job_id": job_id,
            "rows_total": int(len(cleaned_rows)),
            "fieldnames": fieldnames,
            "preview": preview,
            "download_url": f"/api/invoice-download/{job_id}",
            "confirmed_rows": cleaned_rows,
        }
    )


def invoice_download(job_id: str, *, invoice_jobs: Dict[str, str], FileResponseT: Any) -> Any:
    csv_path = invoice_jobs.get(job_id)
    if not csv_path or not os.path.exists(csv_path):
        return FileResponseT(path="", status_code=404)
    return FileResponseT(csv_path, filename=f"invoices_{job_id}.csv", media_type="text/csv")
