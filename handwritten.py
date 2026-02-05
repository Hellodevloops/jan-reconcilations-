import os

from typing import Any, Dict, List


def extract_generic_receipt_fields(lines: List[str]) -> Dict[str, Any]:
    """Extract fields from any type of receipt (supermarket, parking, restaurant, etc.)"""
    import re

    out: Dict[str, Any] = {}
    if not lines:
        return out

    cleaned = [ln.strip() for ln in lines if ln.strip()]
    joined = "\n".join(cleaned).lower()

    # Detect receipt type based on keywords
    receipt_types = {
        "fuel": ["shell", "bp", "esso", "texaco", "diesel", "petrol", "unleaded", "fuel", "litre", "gas station"],
        "supermarket": ["tesco", "sainsbury", "asda", "morrisons", "aldi", "lidl", "waitrose", "m&s", "marks & spencer", "co-op", "supermarket", "grocery"],
        "pharmacy": ["boots", "lloyds pharmacy", "superdrug", "pharmacy", "chemist", "prescription"],
        "parking": ["parking", "car park", "nps", "cpms", "park & ride", "meter"],
        "restaurant": ["restaurant", "cafe", "coffee", "starbucks", "costa", "pret", "eat", "takeaway", "food"],
        "retail": ["primark", "h&m", "zara", "next", "matalan", "tk maxx", "b&m", "poundland", "wilko"],
        "hardware": ["b&q", "homebase", "wickes", "screwfix", "toolstation", "hardware"],
        "electronics": ["currys", "dixons", "argos", "richer sounds", "maplin", "pc world"],
        "postal": ["post office", "royal mail", "hermes", "dpd", "ups", "fedex", "parcel"],
        "toll": ["dart charge", "toll", "congestion charge", "ulez", "low emission"],
    }

    detected_type = "general"
    for rtype, keywords in receipt_types.items():
        if any(kw in joined for kw in keywords):
            detected_type = rtype
            break

    out["receipt_type"] = detected_type

    # Extract merchant name (usually first few lines)
    for i, ln in enumerate(cleaned[:10]):
        low = ln.lower()
        # Skip common non-merchant lines
        if any(x in low for x in ["receipt", "invoice", "copy", "customer", "tel:", "vat no", "reg no"]):
            continue
        if len(ln.strip()) >= 3:
            out["merchant_name"] = ln.strip()[:200]
            break

    # Extract date - multiple patterns
    date_patterns = [
        r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",
        r"\b(\d{1,2})\s+(\d{1,2})\s+(\d{2,4})\b",
        r"\b(\d{4})[/-](\d{1,2})[/-](\d{1,2})\b",
        r"\b(\d{1,2})\s+([A-Za-z]{3,9})\s+(\d{4})\b",  # 10 Jan 2025
    ]
    for ln in cleaned:
        for pattern in date_patterns:
            m = re.search(pattern, ln, re.IGNORECASE)
            if m:
                date_str = m.group(0)
                date_str = date_str.replace("-", "/")
                out["document_date"] = date_str
                break
        if out.get("document_date"):
            break

    # Extract time
    for ln in cleaned:
        m = re.search(r"\b(\d{1,2}:\d{2}(?::\d{2})?)\b", ln)
        if m:
            out["time"] = m.group(1)
            break

    # Extract total amount (look for total, amount, balance)
    total_keywords = ["total", "amount", "balance", "due", "to pay", "paid", "grand total"]
    for i, ln in enumerate(cleaned):
        low = ln.lower()
        if any(kw in low for kw in total_keywords):
            m = re.search(r"£?\s*(\d+\.\d{2})", ln)
            if m:
                try:
                    total = float(m.group(1))
                    if total > 0:
                        out["total_amount"] = total
                        out["buying_price"] = total
                        out["non_vat"] = total
                        break
                except:
                    pass
            # Check next line
            if i + 1 < len(cleaned):
                m = re.search(r"£?\s*(\d+\.\d{2})", cleaned[i + 1])
                if m:
                    try:
                        total = float(m.group(1))
                        if total > 0:
                            out["total_amount"] = total
                            out["buying_price"] = total
                            out["non_vat"] = total
                            break
                    except:
                        pass

    # Extract VAT
    for ln in cleaned:
        low = ln.lower()
        if "vat" in low or "tax" in low:
            m = re.search(r"£?\s*(\d+\.\d{2})", ln)
            if m:
                try:
                    vat = float(m.group(1))
                    if 0 < vat < out.get("total_amount", 999999):
                        out["vat_amount"] = vat
                        break
                except:
                    pass

    # Extract payment method
    payment_keywords = {
        "VISA": ["visa"],
        "Mastercard": ["mastercard", "master card"],
        "Debit Card": ["debit"],
        "Credit Card": ["credit"],
        "Cash": ["cash"],
        "Apple Pay": ["apple pay"],
        "Google Pay": ["google pay"],
        "Contactless": ["contactless"],
    }
    for ln in cleaned:
        low = ln.lower()
        for method, keywords in payment_keywords.items():
            if any(kw in low for kw in keywords):
                out["payment_method"] = method
                break
        if out.get("payment_method"):
            break

    # Extract receipt/transaction number
    for ln in cleaned:
        low = ln.lower()
        if any(x in low for x in ["receipt", "trans", "ref", "txn"]):
            m = re.search(r"(?:no\.?|num|#|ref)?\s*[:\-]?\s*([A-Z0-9]{4,})", low)
            if m:
                out["receipt_no"] = m.group(1).upper()
                break
            m = re.search(r"\b(\d{4,})\b", low)
            if m:
                out["receipt_no"] = m.group(1)
                break

    # For fuel receipts, also try to extract litres and price per litre
    if detected_type == "fuel":
        for ln in cleaned:
            m = re.search(r"(\d+\.?\d*)\s*(?:litres?|ltr?|l)\b", ln.lower())
            if m:
                try:
                    out["quantity_litres"] = float(m.group(1))
                    break
                except:
                    pass

        for ln in cleaned:
            m = re.search(r"£?\s*(\d+\.\d{2,3})\s*(?:/|per|@).*?(?:l|litre)", ln.lower())
            if m:
                try:
                    price = float(m.group(1))
                    if 0.5 < price < 5:
                        out["price_per_litre"] = price
                        break
                except:
                    pass

    # Set category based on receipt type
    category_map = {
        "fuel": "fuel_expense",
        "supermarket": "grocery_expense",
        "pharmacy": "medical_expense",
        "parking": "parking_expense",
        "restaurant": "meal_expense",
        "retail": "shopping_expense",
        "hardware": "maintenance_expense",
        "electronics": "equipment_expense",
        "postal": "postage_expense",
        "toll": "toll_expense",
        "general": "expense",
    }
    out["category"] = category_map.get(detected_type, "expense")

    return out


def extract_fuel_receipt_fields(lines: List[str]) -> Dict[str, Any]:
    """Extract fields from fuel receipts (Shell, etc.)"""
    import re

    out: Dict[str, Any] = {}
    if not lines:
        return out

    cleaned = [ln.strip() for ln in lines if ln.strip()]
    joined = "\n".join(cleaned).lower()

    # Detect if this is a fuel receipt
    is_fuel_receipt = any(x in joined for x in [
        "shell", "diesel", "petrol", "fuel", "pump", "litre", "liter",
        "bp ", "esso", "texaco", "customer receipt"
    ])
    if not is_fuel_receipt:
        return out

    out["receipt_type"] = "fuel"

    # Extract station name - look for Shell locations
    for i, ln in enumerate(cleaned[:20]):
        low = ln.lower()
        if "shell" in low:
            # Get station name from this line or next few lines
            station_parts = []
            for j in range(i, min(i + 5, len(cleaned))):
                part = cleaned[j].strip()
                if part and not any(x in part.lower() for x in ["vat", "telephone", "tel", "receipt", "total"]):
                    station_parts.append(part)
            if station_parts:
                out["station_name"] = " | ".join(station_parts)[:200]
            break

    # If no shell found, look for other station indicators
    if not out.get("station_name"):
        for i, ln in enumerate(cleaned[:15]):
            if any(x in ln.lower() for x in ["high street", "high road", "roundabout", "service station"]):
                # Look backwards for station name
                for j in range(max(0, i - 3), i + 1):
                    part = cleaned[j].strip()
                    if len(part) > 3 and not any(x in part.lower() for x in ["date", "time", "receipt"]):
                        out["station_name"] = part[:200]
                        break
                break

    # Extract date - multiple formats
    date_patterns = [
        r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",  # 10/01/25 or 10-01-2025
        r"\b(\d{1,2})\s+(\d{1,2})\s+(\d{2,4})\b",  # 10 01 25
        r"\b(\d{4})[/-](\d{1,2})[/-](\d{1,2})\b",  # 2025-01-10
    ]
    for ln in cleaned:
        for pattern in date_patterns:
            m = re.search(pattern, ln)
            if m:
                date_str = m.group(0)
                # Normalize to dd/mm/yyyy
                date_str = date_str.replace("-", "/")
                out["document_date"] = date_str
                break
        if out.get("document_date"):
            break

    # Extract time
    for ln in cleaned:
        m = re.search(r"\b(\d{1,2}:\d{2}(?::\d{2})?)\b", ln)
        if m:
            out["time"] = m.group(1)
            break

    # Extract fuel type (Diesel/Petrol)
    for ln in cleaned:
        low = ln.lower()
        if "diesel" in low:
            out["fuel_type"] = "Diesel"
            break
        elif "petrol" in low or "gasoline" in low:
            out["fuel_type"] = "Petrol"
            break
        elif "unleaded" in low:
            out["fuel_type"] = "Unleaded"
            break

    # Extract quantity (litres) - look for patterns like "13.94 litres" or "58.14 @"
    for ln in cleaned:
        # Pattern: number followed by litres or l
        m = re.search(r"(\d+\.?\d*)\s*(?:litres?|ltr?|l)\b", ln.lower())
        if m:
            try:
                out["quantity_litres"] = float(m.group(1))
                break
            except:
                pass
        # Pattern: "@ £1.439/l" or similar with preceding quantity
        m = re.search(r"(\d+\.?\d*)\s*@", ln)
        if m:
            val = float(m.group(1))
            if val > 5 and val < 200:  # Reasonable fuel quantity
                out["quantity_litres"] = val
                break

    # Extract price per litre
    for ln in cleaned:
        low = ln.lower()
        # Pattern: £1.439/l or £1.439/litre or @ £1.439
        m = re.search(r"£?\s*(\d+\.\d{2,3})\s*(?:/|per|@).*?(?:l|litre)", low)
        if m:
            try:
                price = float(m.group(1))
                if 0.5 < price < 5:  # Reasonable fuel price per litre
                    out["price_per_litre"] = price
                    break
            except:
                pass
        # Alternative: look for rate after @
        m = re.search(r"@\s*£?\s*(\d+\.\d{2,3})", low)
        if m:
            try:
                price = float(m.group(1))
                if 0.5 < price < 5:
                    out["price_per_litre"] = price
                    break
            except:
                pass

    # Extract total amount (look for TOTAL line)
    for i, ln in enumerate(cleaned):
        low = ln.lower()
        if "total" in low and any(x in low for x in ["£", "gbp"]):
            m = re.search(r"£?\s*(\d+\.\d{2})", ln)
            if m:
                try:
                    total = float(m.group(1))
                    if total > 5:  # Reasonable fuel purchase
                        out["total_amount"] = total
                        out["buying_price"] = total  # For CSV compatibility
                        out["non_vat"] = total
                        break
                except:
                    pass
            # Check next line if current line is just "TOTAL"
            if i + 1 < len(cleaned):
                m = re.search(r"£?\s*(\d+\.\d{2})", cleaned[i + 1])
                if m:
                    try:
                        total = float(m.group(1))
                        if total > 5:
                            out["total_amount"] = total
                            out["buying_price"] = total
                            out["non_vat"] = total
                            break
                    except:
                        pass

    # Extract VAT amount
    for ln in cleaned:
        low = ln.lower()
        if "vat" in low or "tax" in low:
            m = re.search(r"£?\s*(\d+\.\d{2})", ln)
            if m:
                try:
                    vat = float(m.group(1))
                    if 0 < vat < 50:  # Reasonable VAT for fuel
                        out["vat_amount"] = vat
                        break
                except:
                    pass

    # Extract payment method
    for ln in cleaned:
        low = ln.lower()
        if "visa" in low:
            out["payment_method"] = "VISA"
            break
        elif "mastercard" in low or "master card" in low:
            out["payment_method"] = "Mastercard"
            break
        elif "debit" in low:
            out["payment_method"] = "Debit Card"
            break
        elif "credit" in low:
            out["payment_method"] = "Credit Card"
            break
        elif "cash" in low:
            out["payment_method"] = "Cash"
            break
        elif "contactless" in low:
            out["payment_method"] = out.get("payment_method", "") + " Contactless"
            break

    # Extract receipt/transaction number
    for ln in cleaned:
        low = ln.lower()
        if "receipt" in low and any(x in low for x in ["no", "number", ":", "#"]):
            m = re.search(r"(?:no\.?|number|#)?\s*[:\-]?\s*(\d{4,})", low)
            if m:
                out["receipt_no"] = m.group(1)
                break
        elif "transaction" in low and any(x in low for x in ["no", "number", ":"]):
            m = re.search(r"(?:no\.?|number)?\s*[:\-]?\s*(\d{4,})", low)
            if m:
                out["transaction_no"] = m.group(1)
                break

    # Extract terminal ID
    for ln in cleaned:
        low = ln.lower()
        if "terminal" in low and "id" in low:
            m = re.search(r"(?:id)?\s*[:\-]?\s*(\d{4,})", low)
            if m:
                out["terminal_id"] = m.group(1)
                break

    # Set category
    out["category"] = "fuel_expense"

    return out


async def handwritten_invoice_convert_review(
    request: Any,
    *,
    output_dir: str,
    handwritten_review_jobs: Dict[str, str],
    handwritten_jobs: Dict[str, str],
    UploadFileT: Any,
    StarletteUploadFileT: Any,
    clean_text: Any,
    JSONResponseT: Any,
    uuid4: Any,
    MODEL_REGISTRY: Dict[str, Dict[str, Any]],
    LIGHTON_LOCAL_MODEL_NAME: str,
    handwritten_lighton_multimodel_ocr: Any,
    parse_bbox_output: Any,
    clean_output_text: Any,
    ocr_words_and_lines_from_pil_image: Any,
    crop_from_bbox: Any,
    extract_invoice_fields: Any,
    extract_used_vehicle_invoice_fields: Any,
    invoice_render_page: Any,
    invoice_render_first_page: Any,
    ImageT: Any,
    io_mod: Any,
    base64_mod: Any,
    re_mod: Any,
    write_json: Any,
    write_csv: Any,
    format_csv_value: Any,
) -> Any:
    form = await request.form()

    model_name_in = clean_text(form.get("model_name") if hasattr(form, "get") else "")
    if not model_name_in:
        model_name_in = clean_text(form.get("model") if hasattr(form, "get") else "")

    model_name = (
        model_name_in
        if model_name_in in MODEL_REGISTRY
        else (
            LIGHTON_LOCAL_MODEL_NAME
            if (LIGHTON_LOCAL_MODEL_NAME in MODEL_REGISTRY)
            else "LightOnOCR-2-1B (Best OCR)"
        )
    )

    try:
        temperature = float(clean_text(form.get("temperature") if hasattr(form, "get") else "") or "0.2")
    except Exception:
        temperature = 0.2

    try:
        max_tokens = int(
            clean_text(form.get("max_tokens") if hasattr(form, "get") else "")
            or clean_text(form.get("max_output_tokens") if hasattr(form, "get") else "")
            or "2048"
        )
    except Exception:
        max_tokens = 2048

    try:
        page_num = int(clean_text(form.get("page_num") if hasattr(form, "get") else "") or "1")
    except Exception:
        page_num = 1

    include_crops = clean_text(form.get("include_crops") if hasattr(form, "get") else "").strip().lower() in {"1", "true", "yes", "y", "on"}

    multi_receipts = clean_text(form.get("multi_receipts") if hasattr(form, "get") else "").strip().lower() in {"1", "true", "yes", "y", "on"}

    include_ocr_items = clean_text(form.get("include_ocr_items") if hasattr(form, "get") else "").strip().lower() in {"1", "true", "yes", "y", "on"}

    try:
        receipts_limit = int(clean_text(form.get("receipts_limit") if hasattr(form, "get") else "") or "10")
    except Exception:
        receipts_limit = 10

    try:
        pdf_pages_limit = int(clean_text(form.get("pdf_pages_limit") if hasattr(form, "get") else "") or "1")
    except Exception:
        pdf_pages_limit = 1
    pdf_pages_limit = max(1, min(int(pdf_pages_limit), 200))

    try:
        crops_limit = int(clean_text(form.get("crops_limit") if hasattr(form, "get") else "") or "25")
    except Exception:
        crops_limit = 25

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

    outputs: List[Dict[str, Any]] = []

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
    skipped: List[str] = []

    category_value = "purchase"

    for f in files:
        filename = os.path.basename(getattr(f, "filename", None) or "handwritten")
        content_type = (getattr(f, "content_type", None) or "").lower()

        is_pdf = filename.lower().endswith(".pdf") or content_type == "application/pdf"
        is_image = content_type.startswith("image/") or filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))

        if not (is_pdf or is_image):
            skipped.append(f"{filename} ({content_type or 'unknown'})")
            continue

        content = await f.read()
        if not content:
            warnings.append(f"'{filename}': Empty file.")
            continue

        pil_img: Any = None
        png_bytes: bytes = b""
        pdf_pages: List[Dict[str, Any]] = []

        if is_image:
            if ImageT is not None:
                try:
                    pil_img = ImageT.open(io_mod.BytesIO(content)).convert("RGB")
                    buf0 = io_mod.BytesIO()
                    pil_img.save(buf0, format="PNG")
                    png_bytes = buf0.getvalue()
                    print(f"DEBUG: Image converted to PNG, size: {len(png_bytes)} bytes")
                except Exception as e:
                    print(f"DEBUG: Image conversion failed: {e}")
                    pil_img = None
                    png_bytes = b""
        else:
            tmp_pdf = os.path.join(job_folder, filename)
            try:
                with open(tmp_pdf, "wb") as out:
                    out.write(content)
            except Exception:
                tmp_pdf = ""

            if tmp_pdf and os.path.exists(tmp_pdf):
                wanted_pages = 1
                if multi_receipts:
                    wanted_pages = int(pdf_pages_limit)

                for pidx in range(1, wanted_pages + 1):
                    page_img: Any = None
                    try:
                        print(f"DEBUG: Attempting to render PDF page: {pidx}")
                        page_img = invoice_render_page(tmp_pdf, pidx)
                        print(f"DEBUG: PDF render result: {page_img is not None}")
                    except Exception as e:
                        print(f"DEBUG: PDF render failed: {e}")
                        page_img = None

                    if page_img is None and pidx == 1:
                        try:
                            print("DEBUG: Trying first page render")
                            page_img = invoice_render_first_page(tmp_pdf)
                            print(f"DEBUG: First page render result: {page_img is not None}")
                        except Exception as e:
                            print(f"DEBUG: First page render failed: {e}")
                            page_img = None

                    if page_img is None:
                        if pidx == 1:
                            pil_img = None
                        break

                    try:
                        buf1 = io_mod.BytesIO()
                        page_img.save(buf1, format="PNG")
                        page_png = buf1.getvalue()
                        print(f"DEBUG: PDF converted to PNG, size: {len(page_png)} bytes")
                    except Exception as e:
                        print(f"DEBUG: PDF to PNG conversion failed: {e}")
                        page_png = b""

                    pdf_pages.append({"page_index": int(pidx), "pil_img": page_img, "png_bytes": page_png})

                if pdf_pages:
                    pil_img = pdf_pages[0].get("pil_img")
                    png_bytes = pdf_pages[0].get("png_bytes") or b""

        def _run_single_ocr(pil_img_local: Any, png_bytes_local: bytes) -> Dict[str, Any]:
            raw_output_local = ""
            cleaned_text_local = ""
            detections_local: List[Dict[str, Any]] = []
            engine_local = ""
            lighton_error_local = ""
            easyocr_error_local = ""
            tesseract_error_local = ""

            has_bbox_local = bool((MODEL_REGISTRY.get(model_name) or {}).get("has_bbox"))

            if png_bytes_local:
                try:
                    raw_output_local = handwritten_lighton_multimodel_ocr(
                        png_bytes_local,
                        model_name=model_name,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                except Exception as e:
                    lighton_error_local = str(e)
                    raw_output_local = ""

                if raw_output_local:
                    engine_local = "lighton"
                    if has_bbox_local:
                        try:
                            cleaned_text_local, detections_local = parse_bbox_output(raw_output_local)
                        except Exception:
                            detections_local = []
                            try:
                                cleaned_text_local = clean_output_text(raw_output_local)
                            except Exception:
                                cleaned_text_local = ""
                    else:
                        try:
                            cleaned_text_local = clean_output_text(raw_output_local)
                        except Exception:
                            cleaned_text_local = ""

            easyocr_items_local: List[Dict[str, Any]] = []
            if not cleaned_text_local and pil_img_local is not None:
                try:
                    import numpy as np
                    import easyocr
                except Exception as e:
                    easyocr_error_local = str(e)
                    np = None  # type: ignore
                    easyocr = None  # type: ignore

                if ("np" in locals()) and ("easyocr" in locals()) and np is not None and easyocr is not None:
                    img_arr = np.array(pil_img_local)

                    def _to_py_jsonable(val: Any) -> Any:
                        if val is None or isinstance(val, (str, int, float, bool)):
                            return val
                        if hasattr(val, "tolist") and callable(getattr(val, "tolist")):
                            try:
                                return _to_py_jsonable(val.tolist())
                            except Exception:
                                pass
                        if hasattr(val, "item") and callable(getattr(val, "item")):
                            try:
                                return _to_py_jsonable(val.item())
                            except Exception:
                                pass
                        if isinstance(val, dict):
                            out_d: Dict[str, Any] = {}
                            for k, v in val.items():
                                out_d[str(k)] = _to_py_jsonable(v)
                            return out_d
                        if isinstance(val, (list, tuple)):
                            return [_to_py_jsonable(x) for x in val]
                        return str(val)

                    def _run_easyocr(reader_obj: Any) -> None:
                        nonlocal cleaned_text_local, engine_local, raw_output_local, easyocr_items_local
                        result = reader_obj.readtext(img_arr, detail=1, paragraph=False)
                        words_tmp: List[str] = []
                        for item in result or []:
                            try:
                                bbox, text, confidence = item
                            except Exception:
                                continue
                            txt = clean_text(text)
                            try:
                                conf_f = float(confidence)
                            except Exception:
                                conf_f = 0.0
                            if txt:
                                words_tmp.append(txt)
                                easyocr_items_local.append({"text": txt, "confidence": float(conf_f), "bbox": _to_py_jsonable(bbox)})

                        if words_tmp:
                            cleaned_text_local = " ".join(words_tmp)
                            engine_local = engine_local or "easyocr"
                            if not raw_output_local:
                                raw_output_local = cleaned_text_local

                    try:
                        reader_gpu = getattr(handwritten_invoice_convert_review, "_easyocr_reader_gpu", None)
                        if reader_gpu is None:
                            reader_gpu = easyocr.Reader(["en"], gpu=True)
                            setattr(handwritten_invoice_convert_review, "_easyocr_reader_gpu", reader_gpu)
                        _run_easyocr(reader_gpu)
                    except Exception as e:
                        easyocr_error_local = str(e)

                    if not cleaned_text_local:
                        try:
                            reader_cpu = getattr(handwritten_invoice_convert_review, "_easyocr_reader_cpu", None)
                            if reader_cpu is None:
                                reader_cpu = easyocr.Reader(["en"], gpu=False)
                                setattr(handwritten_invoice_convert_review, "_easyocr_reader_cpu", reader_cpu)
                            _run_easyocr(reader_cpu)
                            if cleaned_text_local:
                                easyocr_error_local = ""
                        except Exception as e:
                            if not easyocr_error_local:
                                easyocr_error_local = str(e)

            if not cleaned_text_local:
                if pil_img_local is not None:
                    try:
                        t_ocr = ocr_words_and_lines_from_pil_image(pil_img_local)
                    except Exception:
                        tesseract_error_local = "tesseract_failed"
                        t_ocr = {"lines": [], "words": []}

                    if (t_ocr.get("lines") or t_ocr.get("words")) and isinstance(t_ocr, dict):
                        engine_local = engine_local or "tesseract"
                        all_words2: List[str] = []
                        for word_data in t_ocr.get("words") or []:
                            if isinstance(word_data, dict) and word_data.get("text"):
                                all_words2.append(clean_text(word_data.get("text")))

                        for line_data in t_ocr.get("lines") or []:
                            if isinstance(line_data, dict) and line_data.get("text"):
                                all_words2.extend(clean_text(line_data.get("text")).split())

                        cleaned_text_local = " ".join([w for w in all_words2 if w])

            lines_local = [x for x in [clean_text(x) for x in str(cleaned_text_local or "").splitlines()] if x]
            words_local = [x for x in re_mod.split(r"\s+", clean_text(cleaned_text_local)) if x]
            all_words_local = [x.strip() for x in re_mod.split(r"\s+", clean_text(cleaned_text_local)) if x.strip()]

            return {
                "engine": engine_local,
                "raw_output": raw_output_local,
                "cleaned_text": cleaned_text_local,
                "lines": [{"index": i + 1, "text": ln} for i, ln in enumerate(lines_local)],
                "words": [{"index": i + 1, "text": w} for i, w in enumerate(words_local)],
                "all_words": all_words_local,
                "ocr_items": easyocr_items_local,
                "detections": detections_local,
                "ocr_debug": {
                    "lighton_error": lighton_error_local,
                    "easyocr_error": easyocr_error_local,
                    "tesseract_error": tesseract_error_local,
                    "easyocr_items": int(len(easyocr_items_local)),
                    "cleaned_len": int(len(cleaned_text_local or "")),
                },
            }

        receipts_payloads: List[Dict[str, Any]] = []
        segmentation_debug_pages: List[Dict[str, Any]] = []

        pages_to_process: List[Dict[str, Any]] = []
        if is_pdf and pdf_pages:
            for pp in pdf_pages:
                if isinstance(pp, dict) and pp.get("pil_img") is not None:
                    pages_to_process.append(pp)
        else:
            pages_to_process = [{"page_index": 1, "pil_img": pil_img, "png_bytes": png_bytes}]

        for pp in pages_to_process:
            page_index = int(pp.get("page_index") or 1)
            page_img = pp.get("pil_img")
            page_png = pp.get("png_bytes") or b""

            if not multi_receipts or page_img is None:
                ocr_res = _run_single_ocr(page_img, page_png)
                receipts_payloads.append({"index": int(len(receipts_payloads) + 1), "page_index": page_index, "bbox": None, **ocr_res})
                continue

            regions: List[Dict[str, Any]] = []
            segmentation_debug: Dict[str, Any] = {"enabled": True, "method": "connected_components", "regions": [], "page_index": page_index}
            try:
                import numpy as np
                import cv2

                img_np = cv2.cvtColor(np.array(page_img), cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
                h, w = gray.shape[:2]

                segmentation_debug["image_wh"] = [int(w), int(h)]

                blur = cv2.GaussianBlur(gray, (3, 3), 0)
                _t, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                kx = max(7, int(w * 0.01))
                ky = max(7, int(h * 0.01))
                kx = min(kx, 25)
                ky = min(ky, 25)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
                mor = cv2.morphologyEx(thr, cv2.MORPH_DILATE, kernel, iterations=1)

                num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mor, connectivity=8)

                min_area = int(0.015 * float(w * h))
                candidates: List[Dict[str, Any]] = []
                for i in range(1, int(num_labels)):
                    x = int(stats[i, cv2.CC_STAT_LEFT])
                    y = int(stats[i, cv2.CC_STAT_TOP])
                    ww = int(stats[i, cv2.CC_STAT_WIDTH])
                    hh = int(stats[i, cv2.CC_STAT_HEIGHT])
                    area = int(stats[i, cv2.CC_STAT_AREA])
                    if area < min_area:
                        continue
                    if ww < 120 or hh < 120:
                        continue
                    bbox_area = int(ww * hh)
                    if bbox_area < min_area:
                        continue
                    candidates.append({"bbox": [x, y, x + ww, y + hh], "area": bbox_area})

                def _iou(a: List[int], b: List[int]) -> float:
                    ax1, ay1, ax2, ay2 = a
                    bx1, by1, bx2, by2 = b
                    ix1 = max(ax1, bx1)
                    iy1 = max(ay1, by1)
                    ix2 = min(ax2, bx2)
                    iy2 = min(ay2, by2)
                    iw = max(0, ix2 - ix1)
                    ih = max(0, iy2 - iy1)
                    inter = float(iw * ih)
                    if inter <= 0:
                        return 0.0
                    a_area = float(max(0, ax2 - ax1) * max(0, ay2 - ay1))
                    b_area = float(max(0, bx2 - bx1) * max(0, by2 - by1))
                    denom = a_area + b_area - inter
                    return float(inter / denom) if denom > 0 else 0.0

                def _merge_boxes(boxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                    merged = True
                    out = list(boxes)
                    while merged:
                        merged = False
                        new_out: List[Dict[str, Any]] = []
                        used = [False] * len(out)
                        for i in range(len(out)):
                            if used[i]:
                                continue
                            a = out[i]
                            ax1, ay1, ax2, ay2 = [int(v) for v in (a.get("bbox") or [0, 0, 0, 0])]
                            for j in range(i + 1, len(out)):
                                if used[j]:
                                    continue
                                b = out[j]
                                bb = b.get("bbox") or [0, 0, 0, 0]
                                bx1, by1, bx2, by2 = [int(v) for v in bb]
                                iou = _iou([ax1, ay1, ax2, ay2], [bx1, by1, bx2, by2])
                                gap_x = max(0, max(bx1 - ax2, ax1 - bx2))
                                gap_y = max(0, max(by1 - ay2, ay1 - by2))
                                close = (gap_x <= int(0.02 * w) and gap_y <= int(0.02 * h))
                                if iou >= 0.15 or close:
                                    ax1 = min(ax1, bx1)
                                    ay1 = min(ay1, by1)
                                    ax2 = max(ax2, bx2)
                                    ay2 = max(ay2, by2)
                                    used[j] = True
                                    merged = True
                            used[i] = True
                            area2 = int(max(0, ax2 - ax1) * max(0, ay2 - ay1))
                            new_out.append({"bbox": [int(ax1), int(ay1), int(ax2), int(ay2)], "area": int(area2)})
                        out = new_out
                    return out

                merged_boxes = _merge_boxes(candidates)
                merged_boxes.sort(key=lambda r: (int((r.get("bbox") or [0, 0, 0, 0])[1]), int((r.get("bbox") or [0, 0, 0, 0])[0])))
                regions = merged_boxes[: max(1, int(receipts_limit))]
                segmentation_debug["candidates"] = int(len(candidates))
                segmentation_debug["regions_count"] = int(len(regions))
                segmentation_debug["regions"] = [
                    {"bbox": [int(v) for v in (r.get("bbox") or [0, 0, 0, 0])], "area": int(r.get("area") or 0)}
                    for r in regions
                ]
            except Exception as e:
                regions = []
                segmentation_debug["error"] = str(e)

            if not regions:
                ww, hh = page_img.size
                regions = [{"bbox": [0, 0, int(ww), int(hh)], "area": int(ww * hh)}]
                segmentation_debug["regions_count"] = 1
                segmentation_debug["regions"] = [{"bbox": [0, 0, int(ww), int(hh)], "area": int(ww * hh)}]

            segmentation_debug_pages.append(segmentation_debug)

            for reg in regions:
                bbox = reg.get("bbox") or [0, 0, 0, 0]
                try:
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                except Exception:
                    x1, y1, x2, y2 = 0, 0, page_img.size[0], page_img.size[1]

                try:
                    crop_img = page_img.crop((x1, y1, x2, y2))
                except Exception:
                    crop_img = page_img

                try:
                    bbuf = io_mod.BytesIO()
                    crop_img.save(bbuf, format="PNG")
                    crop_png = bbuf.getvalue()
                except Exception:
                    crop_png = page_png

                ocr_res = _run_single_ocr(crop_img, crop_png)
                receipts_payloads.append({"index": int(len(receipts_payloads) + 1), "page_index": page_index, "bbox": [int(x1), int(y1), int(x2), int(y2)], **ocr_res})

        joined_text = "\n\n".join([clean_text(r.get("cleaned_text")) for r in receipts_payloads if isinstance(r, dict)])
        cleaned_text = joined_text
        raw_output = "\n\n".join([clean_text(r.get("raw_output")) for r in receipts_payloads if isinstance(r, dict)])
        engine = clean_text(receipts_payloads[0].get("engine") if receipts_payloads else "")
        detections: List[Dict[str, Any]] = []
        crops: List[Dict[str, Any]] = []
        has_bbox = bool((MODEL_REGISTRY.get(model_name) or {}).get("has_bbox"))
        easyocr_items: List[Dict[str, Any]] = []
        lighton_error = ""
        easyocr_error = ""
        tesseract_error = ""
        for rr in receipts_payloads:
            if isinstance(rr, dict):
                for it in rr.get("ocr_items") or []:
                    if isinstance(it, dict):
                        easyocr_items.append(it)
                dbg = rr.get("ocr_debug") or {}
                if isinstance(dbg, dict):
                    lighton_error = lighton_error or clean_text(dbg.get("lighton_error"))
                    easyocr_error = easyocr_error or clean_text(dbg.get("easyocr_error"))
                    tesseract_error = tesseract_error or clean_text(dbg.get("tesseract_error"))

        if not include_ocr_items:
            for rr in receipts_payloads:
                if isinstance(rr, dict):
                    rr["ocr_items"] = []
            easyocr_items = []

        if include_crops and has_bbox and detections and pil_img is not None:
            for det in detections[: max(0, crops_limit)]:
                crop_img = crop_from_bbox(pil_img, det)
                if crop_img is None:
                    continue
                try:
                    b = io_mod.BytesIO()
                    crop_img.save(b, format="PNG")
                    crops.append(
                        {
                            "ref": det.get("ref"),
                            "coords": det.get("coords"),
                            "png_base64": base64_mod.b64encode(b.getvalue()).decode("ascii"),
                        }
                    )
                except Exception:
                    continue

        lines_out = [x for x in [clean_text(x) for x in str(cleaned_text or "").splitlines()] if x]
        words_out = [x for x in re_mod.split(r"\s+", clean_text(cleaned_text)) if x]

        all_words_list = [x.strip() for x in re_mod.split(r"\s+", clean_text(cleaned_text)) if x.strip()]

        payload_out: Dict[str, Any] = {
            "filename": filename,
            "content_type": content_type,
            "model_name": model_name,
            "has_bbox": bool(has_bbox),
            "engine": engine or "",
            "raw_output": raw_output,
            "cleaned_text": cleaned_text,
            "lines": [{"index": i + 1, "text": ln} for i, ln in enumerate(lines_out)],
            "words": [{"index": i + 1, "text": w} for i, w in enumerate(words_out)],
            "all_words": all_words_list,
            "ocr_items": easyocr_items,
            "ocr_debug": {
                "lighton_error": lighton_error,
                "easyocr_error": easyocr_error,
                "tesseract_error": tesseract_error,
                "lighton_raw_len": int(len(raw_output or "")) if (engine == "lighton") else 0,
                "easyocr_items": int(len(easyocr_items)),
                "cleaned_len": int(len(cleaned_text or "")),
            },
            "detections": detections,
        }

        payload_out["receipts"] = receipts_payloads
        payload_out["receipts_count"] = int(len(receipts_payloads))
        try:
            receipts_summary: List[Dict[str, Any]] = []
            for rr in receipts_payloads:
                if not isinstance(rr, dict):
                    continue
                txt = clean_text(rr.get("cleaned_text"))
                receipts_summary.append(
                    {
                        "index": int(rr.get("index") or 0) if str(rr.get("index") or "").strip() else 0,
                        "page_index": int(rr.get("page_index") or 1),
                        "bbox": rr.get("bbox"),
                        "engine": clean_text(rr.get("engine")),
                        "text_len": int(len(txt)),
                        "words": int(len(rr.get("all_words") or [])) if isinstance(rr.get("all_words"), list) else 0,
                        "preview": txt[:240],
                    }
                )
            payload_out["receipts_summary"] = receipts_summary
        except Exception:
            payload_out["receipts_summary"] = []
        if multi_receipts:
            try:
                payload_out["segmentation_debug"] = segmentation_debug_pages
            except Exception:
                payload_out["segmentation_debug"] = {"enabled": True}

        if include_crops:
            payload_out["crops"] = crops

        if cleaned_text:
            warnings.append(f"'{filename}': Extracted handwritten OCR text.")
        else:
            warnings.append(f"'{filename}': No OCR text could be extracted.")

        outputs.append(payload_out)

        for rr in receipts_payloads:
            rr_lines = []
            if isinstance(rr, dict):
                rr_lines = [x for x in [clean_text(x) for x in str(rr.get("cleaned_text") or "").splitlines()] if x]
            extracted: Dict[str, Any] = {}
            try:
                extracted = extract_invoice_fields(rr_lines) if rr_lines else {}
            except Exception:
                extracted = {}

            joined_low = "\n".join(rr_lines).lower()
            used_vehicle_invoice = bool("used vehicle invoice" in joined_low)
            
            # Check if it looks like any type of receipt
            is_receipt = bool(
                any(x in joined_low for x in [
                    "receipt", "customer copy", "total", "visa", "mastercard", "cash",
                    "shell", "tesco", "boots", "parking", "restaurant", "cafe"
                ])
            )
            
            if used_vehicle_invoice:
                try:
                    uv = extract_used_vehicle_invoice_fields(rr_lines)
                except Exception:
                    uv = {}
                for k, v in uv.items():
                    if v not in (None, ""):
                        extracted[k] = v
            elif is_receipt:
                # Use generic receipt extractor for any receipt type
                try:
                    receipt_data = extract_generic_receipt_fields(rr_lines)
                except Exception:
                    receipt_data = {}
                for k, v in receipt_data.items():
                    if v not in (None, ""):
                        extracted[k] = v
                # If it's specifically a fuel receipt, get more details
                if extracted.get("receipt_type") == "fuel":
                    try:
                        fuel_data = extract_fuel_receipt_fields(rr_lines)
                    except Exception:
                        fuel_data = {}
                    for k, v in fuel_data.items():
                        if v not in (None, ""):
                            extracted[k] = v

            buying_price_ex = extracted.get("buying_price")
            non_vat_ex = extracted.get("non_vat")
            if buying_price_ex not in (None, "") and non_vat_ex in (None, ""):
                extracted["non_vat"] = buying_price_ex

            row = {
                "sr_no": int(len(rows_out) + 1),
                "category": extracted.get("category") or category_value,
                "document_date": clean_text(extracted.get("document_date")),
                "supplier": clean_text(extracted.get("supplier") or extracted.get("station_name") or extracted.get("merchant_name")),
                "inv_ref_no": clean_text(extracted.get("inv_ref_no") or extracted.get("receipt_no") or extracted.get("transaction_no")),
                "make": clean_text(extracted.get("make") or extracted.get("fuel_type")),
                "model": clean_text(extracted.get("model")),
                "colour": clean_text(extracted.get("colour")),
                "reg_no": clean_text(extracted.get("reg_no")) or "N/A",
                "buying_price": extracted.get("buying_price") if extracted.get("buying_price") not in (None, "") else "N/A",
                "non_vat": extracted.get("non_vat") if extracted.get("non_vat") not in (None, "") else "N/A",
                "std_net": extracted.get("std_net") if extracted.get("std_net") not in (None, "") else "N/A",
                "vat_amount": extracted.get("vat_amount") if extracted.get("vat_amount") not in (None, "") else "N/A",
            }

            rows_out.append(row)

    if not outputs:
        msg = "No valid handwritten invoice files found"
        if skipped:
            msg += ". Skipped: " + ", ".join(skipped[:20])
        return JSONResponseT({"error": msg, "warnings": warnings, "skipped": skipped}, status_code=400)

    combined_path = os.path.join(job_folder, "combined.csv")
    try:
        write_csv(combined_path, rows_out, fieldnames)
        handwritten_jobs[job_id] = combined_path
    except Exception:
        pass

    draft_path = os.path.join(job_folder, "draft.json")
    try:
        write_json(draft_path, {"fieldnames": fieldnames, "rows": rows_out, "outputs": outputs})
        handwritten_review_jobs[job_id] = draft_path
    except Exception:
        pass

    preview = []
    for r in rows_out[:25]:
        preview.append({k: format_csv_value(r.get(k)) for k in fieldnames})

    return JSONResponseT(
        {
            "job_id": job_id,
            "files_total": int(len(files)),
            "files_output": int(len(outputs)),
            "warnings": warnings,
            "skipped": skipped,
            "fieldnames": fieldnames,
            "preview": preview,
            "draft_rows": rows_out,
            "confirm_url": f"/api/handwritten-invoice-confirm/{job_id}",
            "download_url": f"/api/handwritten-invoice-download/{job_id}",
            "outputs": outputs,
        }
    )


async def handwritten_invoice_confirm(
    job_id: str,
    request: Any,
    *,
    handwritten_review_jobs: Dict[str, str],
    handwritten_jobs: Dict[str, str],
    clean_text: Any,
    to_float_or_none: Any,
    read_json: Any,
    write_json: Any,
    write_csv: Any,
    format_csv_value: Any,
    JSONResponseT: Any,
) -> Any:
    draft_path = handwritten_review_jobs.get(job_id)
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

    handwritten_jobs[job_id] = combined_path

    preview = []
    for r in cleaned_rows[:25]:
        preview.append({k: format_csv_value(r.get(k)) for k in fieldnames})

    return JSONResponseT(
        {
            "job_id": job_id,
            "rows_total": int(len(cleaned_rows)),
            "fieldnames": fieldnames,
            "preview": preview,
            "download_url": f"/api/handwritten-invoice-download/{job_id}",
            "confirmed_rows": cleaned_rows,
        }
    )


def handwritten_invoice_download(job_id: str, *, handwritten_jobs: Dict[str, str], FileResponseT: Any) -> Any:
    csv_path = handwritten_jobs.get(job_id)
    if not csv_path or not os.path.exists(csv_path):
        return FileResponseT(path="", status_code=404)
    return FileResponseT(csv_path, filename=f"handwritten_invoices_{job_id}.csv", media_type="text/csv")
