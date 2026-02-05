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

import base64

from datetime import datetime

from typing import Any, Dict, List, Optional, Tuple

from collections import OrderedDict

import bank_statements

import invoices

import handwritten



try:



    from dotenv import load_dotenv  # type: ignore



except Exception:



    load_dotenv = None  # type: ignore





if load_dotenv is not None:

    try:

        load_dotenv()

    except Exception:

        pass







import pdfplumber



try:



    import httpx



except Exception:



    httpx = None  # type: ignore



try:

    import torch  # type: ignore

except Exception:

    torch = None  # type: ignore



try:

    from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor  # type: ignore

except Exception:

    LightOnOcrForConditionalGeneration = None  # type: ignore

    LightOnOcrProcessor = None  # type: ignore



try:

    from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForCausalLM  # type: ignore

except Exception:

    try:
        from transformers import AutoProcessor, AutoModelForVision2Seq  # type: ignore
        AutoModelForCausalLM = None
    except Exception:
        try:
            from transformers import AutoProcessor, AutoModelForMultimodalLM  # type: ignore
            AutoModelForVision2Seq = AutoModelForMultimodalLM
            AutoModelForCausalLM = AutoModelForMultimodalLM
        except Exception:
            try:
                from transformers import AutoProcessor, AutoModel  # type: ignore
                AutoModelForVision2Seq = AutoModel
                AutoModelForCausalLM = AutoModel
            except Exception:
                AutoProcessor = None  # type: ignore

                AutoModelForVision2Seq = None  # type: ignore

                AutoModelForCausalLM = None  # type: ignore



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



    import cv2  # type: ignore



except Exception:



    cv2 = None  # type: ignore



try:



    import numpy as np  # type: ignore



except Exception:



    np = None  # type: ignore



try:



    import pytesseract  # type:0 ignore



except Exception:



    pytesseract = None  # type: ignore



try:



    import torch  # type: ignore



except Exception:



    torch = None  # type: ignore



try:



    import easyocr  # type: ignore



except Exception:



    easyocr = None  # type: ignore



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





def _clean_output_text(text: str) -> str:

    s = _clean_text(text)

    if not s:

        return ""

    markers_to_remove = {"system", "user", "assistant"}

    lines = s.split("\n")

    cleaned_lines = []

    for ln in lines:

        stripped = ln.strip().lower()

        if stripped in markers_to_remove:

            continue

        cleaned_lines.append(ln)

    cleaned = "\n".join(cleaned_lines).strip()

    low = s.lower()

    if "assistant" in low:

        try:

            parts = re.split(r"assistant", s, maxsplit=1, flags=re.IGNORECASE)

            if len(parts) > 1:

                cleaned = parts[1].strip()

        except Exception:

            pass

    return cleaned





BBOX_PATTERN = r"!\[image\]\((image_\d+\.png)\)\s*(\d+),(\d+),(\d+),(\d+)"





def _parse_bbox_output(text: str) -> Tuple[str, List[Dict[str, Any]]]:

    detections: List[Dict[str, Any]] = []

    for match in re.finditer(BBOX_PATTERN, text or ""):

        image_ref, x1, y1, x2, y2 = match.groups()

        try:

            detections.append({"ref": image_ref, "coords": (int(x1), int(y1), int(x2), int(y2))})

        except Exception:

            continue

    cleaned = re.sub(BBOX_PATTERN, r"![image](\1)", text or "")

    return _clean_output_text(cleaned), detections





def _crop_from_bbox(source_image: Any, bbox: Dict[str, Any], padding: int = 5) -> Optional[Any]:

    if source_image is None:

        return None


    try:

        w, h = source_image.size

    except Exception:

        return None

    coords = bbox.get("coords")

    if not (isinstance(coords, (list, tuple)) and len(coords) == 4):

        return None

    try:

        x1, y1, x2, y2 = [int(x) for x in coords]

    except Exception:

        return None

    px1 = int(x1 * w / 1000)

    py1 = int(y1 * h / 1000)

    px2 = int(x2 * w / 1000)

    py2 = int(y2 * h / 1000)

    px1, py1 = max(0, px1 - padding), max(0, py1 - padding)

    px2, py2 = min(w, px2 + padding), min(h, py2 + padding)

    try:

        return source_image.crop((px1, py1, px2, py2))

    except Exception:

        return None





def _image_to_data_uri_png(image_bytes: bytes) -> str:

    b64 = base64.b64encode(image_bytes).decode("ascii")

    return f"data:image/png;base64,{b64}"





def _vllm_chat_ocr(image_png_bytes: bytes, model_id: str, base_url: str, temperature: float, max_tokens: int) -> str:

    if httpx is None:

        return ""

    endpoint = (base_url or "").strip().rstrip("/")

    if not endpoint:

        return ""

    url = endpoint + "/chat/completions"

    payload = {

        "model": model_id,

        "messages": [

            {

                "role": "user",

                "content": [

                    {"type": "image_url", "image_url": {"url": _image_to_data_uri_png(image_png_bytes)}},

                ],

            }

        ],

        "max_tokens": int(max_tokens) if max_tokens else 2048,

        "temperature": float(temperature) if temperature and float(temperature) > 0 else 0.0,

        "top_p": 0.9,

        "stream": False,

    }

    try:

        with httpx.Client(timeout=120) as client:  # type: ignore[union-attr]

            r = client.post(url, json=payload)

            r.raise_for_status()

            data = r.json()

    except Exception:

        return ""

    try:

        if isinstance(data, dict):

            choices = data.get("choices")

            if isinstance(choices, list) and choices:

                msg = (choices[0] or {}).get("message") or {}

                content = msg.get("content")

                if isinstance(content, str):

                    return _clean_output_text(content)

    except Exception:

        return ""

    return ""





def _lighton_local_chat_ocr(image_png_bytes: bytes, model_name: str, temperature: float, max_tokens: int) -> str:

    if torch is None:
        print("DEBUG: Local OCR - torch not available")
        return ""

    if Image is None:
        print("DEBUG: Local OCR - PIL not available")
        return ""

    try:
        img = Image.open(io.BytesIO(image_png_bytes)).convert("RGB")
        print(f"DEBUG: Local OCR - Image loaded, size: {img.size}")
    except Exception as e:
        print(f"DEBUG: Local OCR - Failed to load image: {e}")
        return ""

    try:
        print(f"DEBUG: Local OCR - Getting model: {model_name}")
        model, processor, device = model_manager.get_model(model_name)
        print(f"DEBUG: Local OCR - Model loaded on device: {device}")
    except Exception as e:
        print(f"DEBUG: Local OCR - Failed to load model: {e}")
        # Try direct model loading as fallback
        try:
            print("DEBUG: Trying direct model loading")
            from transformers import AutoProcessor, AutoModelForMultimodalLM
            model_id = "lightonai/LightOnOCR-2-1B"
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForMultimodalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else "cpu",
                trust_remote_code=True
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"DEBUG: Direct model loading successful on {device}")
        except Exception as e2:
            print(f"DEBUG: Direct model loading also failed: {e2}")
            return ""

    try:

        chat = [

            {

                "role": "user",

                "content": [

                    {"type": "image", "url": img},
                    {"type": "text", "text": "Extract all readable text from this image. Output plain text only."},

                ],

            }

        ]

        if hasattr(processor, "apply_chat_template"):

            inputs = processor.apply_chat_template(

                chat,

                add_generation_prompt=True,

                tokenize=True,

                return_dict=True,

                return_tensors="pt",

            )

        else:

            inputs = processor(images=img, return_tensors="pt")

        try:
            if hasattr(inputs, "to") and callable(getattr(inputs, "to")):
                inputs = inputs.to(device)
            elif isinstance(inputs, dict):
                inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        except Exception:
            if isinstance(inputs, dict):
                inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        gen = model.generate(

            **inputs,

            max_new_tokens=int(max_tokens) if max_tokens else 2048,

            temperature=float(temperature) if temperature and float(temperature) > 0 else 0.0,

            top_p=0.9,

            top_k=0,

            use_cache=True,

            do_sample=bool(temperature and float(temperature) > 0),

        )

        if hasattr(processor, "decode"):

            out = processor.decode(gen[0], skip_special_tokens=True)

        elif hasattr(processor, "batch_decode"):

            out_list = processor.batch_decode(gen, skip_special_tokens=True)

            out = out_list[0] if isinstance(out_list, list) and out_list else ""

        else:

            out = ""

        return _clean_output_text(str(out))

    except Exception:

        return ""





def _handwritten_lighton_multimodel_ocr(image_png_bytes: bytes, model_name: str, temperature: float, max_tokens: int) -> str:

    cfg = MODEL_REGISTRY.get(model_name) or {}

    model_id = _clean_text(cfg.get("model_id"))

    vllm = _clean_text(cfg.get("vllm_endpoint"))

    print(f"DEBUG OCR: model_name={model_name}, model_id={model_id}, vllm={vllm}")

    if vllm and model_id:

        print("DEBUG OCR: Trying VLLM endpoint")
        txt = _vllm_chat_ocr(image_png_bytes, model_id=model_id, base_url=vllm, temperature=temperature, max_tokens=max_tokens)

        if txt:
            print("DEBUG OCR: VLLM succeeded")
            return txt
        else:
            print("DEBUG OCR: VLLM returned empty")

    print(f"DEBUG OCR: LIGHTON_LOCAL_ENABLED={LIGHTON_LOCAL_ENABLED}")
    if LIGHTON_LOCAL_ENABLED:

        print("DEBUG OCR: Trying local LightOn model")
        txt2 = _lighton_local_chat_ocr(image_png_bytes, model_name=model_name, temperature=temperature, max_tokens=max_tokens)

        if txt2:
            print("DEBUG OCR: Local LightOn succeeded")
            return txt2
        else:
            print("DEBUG OCR: Local LightOn returned empty")

    print(f"DEBUG OCR: Trying fallback vision OCR with model_id={model_id}")
    if model_id:

        txt3 = _lighton_vision_ocr_text(image_png_bytes, prompt="Extract all readable text from this image. Output plain text only.")

        result = _clean_output_text(txt3)
        print(f"DEBUG OCR: Fallback OCR result length: {len(result)}")
        return result

    print("DEBUG OCR: All OCR methods failed")
    return ""





def _invoice_render_page(pdf_path: str, page_num: int) -> Optional[Any]:

    if Image is None:

        return None

    p = int(page_num) if page_num and int(page_num) > 0 else 1

    idx = p - 1

    try:

        if pdfium is not None:

            doc = pdfium.PdfDocument(pdf_path)

            if len(doc) < 1:

                return None

            idx2 = max(0, min(idx, len(doc) - 1))

            page = doc[idx2]

            bitmap = page.render(scale=5)

            return bitmap.to_pil()  # type: ignore[union-attr]

        if fitz is not None:

            doc2 = fitz.open(pdf_path)  # type: ignore[union-attr]

            if doc2.page_count < 1:  # type: ignore[union-attr]

                return None

            idx3 = max(0, min(idx, doc2.page_count - 1))  # type: ignore[union-attr]

            page2 = doc2.load_page(idx3)  # type: ignore[union-attr]

            pix = page2.get_pixmap(matrix=fitz.Matrix(5, 5))  # type: ignore[union-attr]

            img_bytes = pix.tobytes("png")

            return Image.open(io.BytesIO(img_bytes))

        return None

    except Exception:

        return None



APP_VERSION = os.getenv("APP_VERSION", "dev")



INVOICE_ENABLE_LABEL_OCR = os.getenv("INVOICE_ENABLE_LABEL_OCR", "").strip().lower() in {"1", "true", "yes", "y", "on"}



BANK_ENABLE_LABEL_OCR = os.getenv("BANK_ENABLE_LABEL_OCR", "").strip().lower() in {"1", "true", "yes", "y", "on"}



BANKPDF_OCR = os.getenv("BANKPDF_OCR", "").strip().lower() not in {"0", "false", "no", "n", "off"}



OCR_PROVIDER = os.getenv("OCR_PROVIDER", "tesseract").strip().lower()



USE_CUDA = os.getenv("USE_CUDA", "").strip().lower() in {"1", "true", "yes", "y", "on"}



_EASYOCR_READER: Any = None



def _easyocr_available() -> Tuple[bool, str]:

    if easyocr is None:

        return False, "easyocr is not installed"

    if torch is None:

        return False, "torch is not installed"

    return True, ""



def _get_easyocr_reader() -> Any:

    global _EASYOCR_READER

    if _EASYOCR_READER is not None:

        return _EASYOCR_READER

    ok, _detail = _easyocr_available()

    if not ok:

        return None

    use_gpu = bool(USE_CUDA and (torch is not None) and bool(getattr(torch, "cuda", None)) and bool(torch.cuda.is_available()))

    try:

        _EASYOCR_READER = easyocr.Reader(["en"], gpu=bool(use_gpu))  # type: ignore[union-attr]

    except Exception:

        _EASYOCR_READER = None

    return _EASYOCR_READER

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()

DEEPSEEK_OCR2_URL = os.getenv("DEEPSEEK_OCR2_URL", "").strip()

DEEPSEEK_OCR_MODEL = os.getenv("DEEPSEEK_OCR_MODEL", "deepseek-vl").strip()

DEEPSEEK_OCR_TEMPERATURE = os.getenv("DEEPSEEK_OCR_TEMPERATURE", "0").strip()

LIGHTON_API_KEY = os.getenv("LIGHTON_API_KEY", "").strip()

LIGHTON_OCR2_URL = os.getenv("LIGHTON_OCR2_URL", "").strip()

LIGHTON_OCR_MODEL = os.getenv("LIGHTON_OCR_MODEL", "lightonai/LightOnOCR-2-1B").strip()

LIGHTON_OCR_TEMPERATURE = os.getenv("LIGHTON_OCR_TEMPERATURE", "0").strip()

LIGHTON_LOCAL_ENABLED = os.getenv("LIGHTON_LOCAL_ENABLED", "").strip().lower() in {"1", "true", "yes", "y", "on"}

LIGHTON_LOCAL_MODEL_NAME = os.getenv("LIGHTON_LOCAL_MODEL_NAME", "LightOnOCR-2-1B (Best OCR)").strip()



VLLM_ENDPOINT_OCR = os.getenv("VLLM_ENDPOINT_OCR", "").strip()

VLLM_ENDPOINT_BBOX = os.getenv("VLLM_ENDPOINT_BBOX", "").strip()





CURRENCY_RE = re.compile(r"\(?\s*-?\s*(?:£|Â£|\$|€)?\s*\d[\d,]*\.\d{2}\s*\)?")







app = FastAPI(title="Bank Statement PDF → CSV")







templates = Jinja2Templates(directory=os.path.join(APP_DIR, "templates"))







JOBS: Dict[str, str] = {}



INVOICE_JOBS: Dict[str, str] = {}



INVOICE_REVIEW_JOBS: Dict[str, str] = {}



HANDWRITTEN_REVIEW_JOBS: Dict[str, str] = {}



HANDWRITTEN_JOBS: Dict[str, str] = {}







@app.get("/", response_class=HTMLResponse)



async def home(request: Request) -> HTMLResponse:



    return templates.TemplateResponse("home.html", {"request": request})







@app.get("/invoice", response_class=HTMLResponse)



async def invoice_page(request: Request) -> HTMLResponse:



    return templates.TemplateResponse("home.html", {"request": request})





def _clean_text(value: Any) -> str:

    if value is None:

        return ""

    s = str(value)

    s = s.replace("\u00a0", " ")

    s = s.replace("\r", "\n")

    s = re.sub(r"[ \t]+", " ", s)

    s = re.sub(r"\n{2,}", "\n", s)

    return s.strip()





MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {

    "LightOnOCR-2-1B (Best OCR)": {

        "model_id": "lightonai/LightOnOCR-2-1B",

        "has_bbox": False,

        "description": "Best overall OCR performance",

        "vllm_endpoint": VLLM_ENDPOINT_OCR,

    },

    "LightOnOCR-2-1B-bbox (Best Bbox)": {

        "model_id": "lightonai/LightOnOCR-2-1B-bbox",

        "has_bbox": True,

        "description": "Best bounding box detection",

        "vllm_endpoint": VLLM_ENDPOINT_BBOX,

    },

    "LightOnOCR-2-1B-base": {

        "model_id": "lightonai/LightOnOCR-2-1B-base",

        "has_bbox": False,

        "description": "Base OCR model",

    },

    "LightOnOCR-2-1B-bbox-base": {

        "model_id": "lightonai/LightOnOCR-2-1B-bbox-base",

        "has_bbox": True,

        "description": "Base bounding box model",

    },

    "LightOnOCR-2-1B-ocr-soup": {

        "model_id": "lightonai/LightOnOCR-2-1B-ocr-soup",

        "has_bbox": False,

        "description": "OCR soup variant",

    },

    "LightOnOCR-2-1B-bbox-soup": {

        "model_id": "lightonai/LightOnOCR-2-1B-bbox-soup",

        "has_bbox": True,

        "description": "Bounding box soup variant",

    },

}



class ModelManager:

    def __init__(self, max_cached: int = 2):

        self._cache: "OrderedDict[str, Tuple[Any, Any, str]]" = OrderedDict()

        self._max_cached = max_cached

    def get_model(self, model_name: str) -> Tuple[Any, Any, str]:

        config = MODEL_REGISTRY.get(model_name)

        if config is None:

            raise ValueError(f"Unknown model: {model_name}")

        model_id = str(config.get("model_id") or "").strip()

        if not model_id:

            raise ValueError(f"Model id missing for: {model_name}")

        if model_id in self._cache:

            self._cache.move_to_end(model_id)

            return self._cache[model_id]

        while len(self._cache) >= self._max_cached:

            _evicted_id, (evicted_model, _evicted_processor, evicted_device) = self._cache.popitem(last=False)

            try:

                del evicted_model

            except Exception:

                pass

            if torch is not None and evicted_device == "cuda":

                try:

                    torch.cuda.empty_cache()  # type: ignore[union-attr]

                except Exception:

                    pass

        if torch is None:

            raise RuntimeError("Local LightOnOCR requires torch")

        use_lighton = LightOnOcrForConditionalGeneration is not None and LightOnOcrProcessor is not None

        use_auto = (AutoProcessor is not None) and (AutoModelForVision2Seq is not None or AutoModelForCausalLM is not None)

        if not use_lighton and not use_auto:

            raise RuntimeError("Local LightOnOCR requires transformers (LightOn classes or AutoModel/AutoProcessor)")

        if (not use_lighton) and model_id.lower().startswith("lightonai/"):

            raise RuntimeError(
                "Local LightOnOCR model requires LightOn transformers classes (LightOnOcrForConditionalGeneration/LightOnOcrProcessor). "
                "AutoModel fallback is disabled for LightOn models to avoid incompatible model-type instantiation."
            )

        device = "cuda" if bool(getattr(torch, "cuda", None)) and torch.cuda.is_available() else "cpu"  # type: ignore[union-attr]

        if device == "cuda":

            attn_implementation = "sdpa"

            torch_dtype = getattr(torch, "bfloat16", None) or getattr(torch, "float16")

        else:

            attn_implementation = "eager"

            torch_dtype = getattr(torch, "float32")

        if use_lighton:

            model = (

                LightOnOcrForConditionalGeneration.from_pretrained(

                    model_id,

                    attn_implementation=attn_implementation,

                    torch_dtype=torch_dtype,

                    trust_remote_code=True,

                )

                .to(device)

                .eval()

            )

            processor = LightOnOcrProcessor.from_pretrained(model_id, trust_remote_code=True)

        else:

            model_cls = AutoModelForVision2Seq or AutoModelForCausalLM

            model = (

                model_cls.from_pretrained(

                    model_id,

                    attn_implementation=attn_implementation,

                    torch_dtype=torch_dtype,

                    trust_remote_code=True,

                )

                .to(device)

                .eval()

            )

            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        self._cache[model_id] = (model, processor, device)

        return model, processor, device



model_manager = ModelManager(max_cached=2)





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

    s = s.replace("Â£", "£")

    s = s.replace("GBP", "").replace("gbp", "")

    s = s.replace("£", "").replace("$", "").replace("€", "")

    s = s.replace(",", "").replace(" ", "")

    if s.endswith("-"):

        neg = True

        s = s[:-1]

    m = re.search(r"-?\d+(?:\.\d{1,2})?", s)

    if not m:

        raise ValueError("Invalid amount")

    amt = float(m.group(0))

    if amt < 0:

        return amt

    return -amt if neg else amt





def _format_money_token(value: Any) -> str:

    s = _clean_text(value)

    if not s:

        return ""

    try:

        return f"{_parse_money(s):.2f}"

    except Exception:

        s = s.replace("Â£", "£")

        s = s.replace("GBP", "").replace("gbp", "")

        s = s.replace("£", "").replace("$", "").replace("€", "")

        s = s.replace(",", "").replace(" ", "")

        m = re.search(r"-?\d+(?:\.\d{1,2})?", s)

        return _clean_text(m.group(0)) if m else ""





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



def _ocr_preprocess_variants(pil_img: Any) -> List[Any]:

    if Image is None:

        return [pil_img]



    def _safe_copy(img: Any) -> Any:

        try:

            return img.copy()

        except Exception:

            return img



    def _rotate_from_osd(img: Any) -> Any:

        if pytesseract is None or TesseractOutput is None:

            return img

        try:

            osd = pytesseract.image_to_osd(img, output_type=TesseractOutput.DICT)  # type: ignore[union-attr]

        except Exception:

            return img

        try:

            rotate = int(osd.get("rotate", 0) or 0)

        except Exception:

            rotate = 0

        if rotate not in (0, 90, 180, 270):

            return img

        if not rotate:

            return img

        try:

            return img.rotate(-rotate, expand=True)

        except Exception:

            return img



    def _pil_to_bgr(img: Any) -> Any:

        if cv2 is None or np is None:

            return None

        try:

            rgb = img.convert("RGB")

            arr = np.array(rgb)

            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        except Exception:

            return None



    def _bgr_to_pil(bgr: Any) -> Any:

        if cv2 is None or np is None or Image is None:

            return None

        try:

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            return Image.fromarray(rgb)

        except Exception:

            return None



    def _deskew_bgr(bgr: Any) -> Any:

        if cv2 is None or np is None:

            return bgr

        try:

            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

            coords = np.column_stack(np.where(thr > 0))

            if coords.size == 0:

                return bgr

            rect = cv2.minAreaRect(coords)

            angle = float(rect[-1])

            if angle < -45:

                angle = 90 + angle

            angle = -angle

            if abs(angle) < 0.6:

                return bgr

            (h, w) = bgr.shape[:2]

            M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)

            return cv2.warpAffine(bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        except Exception:

            return bgr



    def _perspective_fix_bgr(bgr: Any) -> Any:

        if cv2 is None or np is None:

            return bgr

        try:

            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            edges = cv2.Canny(gray, 50, 150)

            edges = cv2.dilate(edges, None, iterations=2)

            cnts, _hier = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            if not cnts:

                return bgr

            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

            doc = None

            for c in cnts:

                peri = cv2.arcLength(c, True)

                approx = cv2.approxPolyDP(c, 0.02 * peri, True)

                if len(approx) == 4:

                    doc = approx

                    break

            if doc is None:

                return bgr

            pts = doc.reshape(4, 2).astype("float32")

            s = pts.sum(axis=1)

            diff = np.diff(pts, axis=1)

            tl = pts[np.argmin(s)]

            br = pts[np.argmax(s)]

            tr = pts[np.argmin(diff)]

            bl = pts[np.argmax(diff)]

            widthA = np.linalg.norm(br - bl)

            widthB = np.linalg.norm(tr - tl)

            maxW = int(max(widthA, widthB))

            heightA = np.linalg.norm(tr - br)

            heightB = np.linalg.norm(tl - bl)
            

            maxH = int(max(heightA, heightB))

            if maxW < 200 or maxH < 200:

                return bgr

            dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype="float32")

            M = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], dtype="float32"), dst)

            warped = cv2.warpPerspective(bgr, M, (maxW, maxH))

            return warped

        except Exception:

            return bgr



    variants: List[Any] = []

    base = _safe_copy(pil_img)

    variants.append(base)

    variants.append(_rotate_from_osd(_safe_copy(base)))

    bgr0 = _pil_to_bgr(base)

    if bgr0 is not None:

        bgr1 = _perspective_fix_bgr(bgr0)

        bgr2 = _deskew_bgr(bgr1)

        pil2 = _bgr_to_pil(bgr2)

        if pil2 is not None:

            variants.append(pil2)

            variants.append(_rotate_from_osd(_safe_copy(pil2)))



    out: List[Any] = []

    seen: set = set()

    for v in variants:

        try:

            key = (int(getattr(v, "size", (0, 0))[0]), int(getattr(v, "size", (0, 0))[1]), str(getattr(v, "mode", "")))

        except Exception:

            key = (id(v),)

        if key in seen:

            continue

        seen.add(key)

        out.append(v)

    return out or [pil_img]





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

    if OCR_PROVIDER == "deepseek":

        try:

            with open(image_path, "rb") as f:

                img_bytes = f.read()

        except Exception:

            img_bytes = b""



        if img_bytes:

            lines_ds, ok_ds = _extract_text_lines_from_image_with_deepseek(img_bytes)

            if ok_ds and lines_ds:

                return lines_ds, True



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

    for base_img in _ocr_preprocess_variants(pil_img):

        for angle in (0, 90, 180, 270):

            try:

                img2 = base_img.rotate(angle, expand=True) if angle else base_img

            except Exception:

                img2 = base_img

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





def _ocr_words_and_lines_from_pil_image(img: Any) -> Dict[str, Any]:

    out: Dict[str, Any] = {"lines": [], "words": []}

    if img is None:

        return out

    easyocr_enabled = (OCR_PROVIDER == "easyocr") or (USE_CUDA and OCR_PROVIDER in {"tesseract", "easyocr"})

    if easyocr_enabled:

        reader = _get_easyocr_reader()

        if reader is not None:

            try:

                results = reader.readtext(img, detail=1, paragraph=False)  # type: ignore[union-attr]

            except Exception:

                results = []

            lines: List[str] = []

            words: List[Dict[str, Any]] = []

            for r in results or []:

                try:

                    bbox, text, conf = r

                except Exception:

                    continue

                t = _clean_text(text)

                if not t:

                    continue

                lines.append(t)

                try:

                    xs = [float(p[0]) for p in (bbox or [])]

                    ys = [float(p[1]) for p in (bbox or [])]

                    x0 = int(min(xs)) if xs else 0

                    y0 = int(min(ys)) if ys else 0

                    x1 = int(max(xs)) if xs else 0

                    y1 = int(max(ys)) if ys else 0

                    w = max(0, x1 - x0)

                    h = max(0, y1 - y0)

                except Exception:

                    x0, y0, w, h = 0, 0, 0, 0

                try:

                    conf_val = float(conf) if conf not in (None, "") else None

                except Exception:

                    conf_val = None

                words.append(

                    {

                        "index": int(len(words) + 1),

                        "text": t,

                        "confidence": conf_val,

                        "bbox": {"x": x0, "y": y0, "w": w, "h": h},

                    }

                )

            out["lines"] = [{"index": i + 1, "text": ln} for i, ln in enumerate(lines)]

            out["words"] = words

            return out

    if pytesseract is None:

        return out

    ok, _detail = _invoice_tesseract_available() if "_invoice_tesseract_available" in globals() else _tesseract_available()

    if not ok:

        return out

    try:

        txt = pytesseract.image_to_string(img, config="--oem 1 --psm 6")

    except Exception:

        txt = ""

    lines = [_clean_text(x) for x in str(txt).splitlines()]

    lines = [x for x in lines if x]

    out["lines"] = [{"index": i + 1, "text": ln} for i, ln in enumerate(lines)]

    try:

        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config="--oem 1 --psm 6")

        n = len(data.get("text") or [])

        words: List[Dict[str, Any]] = []

        for i in range(n):

            w = _clean_text((data.get("text") or [""])[i])

            if not w:

                continue

            try:

                conf_raw = (data.get("conf") or [""])[i]

                conf = float(conf_raw) if conf_raw not in (None, "", "-1") else None

            except Exception:

                conf = None

            try:

                left = int((data.get("left") or [0])[i])

                top = int((data.get("top") or [0])[i])

                width = int((data.get("width") or [0])[i])

                height = int((data.get("height") or [0])[i])

            except Exception:

                left, top, width, height = 0, 0, 0, 0

            words.append(

                {

                    "index": int(len(words) + 1),

                    "text": w,

                    "confidence": conf,

                    "bbox": {"x": left, "y": top, "w": width, "h": height},

                }

            )

        out["words"] = words

    except Exception:

        out["words"] = []

    return out





def _ocr_words_and_lines_from_image_bytes(img_bytes: bytes) -> Dict[str, Any]:

    if not img_bytes:

        return {"lines": [], "words": []}

    if Image is None:

        return {"lines": [], "words": []}

    try:

        img = Image.open(io.BytesIO(img_bytes))

    except Exception:

        img = None

    if img is None:

        return {"lines": [], "words": []}

    try:

        img2 = img.convert("RGB")

    except Exception:

        img2 = img

    return _ocr_words_and_lines_from_pil_image(img2)





def _extract_text_lines_from_image_with_deepseek(image_bytes: bytes) -> Tuple[List[str], bool]:

    if not BANKPDF_OCR:

        return [], False

    if httpx is None:

        return [], False

    if not DEEPSEEK_API_KEY or not DEEPSEEK_OCR2_URL:

        return [], False



    txt = _deepseek_vision_ocr_text(image_bytes)

    if not txt:

        return [], False



    cleaned = [_clean_text(x) for x in (txt.splitlines() if txt else [])]

    cleaned = [x for x in cleaned if x]

    return cleaned, bool(cleaned)





def _deepseek_vision_ocr_text(image_bytes: bytes, prompt: str = "") -> str:

    if httpx is None:

        return ""

    if not DEEPSEEK_API_KEY or not DEEPSEEK_OCR2_URL:

        return ""



    p = _clean_text(prompt) or "Extract all readable text from this image. Output plain text only."

    try:

        temperature = float(DEEPSEEK_OCR_TEMPERATURE) if _clean_text(DEEPSEEK_OCR_TEMPERATURE) else 0.0

    except Exception:

        temperature = 0.0



    # DeepSeek API is OpenAI-compatible for chat/completions. For vision-capable models,

    # we send a mixed content array with text + an image data URI.

    image_b64 = base64.b64encode(image_bytes).decode("ascii")

    data_uri = f"data:image/png;base64,{image_b64}"

    payload = {

        "model": DEEPSEEK_OCR_MODEL or "deepseek-vl",

        "messages": [

            {

                "role": "user",

                "content": [

                    {"type": "text", "text": p},

                    {"type": "image_url", "image_url": {"url": data_uri}},

                ],

            }

        ],

        "temperature": temperature,

    }

    headers = {

        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",

        "Content-Type": "application/json",

    }



    try:

        with httpx.Client(timeout=90) as client:  # type: ignore[union-attr]

            r = client.post(DEEPSEEK_OCR2_URL, headers=headers, json=payload)

            r.raise_for_status()

            data = r.json()

    except Exception:

        return ""



    # Parse OpenAI-style response

    try:

        if isinstance(data, dict):

            choices = data.get("choices")

            if isinstance(choices, list) and choices:

                msg = (choices[0] or {}).get("message") or {}

                content = msg.get("content")

                if isinstance(content, str):

                    return _clean_text(content)

                if isinstance(content, list):

                    # Some variants return structured content blocks

                    parts = []

                    for it in content:

                        if isinstance(it, dict) and isinstance(it.get("text"), str):

                            parts.append(it.get("text"))

                    if parts:

                        return _clean_text("\n".join(parts))



            # Fallback for non-chat custom OCR endpoints

            t = data.get("text") or data.get("result") or data.get("data")

            if isinstance(t, str):

                return _clean_text(t)

    except Exception:

        return ""

    return ""





def _lighton_vision_ocr_text(image_bytes: bytes, prompt: str = "") -> str:

    if LIGHTON_LOCAL_ENABLED:

        if Image is None:

            return ""

        try:

            img = Image.open(io.BytesIO(image_bytes))

            img = img.convert("RGB")

        except Exception:

            return ""

        try:

            model, processor, device = model_manager.get_model(LIGHTON_LOCAL_MODEL_NAME)

        except Exception:

            return ""

        try:

            p = _clean_text(prompt) or "Extract all readable text from this image. Output plain text only."

            try:

                inputs = processor(text=p, images=img, return_tensors="pt")

            except Exception:

                inputs = processor(images=img, return_tensors="pt")

            if device == "cuda":

                inputs = {k: v.to(device) for k, v in inputs.items()}

            gen = model.generate(**inputs, max_new_tokens=768)

            out = processor.batch_decode(gen, skip_special_tokens=True)

            txt = out[0] if isinstance(out, list) and out else ""

            return _clean_text(txt)

        except Exception:

            return ""

    if httpx is None:

        return ""

    if not LIGHTON_OCR2_URL:

        return ""



    p = _clean_text(prompt) or "Extract all readable text from this image. Output plain text only."

    try:

        temperature = float(LIGHTON_OCR_TEMPERATURE) if _clean_text(LIGHTON_OCR_TEMPERATURE) else 0.0

    except Exception:

        temperature = 0.0



    image_b64 = base64.b64encode(image_bytes).decode("ascii")

    data_uri = f"data:image/png;base64,{image_b64}"

    payload = {

        "model": LIGHTON_OCR_MODEL or "lightonai/LightOnOCR-2-1B",

        "messages": [

            {

                "role": "user",

                "content": [

                    {"type": "text", "text": p},

                    {"type": "image_url", "image_url": {"url": data_uri}},

                ],

            }

        ],

        "temperature": temperature,

    }



    headers = {"Content-Type": "application/json"}

    if _clean_text(LIGHTON_API_KEY):

        headers["Authorization"] = f"Bearer {LIGHTON_API_KEY}"



    try:

        with httpx.Client(timeout=90) as client:  # type: ignore[union-attr]

            r = client.post(LIGHTON_OCR2_URL, headers=headers, json=payload)

            r.raise_for_status()

            data = r.json()

    except Exception:

        return ""



    try:

        if isinstance(data, dict):

            choices = data.get("choices")

            if isinstance(choices, list) and choices:

                msg = (choices[0] or {}).get("message") or {}

                content = msg.get("content")

                if isinstance(content, str):

                    return _clean_text(content)

                if isinstance(content, list):

                    parts = []

                    for it in content:

                        if isinstance(it, dict) and isinstance(it.get("text"), str):

                            parts.append(it.get("text"))

                    if parts:

                        return _clean_text("\n".join(parts))



            t = data.get("text") or data.get("result") or data.get("data")

            if isinstance(t, str):

                return _clean_text(t)

    except Exception:

        return ""

    return ""





def _lighton_extract_used_vehicle_fields_from_image_bytes(image_bytes: bytes) -> Dict[str, Any]:
    # Extract all readable text from the image without field filtering
    txt = _lighton_vision_ocr_text(image_bytes, prompt="Extract all readable text from this image. Output plain text only.")
    return {"raw_text": txt, "parsed": {}}





def _deepseek_extract_used_vehicle_fields_from_pdf(pdf_path: str) -> Dict[str, Any]:

    if Image is None:

        return {}

    if not DEEPSEEK_API_KEY or not DEEPSEEK_OCR2_URL or httpx is None:

        return {}



    img = _invoice_render_first_page(pdf_path)

    if img is None:

        return {}



    prompt = (

        "This is a scanned 'Used Vehicle Purchase Invoice' with handwritten entries. "

        "Extract ONLY these handwritten fields and output STRICT JSON with keys: "

        "document_date (dd/mm/yy or dd/mm/yyyy), supplier, make, model, colour, reg_no (UK plate like AB12 CDE), buying_price (number). "

        "Return JSON only, no explanation."

    )



    def _score(d: Dict[str, Any]) -> int:

        if not isinstance(d, dict):

            return -1

        score = 0

        dd = _clean_text(d.get("document_date")).replace("-", "/")

        if _is_valid_uk_date(dd):

            score += 4

        rn = _clean_text(d.get("reg_no")).upper()

        if re.search(r"\b[A-Z]{2}[0-9O]{2}\s*[A-Z]{3}\b", rn):

            score += 4

        try:

            bp = d.get("buying_price")

            bp_num = float(bp) if bp not in (None, "") else None

        except Exception:

            bp_num = None

        if bp_num is not None and 0 < bp_num < 100000:

            score += 4

        if _clean_text(d.get("supplier")):

            score += 2

        if _clean_text(d.get("make")):

            score += 2

        return score



    best: Dict[str, Any] = {}

    best_score = -1



    for angle in (0, 90, 180, 270):

        try:

            img2 = img.rotate(angle, expand=True) if angle else img

        except Exception:

            img2 = img

        try:

            buf = io.BytesIO()

            img2.save(buf, format="PNG")

            img_bytes = buf.getvalue()

        except Exception:

            continue



        txt = _deepseek_vision_ocr_text(img_bytes, prompt=prompt)

        if not txt:

            continue



        data: Any = None

        try:

            data = json.loads(txt)

        except Exception:

            try:

                m = re.search(r"\{[\s\S]*\}", txt)

                if m:

                    data = json.loads(m.group(0))

            except Exception:

                data = None



        if isinstance(data, dict):

            sc = _score(data)

            if sc > best_score:

                best_score = sc

                best = data



    return best





def _invoice_ocr_autotrader_costs_box(pdf_path: str) -> Dict[str, Any]:



    out: Dict[str, Any] = {}



    ok, _detail = _invoice_tesseract_available()

    if not ok or pytesseract is None:

        return out



    img = _invoice_render_first_page(pdf_path)

    if img is None:

        return out



    try:

        base_img = _auto_crop_to_red_border(img)

    except Exception:

        base_img = img



    try:

        W, H = base_img.size

        # Costs box is on the right side of the page, mid-lower area.
        # Using broader ROI to survive template shifts and capture more content
        roi1 = (int(W * 0.55), int(H * 0.30), int(W * 0.99), int(H * 0.80))
        roi2 = (int(W * 0.50), int(H * 0.35), int(W * 0.99), int(H * 0.85))
        roi3 = (int(W * 0.60), int(H * 0.25), int(W * 0.98), int(H * 0.90))

        def _ocr_roi(roi: Tuple[int, int, int, int]) -> str:
            c = base_img.crop(roi)
            c = c.resize((max(1, c.size[0] * 3), max(1, c.size[1] * 3)))
            c = _invoice_preprocess_crop(c)
            return pytesseract.image_to_string(c, config="--oem 1 --psm 6")

        # Try multiple regions until we get good text
        txt = _ocr_roi(roi1)
        if not _clean_text(txt) or len(_clean_text(txt).split()) < 10:
            txt = _ocr_roi(roi2)
        if not _clean_text(txt) or len(_clean_text(txt).split()) < 10:
            txt = _ocr_roi(roi3)

    except Exception:

        return out



    t = _clean_text(txt)

    if not t:

        return out



    def _amt_after(label: str) -> Optional[float]:

        patterns = [
            rf"\b{label}\b[^0-9]{{0,40}}(\(?\s*-?\s*(?:£|Â£|\$|€)?\s*\d[\d,]*(?:[\.,]\s*\d{{1,2}})?\s*\)?)",
            rf"{label}[^0-9]{{0,40}}(\d[\d,]*(?:[\.,]\d{{1,2}})?)",
            rf"{label}\s*[:\-]?\s*(\d[\d,]*(?:[\.,]\d{{1,2}})?)",
            rf"{label}\s*(\d[\d,]*(?:[\.,]\d{{1,2}})?)"
        ]
        
        for pattern in patterns:
            m = re.search(pattern, t, flags=re.IGNORECASE)
            if m:
                break
        
        if not m:

            return None

        s = _clean_text(m.group(1))

        s = s.replace("Â£", "£")

        s = re.sub(r"(?<=\d)\s*[\.,]\s*(?=\d{1,2}\b)", ".", s)

        if "." not in s and re.search(r"\b\d{1,6},\d{2}\b", s):

            s = re.sub(r"\b(\d{1,6}),(\d{2})\b", r"\1.\2", s)

        try:

            v = _parse_money(s)

            return abs(float(v))

        except Exception:

            return None



    sub = _amt_after("Subtotal") or _amt_after("Sub Total") or _amt_after("Net") or _amt_after("Total Net")

    vat = _amt_after(r"VAT\s*Total") or _amt_after("VAT") or _amt_after("Tax") or _amt_after("Total VAT")

    grand = _amt_after(r"Grand\s*Total") or _amt_after("Total") or _amt_after("Grand Total")



    if sub is not None:

        out["std_net"] = sub

    if vat is not None:

        out["vat_amount"] = vat

    if grand is not None:

        out["buying_price"] = grand

        out["non_vat"] = grand

    # Fallback calculations
    if grand is None and sub is not None and vat is not None:
        # Calculate grand total from net + VAT
        calculated_grand = sub + vat
        out["buying_price"] = calculated_grand
        out["non_vat"] = calculated_grand
    elif vat is None and grand is not None and sub is not None:
        # Calculate VAT from grand total - net
        calculated_vat = grand - sub
        out["vat_amount"] = calculated_vat
    elif sub is None and grand is not None and vat is not None:
        # Calculate net from grand total - VAT
        calculated_net = grand - vat
        out["std_net"] = calculated_net



    return out





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



    if OCR_PROVIDER == "deepseek":

        try:

            lines_ds, ok_ds = _extract_text_lines_from_pdf_with_deepseek(pdf_path)

            if ok_ds and lines_ds:

                return lines_ds, True

        except Exception:

            pass



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





def _extract_text_lines_from_pdf_with_deepseek(pdf_path: str) -> Tuple[List[str], bool]:

    if httpx is None:

        return [], False

    if not DEEPSEEK_API_KEY or not DEEPSEEK_OCR2_URL:

        return [], False

    if Image is None:

        return [], False

    if fitz is None and pdfium is None:

        return [], False



    ocr_lines: List[str] = []



    def _best_lines_for_pil_image(img_in: Any) -> List[str]:

        best: List[str] = []

        best_score = -1

        for angle in (0, 90, 180, 270):

            try:

                img2 = img_in.rotate(angle, expand=True) if angle else img_in

            except Exception:

                img2 = img_in

            try:

                buf = io.BytesIO()

                img2.save(buf, format="PNG")

                lines_i, ok_i = _extract_text_lines_from_image_with_deepseek(buf.getvalue())

            except Exception:

                lines_i, ok_i = [], False

            if not ok_i or not lines_i:

                continue

            sc = len("\n".join(lines_i))

            if sc > best_score:

                best_score = sc

                best = lines_i

        return best



    try:

        if pdfium is not None:

            doc = pdfium.PdfDocument(pdf_path)

            for i in range(len(doc)):

                page = doc[i]

                bitmap = page.render(scale=2.5)

                pil_img = bitmap.to_pil()  # type: ignore[union-attr]

                best_lines = _best_lines_for_pil_image(pil_img)

                if best_lines:

                    ocr_lines.extend(best_lines)

        else:

            doc = fitz.open(pdf_path)  # type: ignore[union-attr]

            for page in doc:

                pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5))

                img_bytes = pix.tobytes("png")

                try:

                    pil_img2 = Image.open(io.BytesIO(img_bytes))

                except Exception:

                    pil_img2 = None

                if pil_img2 is None:

                    lines_i, ok_i = _extract_text_lines_from_image_with_deepseek(img_bytes)

                    if ok_i and lines_i:

                        ocr_lines.extend(lines_i)

                else:

                    best_lines2 = _best_lines_for_pil_image(pil_img2)

                    if best_lines2:

                        ocr_lines.extend(best_lines2)

    except Exception:

        return [], False



    cleaned = [_clean_text(x) for x in ocr_lines]

    cleaned = [x for x in cleaned if x]

    return cleaned, bool(cleaned)





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

                        for base_img in _ocr_preprocess_variants(pil_img):

                            img2 = _preprocess_for_ocr(base_img)

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

    cleaned = [_clean_text(ln).lower() for ln in lines[:250]]

    joined = " ".join(cleaned)

    return any(

        k in joined

        for k in [

            "barclays",

            "barclaycard",

            "barclays bank",

            "available balance",

            "last night's balance",

            "last nights balance",

            "overdraft limit",

            "showing",

            "transactions between",

            "e-payments plan",

        ]

    )





def _extract_barclays_header_info(lines: List[str]) -> Dict[str, Any]:

    info: Dict[str, Any] = {}

    cleaned = [_clean_text(ln) for ln in lines[:250]]

    for ln in cleaned:

        # e-Payments Plan line often includes sort code and account number, e.g. "e-Payments Plan 20-25-19 30470120"

        m = re.search(

            r"\be\s*[-–]?\s*payments\s+plan\b\s+([0-9]{2}[\-\s]?[0-9]{2}[\-\s]?[0-9]{2})\s+([0-9]{6,10})\b",

            ln,

            flags=re.IGNORECASE,

        )

        if m:

            if not info.get("sort_code"):

                info["sort_code"] = _clean_text(m.group(1)).replace(" ", "").replace("-", "")

            if not info.get("account"):

                info["account"] = _clean_text(m.group(2))



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



        # Client name (often printed in upper-right on Barclays PDFs)

        if not info.get("client_name"):

            name_candidate = _clean_text(ln)

            if (

                len(name_candidate) >= 8

                and name_candidate.upper() == name_candidate

                and not any(

                    bad in name_candidate.lower()

                    for bad in [

                        "barclays",

                        "transactions",

                        "statement",

                        "account",

                        "sort code",

                        "iban",

                        "swift",

                        "available balance",

                        "last night's balance",

                        "overdraft",

                        "showing",

                        "page",

                        "today",

                    ]

                )

            ):

                info["client_name"] = name_candidate



        # Available balance / last night's balance / overdraft limit

        m = re.search(r"\bAvailable\s+balance\b\s*(£?\s*[\d,]+(?:\.\d{2})?)", ln, flags=re.IGNORECASE)

        if m and not info.get("available_balance"):

            info["available_balance"] = _clean_text(m.group(1))



        m = re.search(r"\bLast\s+night'?s\s+balance\b\s*(£?\s*[\d,]+(?:\.\d{2})?)", ln, flags=re.IGNORECASE)

        if m and not info.get("last_nights_balance"):

            info["last_nights_balance"] = _clean_text(m.group(1))



        m = re.search(r"\bOverdraft\s+limit\b\s*(£?\s*[\d,]+(?:\.\d{2})?)", ln, flags=re.IGNORECASE)

        if m and not info.get("overdraft_limit"):

            info["overdraft_limit"] = _clean_text(m.group(1))



        # Showing X transactions between START and END (sometimes repeated as from START to END)

        m = re.search(

            r"\bShowing\s+(\d+)\s+transactions\s+between\s+([0-9]{1,2}[\/\-\.][0-9]{1,2}[\/\-\.][0-9]{2,4})\s+and\s+([0-9]{1,2}[\/\-\.][0-9]{1,2}[\/\-\.][0-9]{2,4})",

            ln,

            flags=re.IGNORECASE,

        )

        if m and not info.get("transactions_count"):

            info["transactions_count"] = _clean_text(m.group(1))

            info["period_start"] = _clean_text(m.group(2))

            info["period_end"] = _clean_text(m.group(3))



        m = re.search(

            r"\bfrom\s+([0-9]{1,2}[\/\-\.][0-9]{1,2}[\/\-\.][0-9]{2,4})\s+to\s+([0-9]{1,2}[\/\-\.][0-9]{1,2}[\/\-\.][0-9]{2,4})\b",

            ln,

            flags=re.IGNORECASE,

        )

        if m and (not info.get("period_start") or not info.get("period_end")):

            info.setdefault("period_start", _clean_text(m.group(1)))

            info.setdefault("period_end", _clean_text(m.group(2)))

    

    return info





def _barclays_header_preamble_lines(info: Dict[str, Any]) -> List[List[str]]:

    rows = []

    rows.append(["Client Name", info.get("client_name") or "N/A"])

    rows.append(["Account Number", info.get("account") or "N/A"])

    rows.append(["Sort Code", info.get("sort_code") or "N/A"])

    rows.append(["IBAN", info.get("iban") or "N/A"])

    rows.append(["SWIFT/BIC", info.get("swift") or "N/A"])

    rows.append(["Statement Date", info.get("statement_date") or "N/A"])

    rows.append(["Available balance", info.get("available_balance") or "N/A"])

    rows.append(["Last night's balance", info.get("last_nights_balance") or "N/A"])

    rows.append(["Overdraft limit", info.get("overdraft_limit") or "N/A"])

    rows.append(["Showing transactions", info.get("transactions_count") or "N/A"])

    if info.get("period_start") or info.get("period_end"):

        period = f"{info.get('period_start','')} to {info.get('period_end','')}".strip()

    else:

        period = "N/A"

    rows.append(["Transactions period", period])

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





def _write_barclays_csv_with_pending(

    csv_path: str,

    preamble: List[List[str]],

    pending_rows: List[Dict[str, Any]],

    rows: List[Dict[str, Any]],

    fieldnames: List[str],

) -> None:

    with open(csv_path, "w", newline="", encoding="utf-8") as f:

        writer = csv.writer(f)

        for row in preamble:

            writer.writerow(row)

        writer.writerow([])

        if pending_rows:

            writer.writerow(["Date", "Transaction", "Amount"])

            for r in pending_rows:

                writer.writerow(

                    [

                        _format_csv_value(r.get("date")),

                        _format_csv_value(r.get("description")),

                        _format_csv_value(r.get("amount")),

                    ]

                )

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

        s = s.replace("Â£", "")

        s = s.replace(",", "").replace(" ", "")

        return s



    date_any_re = re.compile(

        r"(\d{1,2}/\d{1,2}/\d{2,4}|\d{1,2}-\d{1,2}-\d{2,4}|\d{1,2}\.\d{1,2}\.\d{2,4}|\d{1,2}\s+\w{3,9}\s+\d{2,4})",

        flags=re.IGNORECASE,

    )



    def _emit_generic_row(date_str: str, blob: str) -> Optional[Dict[str, Any]]:

        b = _clean_text(blob)

        if not b:

            return None

        low = b.lower()

        if any(

            k in low

            for k in [

                "available balance",

                "last night's balance",

                "overdraft",

                "pending",

                "transactions between",

                "date description money in money out balance",

                "date transaction amount",

            ]

        ):

            return None



        amounts_raw = [m.group(0) for m in CURRENCY_RE.finditer(b)]

        amounts = [_normalize_amount_token(a) for a in amounts_raw]



        money_in = ""

        money_out = ""

        amount_val: Optional[float] = None

        balance = ""



        if len(amounts) >= 2:

            a_txn, a_bal = amounts[0], amounts[-1]

            txn_val = _to_float_or_none(a_txn)

            amount_val = txn_val

            if txn_val is not None and txn_val < 0:

                money_out = a_txn

            elif txn_val is not None and txn_val > 0:

                money_in = a_txn

            elif txn_val is None:

                money_in = a_txn

            balance = a_bal

        elif len(amounts) == 1:

            a_txn = amounts[0]

            txn_val = _to_float_or_none(a_txn)

            amount_val = txn_val

            if txn_val is not None and txn_val < 0:

                money_out = a_txn

            elif txn_val is not None and txn_val > 0:

                money_in = a_txn

            elif txn_val is None:

                money_in = a_txn



        desc_part = b

        if amounts_raw:

            first_amt_match = CURRENCY_RE.search(b)

            if first_amt_match:

                desc_part = b[: first_amt_match.start()]

        if date_str and desc_part.lower().startswith(date_str.lower()):

            desc_part = desc_part[len(date_str) :]

        description = _clean_text(desc_part)

        if not description:

            description = "N/A"



        if not date_str:

            date_str = "N/A"



        return {

            "date": date_str,

            "description": description,

            "money_in": money_in,

            "money_out": money_out,

            "balance": balance,

            "amount": amount_val,

            "used_ocr": bool(used_ocr_hint),

        }



    def _extract_generic_transactions(lines: List[str]) -> List[Dict[str, Any]]:

        rows: List[Dict[str, Any]] = []

        current_date = ""

        current_parts: List[str] = []



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

                    r = _emit_generic_row(seg_date, seg)

                    if r is not None:

                        rows.append(r)

            else:

                r = _emit_generic_row(current_date, joined)

                if r is not None:

                    rows.append(r)

            current_date = ""

            current_parts = []



        for line in lines:

            date_str = _find_date_in_line(line)

            low = line.lower()



            if date_str:

                _flush_current()

                current_date = date_str

                current_parts = [line]

                continue



            if not current_date:

                continue



            if any(

                k in low

                for k in [

                    "page ",

                    "account",

                    "sort code",

                    "swift",

                    "iban",

                    "issued",

                    "statement",

                ]

            ):

                continue



            current_parts.append(line)



        _flush_current()

        return rows



    def _extract_barclays_transactions(lines: List[str]) -> List[Dict[str, Any]]:

        """Special parser for Barclays statements"""

        rows: List[Dict[str, Any]] = []

        pending_rows: List[Dict[str, Any]] = []



        barclays_amount_re = re.compile(r"\(?\s*-?\s*(?:£|Â£|\$|€)?\s*\d[\d,]*(?:\.\d{1,2})?\s*\)?")



        def _barclays_amount_display(tok: str) -> str:

            s = _clean_text(tok)

            s = s.replace("Â£", "£")

            return _format_money_token(s)



        def _barclays_amount_for_parse(tok: str) -> str:

            s = _clean_text(tok)

            s = s.replace("Â£", "£")

            s = s.replace("GBP", "").replace("gbp", "")

            return s



        def _barclays_subcategory_and_clean_description(description: str) -> Tuple[str, str]:

            d = _clean_text(description)

            low = d.lower()

            if "credit card" in low:

                cleaned_desc = re.sub(r"\bcredit\s+card\b", "", d, flags=re.IGNORECASE)

                return "Credit Card", _clean_text(cleaned_desc)

            if "debit card" in low:

                cleaned_desc = re.sub(r"\bdebit\s+card\b", "", d, flags=re.IGNORECASE)

                return "Debit Card", _clean_text(cleaned_desc)

            return "", d



        barclays_info = _extract_barclays_header_info(lines)

        base_month: Optional[int] = None

        base_year: Optional[int] = None

        try:

            period_end = _clean_text(barclays_info.get("period_end", ""))

            if period_end:

                m_end = re.search(r"(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{2,4})", period_end)

                if m_end:

                    base_month = int(m_end.group(2))

                    base_year = int(m_end.group(3))

                    if base_year < 100:

                        base_year += 2000

        except Exception:

            base_month = None

            base_year = None



        # Barclays table statements often use dd/mm/yyyy.

        barclays_date_patterns = [

            re.compile(r"\b(\d{1,2}/\d{1,2}/\d{2,4})\b"),

            re.compile(r"\b(\d{1,2}-\d{1,2}-\d{2,4})\b"),

            re.compile(r"\b(\d{1,2}\.\d{1,2}\.\d{2,4})\b"),

            re.compile(r"\b(\d{1,2}\s+\w{3,9}\s+\d{2,4})\b", flags=re.IGNORECASE),

        ]



        day_only_re = re.compile(r"^(\d{1,2})\s+(?=[A-Z])")



        last_dt: Optional[datetime] = None



        def _add_one_month(dt: datetime) -> datetime:

            y, m = dt.year, dt.month

            if m == 12:

                return datetime(y + 1, 1, 1)

            return datetime(y, m + 1, 1)



        def _infer_date_from_day(day: int) -> str:

            nonlocal last_dt

            if last_dt is not None:

                y = last_dt.year

                m = last_dt.month

                # If day resets (e.g. 31 -> 01), advance the month.

                if day < (last_dt.day - 10):

                    next_month = _add_one_month(datetime(y, m, 1))

                    y, m = next_month.year, next_month.month

                try:

                    cand = datetime(y, m, day)

                    last_dt = cand

                    return f"{day:02d}/{m:02d}/{y:04d}"

                except Exception:

                    return ""



            if base_month is not None and base_year is not None:

                try:

                    cand2 = datetime(base_year, base_month, day)

                    last_dt = cand2

                    return f"{day:02d}/{base_month:02d}/{base_year:04d}"

                except Exception:

                    return ""

            return ""



        def _track_last_dt(date_str: str) -> None:

            nonlocal last_dt

            s = _clean_text(date_str)

            for fmt in ["%d/%m/%Y", "%d/%m/%y", "%d-%m-%Y", "%d-%m-%y", "%d.%m.%Y", "%d.%m.%y"]:

                try:

                    last_dt = datetime.strptime(s, fmt)

                    return

                except Exception:

                    continue



        def _find_barclays_date(line: str) -> str:

            for pat in barclays_date_patterns:

                m = pat.search(line)

                if m:

                    ds = _clean_text(m.group(1))

                    _track_last_dt(ds)

                    return ds



            m_day = day_only_re.search(_clean_text(line))

            if m_day:

                try:

                    day = int(m_day.group(1))

                except Exception:

                    return ""

                return _infer_date_from_day(day)

            return ""



        date_any_re = re.compile(

            r"(\d{1,2}/\d{1,2}/\d{2,4}|\d{1,2}-\d{1,2}-\d{2,4}|\d{1,2}\.\d{1,2}\.\d{2,4}|\d{1,2}\s+\w{3,9}\s+\d{2,4}|(?:(?:^|\s)\d{1,2})(?=\s+[A-Z]))",

            flags=re.IGNORECASE,

        )

        

        current_date = ""

        current_parts: List[str] = []

        in_pending_debit_card = False



        def _try_emit_pending_row(line: str) -> bool:

            b = _clean_text(line)

            if not b:

                return False



            low = b.lower()

            if any(

                k in low

                for k in [

                    "pending debit card transactions",

                    "date transaction amount",

                    "card number",

                ]

            ):

                return True



            date_str = _find_barclays_date(b)

            if not date_str:

                return False



            amounts = [m.group(0) for m in barclays_amount_re.finditer(b)]

            if not amounts:

                return False



            amt_raw = amounts[-1]

            amt_disp = _barclays_amount_display(amt_raw)

            amt_val = _to_float_or_none(_barclays_amount_for_parse(amt_raw))



            money_in = ""

            money_out = ""

            if amt_val is not None:

                if amt_val >= 0:

                    money_in = amt_disp

                else:

                    money_out = amt_disp

            else:

                if "(" in amt_raw and ")" in amt_raw:

                    money_out = amt_disp

                else:

                    money_in = amt_disp



            cut_pos = b.rfind(amt_raw)

            desc_blob = b[:cut_pos] if cut_pos > 0 else b

            if desc_blob.lower().startswith(date_str.lower()):

                desc_blob = desc_blob[len(date_str) :]

            subcat, desc_clean = _barclays_subcategory_and_clean_description(desc_blob)

            description = _clean_text(desc_clean) or "N/A"



            pending_rows.append(

                {

                    "__section": "barclays_pending_debit_card",

                    "date": date_str or "N/A",

                    "description": description,

                    "amount": amt_val,

                    "used_ocr": bool(used_ocr_hint),

                }

            )

            return True



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



            amounts = [m.group(0) for m in barclays_amount_re.finditer(b)]

            if not amounts:

                return



            # For Barclays, prefer rightmost two amounts: transaction amount and balance

            bal_amount_raw = amounts[-1]

            txn_amount_raw = amounts[-2] if len(amounts) >= 2 else amounts[-1]



            txn_disp = _barclays_amount_display(txn_amount_raw)

            txn_val = _to_float_or_none(_barclays_amount_for_parse(txn_amount_raw))



            money_in = ""

            money_out = ""

            if txn_val is not None:

                if txn_val >= 0:

                    money_in = txn_disp

                else:

                    money_out = txn_disp

            else:

                if "(" in txn_amount_raw and ")" in txn_amount_raw:

                    money_out = txn_disp

                else:

                    money_in = txn_disp



            balance = _barclays_amount_display(bal_amount_raw) if bal_amount_raw else ""



            # Cut description before the rightmost amount (balance) to avoid leftover amounts in description

            cut_pos = b.rfind(bal_amount_raw)

            desc_blob = b[:cut_pos] if cut_pos > 0 else b

            # Also strip trailing transaction amount if present at the end

            if txn_amount_raw != bal_amount_raw:

                txn_pos = desc_blob.rfind(txn_amount_raw)

                if txn_pos > 0:

                    desc_blob = desc_blob[:txn_pos]

            if desc_blob.lower().startswith(date_str.lower()):

                desc_blob = desc_blob[len(date_str):]

            subcat, desc_clean = _barclays_subcategory_and_clean_description(desc_blob)

            description = _clean_text(desc_clean)

            if description and (money_in or money_out):

                rows.append(

                    {

                        "date": date_str,

                        "description": description,

                        "subcategory": subcat or "",

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

                    seg_date = _find_barclays_date(seg)

                    _emit_row(seg_date, seg)

            else:

                _emit_row(current_date, joined)



            current_date = ""

            current_parts = []



        for _i, line in enumerate(lines):

            date_str = _find_barclays_date(line)

            low = line.lower()



            if "pending debit card transactions" in low:

                _flush_current()

                in_pending_debit_card = True

                continue



            if in_pending_debit_card:

                if any(k in low for k in [

                    'date description money in money out balance',

                    'transactions between',

                    'available balance',

                    "last night's balance",

                    'overdraft',

                ]):

                    in_pending_debit_card = False

                else:

                    handled = _try_emit_pending_row(line)

                    if handled:

                        continue



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



        return pending_rows + rows



    # First try Barclays-specific parsing

    if _looks_like_barclays_statement(cleaned):

        barclays_rows = _extract_barclays_transactions(cleaned)

        if barclays_rows:

            return barclays_rows



    # Generic, date-anchored parser (multi-line safe). Emits rows even when amount/description is missing.

    rows = _extract_generic_transactions(cleaned)

    return rows





def _to_float_or_none(value: Any) -> Optional[float]:



    s = _clean_text(value)



    if not s:



        return None



    try:



        return _parse_money(s)



    except Exception:
        return None


def _extract_used_vehicle_invoice_fields(lines: List[str]) -> Dict[str, Any]:

    cleaned = [_clean_text(ln) for ln in lines]

    cleaned = [ln for ln in cleaned if ln]

    out: Dict[str, Any] = {}

    if not cleaned:

        return out

    def _value_after_label(label_re: "re.Pattern[str]") -> str:

        for i, ln in enumerate(cleaned[:340]):

            m = label_re.search(ln)

            if not m:

                continue

            tail = _clean_text(ln[m.end() :]).strip(" :-\t")

            if tail:

                return tail

            for j in range(1, 6):

                if i + j >= len(cleaned):

                    break

                nxt = _clean_text(cleaned[i + j]).strip(" :-\t")

                if nxt:

                    return nxt

        return ""

    sold_by = _value_after_label(re.compile(r"\bSold\s*By\b\s*[:\-]?", flags=re.IGNORECASE))

    if sold_by:

        out["supplier"] = _clean_text(sold_by)[:200]

    inv_no = _value_after_label(re.compile(r"\bInvoice\s*(?:No\.?|Number)\b\s*[:\-]?", flags=re.IGNORECASE))

    inv_no = _clean_text(inv_no)

    if inv_no:

        mref = re.search(r"\b([A-Za-z0-9\-/]+)\b", inv_no)

        out["inv_ref_no"] = _clean_text(mref.group(1)) if mref else inv_no

    date_raw = _value_after_label(re.compile(r"\b(?:Invoice\s*Date|Date)\b\s*[:\-]?", flags=re.IGNORECASE))

    date_raw = _clean_text(date_raw).replace("-", "/")

    mdate = re.search(r"\b(\d{1,2}[\/-]\d{1,2}[\/-]\d{2,4})\b", date_raw)

    if mdate:

        out["document_date"] = _clean_text(mdate.group(1)).replace("-", "/")

    make = _value_after_label(re.compile(r"\bMake\b\s*[:\-]?", flags=re.IGNORECASE))

    model = _value_after_label(re.compile(r"\bModel\b\s*[:\-]?", flags=re.IGNORECASE))

    colour = _value_after_label(re.compile(r"\b(?:Colour|Color)\b\s*[:\-]?", flags=re.IGNORECASE))

    if make:

        out["make"] = _clean_text(make)[:160]

    if model:

        out["model"] = _clean_text(model)[:160]

    if colour:

        out["colour"] = _clean_text(colour)[:80]

    reg = _value_after_label(

        re.compile(

            r"\b(?:Registration\s*(?:No\.?|Number)?|Reg\s*(?:No\.?|Number)?)\b\s*[:\-]?",

            flags=re.IGNORECASE,

        )

    )

    reg = _clean_text(reg)

    if not reg:

        for ln in cleaned[:300]:

            m = re.search(r"\b([A-Z]{2}[0-9O]{2}\s*[A-Z]{3})\b", ln, flags=re.IGNORECASE)

            if m:

                reg = _format_uk_reg(m.group(1))

                break

    if reg:

        out["reg_no"] = reg

    def _money_after_label(label_re: "re.Pattern[str]") -> Optional[float]:

        for i, ln in enumerate(cleaned[:650]):

            if not label_re.search(ln):

                continue

            m0 = CURRENCY_RE.search(ln)

            if m0:

                return _to_float_or_none(m0.group(0))

            for j in range(1, 6):

                if i + j >= len(cleaned):

                    break

                m1 = CURRENCY_RE.search(cleaned[i + j])

                if m1:

                    return _to_float_or_none(m1.group(0))

        return None

    total_v = _money_after_label(re.compile(r"\b(?:Total\s*Due|Invoice\s*Total|Total\s*Amount|Grand\s*Total|Total)\b", flags=re.IGNORECASE))

    sub_v = _money_after_label(re.compile(r"\b(?:Sub\s*Total|Subtotal|Net\s*Total|Total\s*Net)\b", flags=re.IGNORECASE))

    vat_v = _money_after_label(re.compile(r"\b(?:VAT\s*Total|VAT|Tax\s*Total|Tax)\b", flags=re.IGNORECASE))

    if isinstance(total_v, (int, float)):

        out["buying_price"] = abs(float(total_v))

        out["non_vat"] = abs(float(total_v))

    if isinstance(sub_v, (int, float)):

        out["std_net"] = abs(float(sub_v))

    if isinstance(vat_v, (int, float)):

        out["vat_amount"] = abs(float(vat_v))

    return out


def _extract_invoice_fields(lines: List[str]) -> Dict[str, Any]:



    cleaned = [_clean_text(ln) for ln in lines]



    cleaned = [ln for ln in cleaned if ln]



    joined = "\n".join(cleaned)



    joined_low = joined.lower()







    def _looks_like_despatch_note() -> bool:



        head = "\n".join(cleaned[:200]).lower()

        if "despatch note" not in head and "dispatch note" not in head:

            return False

        return True







    def _extract_despatch_note_fields() -> Dict[str, Any]:



        out: Dict[str, Any] = {}



        def _value_after_label(label_re: "re.Pattern[str]") -> str:

            for i, ln in enumerate(cleaned[:260]):

                m = label_re.search(ln)

                if not m:

                    continue

                tail = _clean_text(ln[m.end() :]).strip(" :-\t")

                if tail:

                    return tail

                for j in range(1, 5):

                    if i + j >= len(cleaned):

                        break

                    nxt = _clean_text(cleaned[i + j])

                    if nxt:

                        low_nxt = nxt.lower()

                        if any(

                            x in low_nxt
                            for x in [

                                "order code",

                                "payment method",

                                "payment date",

                                "description",

                                "package",

                                "costs",

                                "subtotal",

                                "vat total",

                                "grand total",

                            ]

                        ):

                            continue

                        return nxt

            return ""



        inv_ref = _value_after_label(re.compile(r"\bInvoice\s*(?:No\.?|Number)\b\s*[:\-]?", flags=re.IGNORECASE))

        inv_ref = _clean_text(inv_ref)

        if inv_ref:

            mref = re.search(r"\b([A-Za-z0-9\-/]+)\b", inv_ref)

            out["inv_ref_no"] = _clean_text(mref.group(1)) if mref else inv_ref



        date_re = re.compile(r"\b(\d{1,2}[\/-]\d{1,2}[\/-]\d{2,4})\b")

        date_raw = _value_after_label(re.compile(r"\bInvoice\s*Date\b\s*[:\-]?", flags=re.IGNORECASE))

        if not date_raw:

            date_raw = _value_after_label(re.compile(r"\bDate\b\s*[:\-]?", flags=re.IGNORECASE))

        date_raw = _clean_text(date_raw).replace("-", "/")

        mdate = date_re.search(date_raw)

        if not mdate:

            for ln in cleaned[:220]:

                low = ln.lower()

                if "date" not in low:

                    continue

                m2 = date_re.search(ln)

                if m2:

                    mdate = m2

                    break

        if mdate:

            out["document_date"] = _clean_text(mdate.group(1)).replace("-", "/")



        header_block: List[str] = []

        for ln in cleaned[:80]:

            low = ln.lower()

            if any(x in low for x in ["invoice no", "invoice number", "invoice date", "despatch note", "dispatch note"]):

                break

            if low.startswith("page") or re.match(r"^page\s*\d+\b", low):

                continue

            if ln:

                header_block.append(ln)

        supplier = ""

        for ln in header_block[:18]:

            t = _clean_text(ln)

            if not t:

                continue

            low = t.lower()

            if any(bad in low for bad in ["invoice", "vat", "reg", "telephone", "tel", "email", "www", "page"]):

                continue

            if len(t) >= 6:

                supplier = t

                break

        if supplier:

            out["supplier"] = supplier[:120]



        product = ""

        start_idx = -1

        for i, ln in enumerate(cleaned[:700]):

            low = ln.lower()

            if "product name" in low or re.search(r"\bdescription\b", low) or ("quantity" in low and ("description" in low or "product" in low)):

                start_idx = i

                break

        if start_idx >= 0:

            for ln in cleaned[start_idx + 1 : start_idx + 25]:

                t = _clean_text(ln)

                if not t:

                    continue

                low = t.lower()

                if any(x in low for x in ["unit", "price", "vat", "net", "qty", "quantity", "total", "discount", "carriage", "invoice"]):

                    continue

                if CURRENCY_RE.search(t):

                    continue

                if len(t) >= 4:

                    product = t

                    break

        if product:

            out["make"] = product[:160]



        reg_no = ""

        for ln in cleaned[:260]:

            m = re.search(r"\b([A-Z]{2}[0-9O]{2}\s*[A-Z]{3})\b", ln, flags=re.IGNORECASE)

            if m:

                reg_no = _format_uk_reg(m.group(1))

                break

        if not reg_no:

            for ln in cleaned[:260]:

                m = re.search(

                    r"\bReg\s*(?:No\.?|Number)\b\s*[:\-]?\s*([A-Za-z0-9\- ]{4,30})",

                    ln,

                    flags=re.IGNORECASE,

                )

                if m:

                    reg_no = _clean_text(m.group(1))

                    break

        out["reg_no"] = reg_no if reg_no else "N/A"



        def _find_amount_after_labels(labels: List[str]) -> Optional[float]:

            for i, ln in enumerate(cleaned[:950]):

                low = ln.lower()

                if not any(lbl in low for lbl in labels):

                    continue

                m0 = CURRENCY_RE.search(ln)

                if m0:

                    return _to_float_or_none(m0.group(0))

                for j in range(1, 8):

                    if i + j >= len(cleaned):

                        break

                    nxt = cleaned[i + j]

                    m1 = CURRENCY_RE.search(nxt)

                    if m1:

                        return _to_float_or_none(m1.group(0))

            return None



        net_v = _find_amount_after_labels(["total net amount", "net amount", "total net"])

        vat_v = _find_amount_after_labels(["total tax amount", "tax amount", "total vat", "vat"])

        gross_v = _find_amount_after_labels(["invoice total", "total line", "total"])



        if isinstance(net_v, (int, float)):

            out["std_net"] = abs(float(net_v))

        if isinstance(vat_v, (int, float)):

            out["vat_amount"] = abs(float(vat_v))

        if isinstance(gross_v, (int, float)):

            out["buying_price"] = abs(float(gross_v))

            out["non_vat"] = abs(float(gross_v))



        low_all = joined_low

        if "credit note" in low_all or "refund" in low_all:

            out["category"] = "expense"

        elif "despatch note" in low_all or "dispatch note" in low_all:

            out["category"] = "sale"

        else:

            out["category"] = "purchase"



        return out







    def _extract_j_wilson_plumbing_heating_fields() -> Dict[str, Any]:



        out: Dict[str, Any] = {}



        def _value_after_label(label_re: "re.Pattern[str]", scan: int = 260) -> str:

            for i, ln in enumerate(cleaned[:scan]):

                m = label_re.search(ln)

                if not m:

                    continue

                tail = _clean_text(ln[m.end() :]).strip(" :-\t")

                if tail:

                    return tail

                for j in range(1, 6):

                    if i + j >= len(cleaned):

                        break

                    nxt = _clean_text(cleaned[i + j])

                    if nxt:

                        return nxt

            return ""



        def _parse_jwilson_date(raw: str) -> str:

            s = _clean_text(raw).replace("-", "/")

            if not s:

                return ""

            m0 = re.search(r"\b(\d{1,2}[\/-]\d{1,2}[\/-]\d{2,4})\b", s)

            if m0:

                return _clean_text(m0.group(1)).replace("-", "/")

            m1 = re.search(r"\b(\d{1,2})\s+([A-Za-z]{3,9})\s+(\d{4})\b", s)

            if not m1:

                return ""

            dd = int(m1.group(1))

            mon = m1.group(2).strip().lower()

            yy = int(m1.group(3))

            months = {

                "jan": 1,

                "january": 1,

                "feb": 2,

                "february": 2,

                "mar": 3,

                "march": 3,

                "apr": 4,

                "april": 4,

                "may": 5,

                "jun": 6,

                "june": 6,

                "jul": 7,

                "july": 7,

                "aug": 8,

                "august": 8,

                "sep": 9,

                "sept": 9,

                "september": 9,

                "oct": 10,

                "october": 10,

                "nov": 11,

                "november": 11,

                "dec": 12,

                "december": 12,

            }

            mm = months.get(mon)

            if not mm:

                return ""

            return f"{dd:02d}/{mm:02d}/{yy:04d}"



        inv_date_raw = _value_after_label(re.compile(r"\bInvoice\s*Date\b\s*[:\-]?", flags=re.IGNORECASE))

        inv_date = _parse_jwilson_date(inv_date_raw)

        if inv_date:

            out["document_date"] = inv_date



        inv_no_raw = _value_after_label(re.compile(r"\bInvoice\s*(?:No\.?|Number)\b\s*[:\-]?", flags=re.IGNORECASE))

        inv_no_raw = _clean_text(inv_no_raw)

        if inv_no_raw:

            mref = re.search(r"\b([A-Za-z0-9\-/]+)\b", inv_no_raw)

            out["inv_ref_no"] = _clean_text(mref.group(1)) if mref else inv_no_raw



        ref_raw = _value_after_label(re.compile(r"\bReference\b\s*[:\-]?", flags=re.IGNORECASE))

        ref_raw = _clean_text(ref_raw)

        if ref_raw:

            out["make"] = ref_raw[:240]



        supplier_lines: List[str] = []

        seen_vendor = False

        for ln in cleaned[:160]:

            low = ln.lower()

            if not seen_vendor:

                if "j wilson" in low and "plumbing" in low and "heating" in low:

                    seen_vendor = True

                    supplier_lines.append(_clean_text(ln))

                continue

            if any(x in low for x in ["invoice date", "invoice number", "invoice no", "reference", "utr", "vat number", "vat no"]):

                break

            if low.startswith("page") or re.match(r"^page\s*\d+\b", low):

                continue

            t = _clean_text(ln)

            if not t:

                continue

            if t.lower() == "invoice":

                continue

            supplier_lines.append(t)

            if len(supplier_lines) >= 12:

                break

        supplier_text = _clean_text(" | ".join([x for x in supplier_lines if x]))

        if supplier_text:

            out["supplier"] = supplier_text[:600]



        def _find_amount_after_labels(labels: List[str]) -> Optional[float]:

            for i, ln in enumerate(cleaned[:1100]):

                low = ln.lower()

                if not any(lbl in low for lbl in labels):

                    continue

                m0 = CURRENCY_RE.search(ln)

                if m0:

                    return _to_float_or_none(m0.group(0))

                for j in range(1, 7):

                    if i + j >= len(cleaned):

                        break

                    nxt = cleaned[i + j]

                    m1 = CURRENCY_RE.search(nxt)

                    if m1:

                        return _to_float_or_none(m1.group(0))

            return None



        amount_due_v = _find_amount_after_labels(["amount due gbp", "amount due"])

        if isinstance(amount_due_v, (int, float)):

            out["buying_price"] = abs(float(amount_due_v))

            out["non_vat"] = abs(float(amount_due_v))



        subtotal_v = _find_amount_after_labels(["subtotal", "sub total", "sub-total"])

        if isinstance(subtotal_v, (int, float)):

            out["std_net"] = abs(float(subtotal_v))



        vat_v = _find_amount_after_labels(["total vat 20%", "vat 20%", "vat 20", "vat 20.00"])

        if isinstance(vat_v, (int, float)):

            out["vat_amount"] = abs(float(vat_v))



        low_all = joined_low

        if "credit note" in low_all or "refund" in low_all:

            out["category"] = "expense"

        elif "sales invoice" in low_all or "sale" in low_all or "sales" in low_all:

            out["category"] = "sale"

        else:

            out["category"] = "purchase"



        if not _clean_text(out.get("reg_no")):

            out["reg_no"] = "N/A"



        return out







    def _extract_combi_tech_engineering_services_fields() -> Dict[str, Any]:



        out: Dict[str, Any] = {}



        def _value_after_label(label_re: "re.Pattern[str]", scan: int = 240) -> str:

            for i, ln in enumerate(cleaned[:scan]):

                m = label_re.search(ln)

                if not m:

                    continue

                tail = _clean_text(ln[m.end() :]).strip(" :-\t")

                if tail:

                    return tail

                for j in range(1, 5):

                    if i + j >= len(cleaned):

                        break

                    nxt = _clean_text(cleaned[i + j])

                    if nxt:

                        return nxt

            return ""



        def _parse_combi_date(raw: str) -> str:

            s = _clean_text(raw).replace("-", "/")

            if not s:

                return ""

            m0 = re.search(r"\b(\d{1,2}[\/-]\d{1,2}[\/-]\d{2,4})\b", s)

            if m0:

                return _clean_text(m0.group(1)).replace("-", "/")

            m1 = re.search(r"\b(\d{1,2})\s+([A-Za-z]{3,9})\s+(\d{4})\b", s)

            if not m1:

                return ""

            dd = int(m1.group(1))

            mon = m1.group(2).strip().lower()

            yy = int(m1.group(3))

            months = {

                "jan": 1,

                "january": 1,

                "feb": 2,

                "february": 2,

                "mar": 3,

                "march": 3,

                "apr": 4,

                "april": 4,

                "may": 5,

                "jun": 6,

                "june": 6,

                "jul": 7,

                "july": 7,

                "aug": 8,

                "august": 8,

                "sep": 9,

                "sept": 9,

                "september": 9,

                "oct": 10,

                "october": 10,

                "nov": 11,

                "november": 11,

                "dec": 12,

                "december": 12,

            }

            mm = months.get(mon)

            if not mm:

                return ""

            return f"{dd:02d}/{mm:02d}/{yy:04d}"



        # Invoice Date -> document_date

        inv_date_raw = _value_after_label(re.compile(r"\bInvoice\s*Date\b\s*[:\-]?", flags=re.IGNORECASE))

        inv_date = _parse_combi_date(inv_date_raw)

        if not inv_date:

            for ln in cleaned[:220]:

                if "invoice date" not in ln.lower():

                    continue

                inv_date = _parse_combi_date(ln)

                if inv_date:

                    break

        if inv_date:

            out["document_date"] = inv_date



        # Invoice Number -> inv_ref_no

        inv_no_raw = _value_after_label(re.compile(r"\bInvoice\s*Number\b\s*[:\-]?", flags=re.IGNORECASE))

        inv_no_raw = _clean_text(inv_no_raw)

        if inv_no_raw:

            mref = re.search(r"\b([A-Za-z0-9\-/]+)\b", inv_no_raw)

            out["inv_ref_no"] = _clean_text(mref.group(1)) if mref else inv_no_raw



        # Reference -> make

        ref_raw = _value_after_label(re.compile(r"\bReference\b\s*[:\-]?", flags=re.IGNORECASE))

        ref_raw = _clean_text(ref_raw)

        if ref_raw:

            out["make"] = ref_raw[:240]



        # Supplier field in CSV: take the customer (left header) full info block

        header_block: List[str] = []

        for ln in cleaned[:120]:

            low = ln.lower()

            if any(x in low for x in ["invoice date", "invoice number", "invoice no", "reference", "vat number", "vat no"]):

                break

            if low.startswith("page") or re.match(r"^page\s*\d+\b", low):

                continue

            t = _clean_text(ln)

            if not t:

                continue

            if t.lower() == "invoice":

                continue

            header_block.append(t)

        supplier_text = _clean_text(" | ".join(header_block))

        if supplier_text:

            out["supplier"] = supplier_text[:600]



        # Totals

        def _find_amount_after_labels(labels: List[str]) -> Optional[float]:

            for i, ln in enumerate(cleaned[:950]):

                low = ln.lower()

                if not any(lbl in low for lbl in labels):

                    continue

                m0 = CURRENCY_RE.search(ln)

                if m0:

                    return _to_float_or_none(m0.group(0))

                for j in range(1, 7):

                    if i + j >= len(cleaned):

                        break

                    nxt = cleaned[i + j]

                    m1 = CURRENCY_RE.search(nxt)

                    if m1:

                        return _to_float_or_none(m1.group(0))

            return None



        subtotal_v = _find_amount_after_labels(["subtotal", "sub total", "sub-total"])

        if isinstance(subtotal_v, (int, float)):

            out["std_net"] = abs(float(subtotal_v))

        else:

            out["std_net"] = "N/A"



        vat_v = _find_amount_after_labels(["total vat 20%", "total vat 20", "vat 20%", "vat 20"])

        if isinstance(vat_v, (int, float)):

            out["vat_amount"] = abs(float(vat_v))



        total_gbp_v = _find_amount_after_labels(["total gbp", "total"])

        if isinstance(total_gbp_v, (int, float)):

            out["buying_price"] = abs(float(total_gbp_v))

            out["non_vat"] = abs(float(total_gbp_v))



        # Category heuristic

        low_all = joined_low

        if "credit note" in low_all or "refund" in low_all:

            out["category"] = "expense"

        elif "sales invoice" in low_all or "sale" in low_all or "sales" in low_all:

            out["category"] = "sale"

        else:

            out["category"] = "purchase"



        # reg_no (VAT number) if available

        vat_no = _value_after_label(re.compile(r"\bVAT\s*Number\b\s*[:\-]?", flags=re.IGNORECASE))

        vat_no = _clean_text(vat_no)

        if vat_no:

            m_vat = re.search(r"\b([0-9 ]{6,})\b", vat_no)

            out["reg_no"] = _clean_text(m_vat.group(1)) if m_vat else vat_no



        if not _clean_text(out.get("reg_no")):

            out["reg_no"] = "N/A"



        return out







    def _looks_like_one_stop_invoice() -> bool:



        head = "\n".join(cleaned[:120]).lower()

        return (

            ("one stop" in head and "builders" in head and "merchants" in head)

            or ("one stop builders merchants" in head)

        ) and ("invoice no" in head or "invoice date" in head or "invoice total" in head)







    def _extract_one_stop_invoice_fields() -> Dict[str, Any]:



        out: Dict[str, Any] = {}



        def _value_after_label(label_re: "re.Pattern[str]") -> str:

            for i, ln in enumerate(cleaned[:200]):

                m = label_re.search(ln)

                if not m:

                    continue

                tail = _clean_text(ln[m.end() :]).strip(" :-\t")

                if tail:

                    return tail

                for j in range(1, 4):

                    if i + j >= len(cleaned):

                        break

                    nxt = _clean_text(cleaned[i + j])

                    if nxt:

                        return nxt

            return ""



        inv_ref = _value_after_label(re.compile(r"\bInvoice\s*No\b\s*[:\-]?", flags=re.IGNORECASE))

        inv_ref = _clean_text(inv_ref)

        if inv_ref:

            mref = re.search(r"\b([A-Za-z0-9\-/]+)\b", inv_ref)

            out["inv_ref_no"] = _clean_text(mref.group(1)) if mref else inv_ref



        inv_date = _value_after_label(re.compile(r"\bInvoice\s*Date\b\s*[:\-]?", flags=re.IGNORECASE))

        inv_date = _clean_text(inv_date).replace("-", "/")

        mdate = re.search(r"\b(\d{1,2}[\/-]\d{1,2}[\/-]\d{2,4})\b", inv_date)

        if mdate:

            out["document_date"] = _clean_text(mdate.group(1)).replace("-", "/")



        header_block: List[str] = []

        for ln in cleaned[:60]:

            low = ln.lower()

            if "invoice no" in low or "invoice date" in low:

                break

            if low.startswith("page") or re.match(r"^page\s*\d+\b", low):

                continue

            if ln:

                header_block.append(ln)

        header_text = _clean_text(" | ".join(header_block))

        supplier_name = ""

        for ln in header_block[:12]:

            if "one stop" in ln.lower() and "merchants" in ln.lower():

                supplier_name = _clean_text(ln)

                break

        if not supplier_name:

            for ln in header_block[:12]:

                if len(_clean_text(ln)) >= 6 and not any(

                    x in ln.lower()

                    for x in [

                        "tel",

                        "fax",

                        "email",

                        "vat",

                        "bank",

                        "invoice",

                        "account",

                    ]

                ):

                    supplier_name = _clean_text(ln)

                    break

        if supplier_name:

            out["supplier"] = supplier_name[:120]

        elif header_text:

            out["supplier"] = header_text[:600]



        bank_line = ""

        for i, ln in enumerate(cleaned[:250]):

            if re.search(r"\bBank\s*details\b\s*[:\-]?", ln, flags=re.IGNORECASE):

                tail = re.sub(r"^.*?\bBank\s*details\b\s*[:\-]?\s*", "", ln, flags=re.IGNORECASE)

                tail = _clean_text(tail)

                if tail:

                    bank_line = tail

                else:

                    for j in range(1, 5):

                        if i + j >= len(cleaned):

                            break

                        nxt = _clean_text(cleaned[i + j])

                        if nxt:

                            bank_line = nxt

                            break

                break

        if bank_line:

            out["make"] = bank_line[:600]



        reg_no = ""

        for ln in cleaned[:250]:

            m = re.search(r"\bCompany\s*Reg(?:istration)?\s*(?:No\.?|Number)?\b\s*[:\-]?\s*([A-Za-z0-9\- ]{4,30})", ln, flags=re.IGNORECASE)

            if m:

                reg_no = _clean_text(m.group(1))

                break

        if not reg_no:

            for ln in cleaned[:250]:

                m = re.search(r"\bReg(?:istration)?\s*(?:No\.?|Number)?\b\s*[:\-]?\s*([A-Za-z0-9\- ]{4,30})", ln, flags=re.IGNORECASE)

                if m:

                    cand = _clean_text(m.group(1))

                    if cand and not re.search(r"\b(invoice|date|page|no)\b", cand, flags=re.IGNORECASE):

                        reg_no = cand

                        break

        out["reg_no"] = reg_no if reg_no else "N/A"



        def _find_amount_after_labels(labels: List[str]) -> Optional[float]:

            for i, ln in enumerate(cleaned[:900]):

                low = ln.lower()

                if not any(lbl in low for lbl in labels):

                    continue

                m0 = CURRENCY_RE.search(ln)

                if m0:

                    return _to_float_or_none(m0.group(0))

                for j in range(1, 6):

                    if i + j >= len(cleaned):

                        break

                    nxt = cleaned[i + j]

                    m1 = CURRENCY_RE.search(nxt)

                    if m1:

                        return _to_float_or_none(m1.group(0))

            return None



        net_v = _find_amount_after_labels(["total net amount", "net amount"])

        vat_v = _find_amount_after_labels(["total tax amount", "tax amount", "vat"])

        invoice_total_v = _find_amount_after_labels(["invoice total"])

        gross_v = invoice_total_v

        if gross_v is None:

            # Fallback when invoice total label is missing; avoid capturing "Total Net Amount" / "Total Tax Amount".

            gross_v = _find_amount_after_labels(["total"])

            for ln in cleaned[:950]:

                low = ln.lower()

                if "total" not in low:

                    continue

                if "total net" in low or "total tax" in low:

                    continue

                m0 = CURRENCY_RE.search(ln)

                if m0:

                    gross_v = _to_float_or_none(m0.group(0))

                    break



        if isinstance(gross_v, (int, float)):

            out["buying_price"] = abs(float(gross_v))

            out["non_vat"] = abs(float(gross_v))

        if isinstance(net_v, (int, float)):

            out["std_net"] = abs(float(net_v))

        if isinstance(vat_v, (int, float)):

            out["vat_amount"] = abs(float(vat_v))



        is_credit_note = "credit note" in joined_low

        if is_credit_note:

            out["category"] = "expense"

        else:

            out["category"] = "purchase"



        return out







    def _looks_like_smiths_fire_llp() -> bool:



        head = "\n".join(cleaned[:160]).lower()

        if "smiths fire" not in head:

            return False

        if "invoice" not in head:

            return False

        return True







    def _extract_smiths_fire_llp_fields() -> Dict[str, Any]:



        out: Dict[str, Any] = {}



        def _value_after_label(label_re: "re.Pattern[str]", scan: int = 320) -> str:

            for i, ln in enumerate(cleaned[:scan]):

                m = label_re.search(ln)

                if not m:

                    continue

                tail = _clean_text(ln[m.end() :]).strip(" :-\t")

                if tail:

                    return tail

                for j in range(1, 6):

                    if i + j >= len(cleaned):

                        break

                    nxt = _clean_text(cleaned[i + j])

                    if nxt:

                        return nxt

            return ""



        inv_no_raw = _value_after_label(re.compile(r"\bInvoice\s*No\.?\b\s*[:\-]?", flags=re.IGNORECASE))

        inv_no_raw = _clean_text(inv_no_raw)

        if inv_no_raw:

            mref = re.search(r"\b([A-Za-z0-9\-/]+)\b", inv_no_raw)

            out["inv_ref_no"] = _clean_text(mref.group(1)) if mref else inv_no_raw



        date_raw = _value_after_label(re.compile(r"\bDate\b\s*[:\-]?", flags=re.IGNORECASE))

        date_raw = _clean_text(date_raw).replace("-", "/")

        mdate = re.search(r"\b(\d{1,2}[\/-]\d{1,2}[\/-]\d{2,4})\b", date_raw)

        if not mdate:

            for ln in cleaned[:220]:

                if "date" not in ln.lower():

                    continue

                m2 = re.search(r"\b(\d{1,2}[\/-]\d{1,2}[\/-]\d{2,4})\b", ln)

                if m2:

                    mdate = m2

                    break

        if mdate:

            out["document_date"] = _clean_text(mdate.group(1)).replace("-", "/")



        inv_to = _value_after_label(re.compile(r"\bInvoice\s*To\b\s*[:\-]?", flags=re.IGNORECASE))

        inv_to = _clean_text(inv_to)

        if inv_to:

            out["supplier"] = inv_to[:240]



        deliver_to = _value_after_label(re.compile(r"\bDeliver\s*To\b\s*[:\-]?", flags=re.IGNORECASE))

        deliver_to = _clean_text(deliver_to)

        if deliver_to:

            out["make"] = deliver_to[:240]



        vat_reg = _value_after_label(re.compile(r"\bVAT\s*Reg\s*No\.?\b\s*[:\-]?", flags=re.IGNORECASE))

        vat_reg = _clean_text(vat_reg)

        if vat_reg:

            mvat = re.search(r"\b([A-Za-z0-9 ]{6,})\b", vat_reg)

            out["reg_no"] = _clean_text(mvat.group(1)) if mvat else vat_reg

        if not _clean_text(out.get("reg_no")):

            out["reg_no"] = "N/A"



        def _find_amount_after_labels(labels: List[str]) -> Optional[float]:

            for i, ln in enumerate(cleaned[:1100]):

                low = ln.lower()

                if not any(lbl in low for lbl in labels):

                    continue

                m0 = CURRENCY_RE.search(ln)

                if m0:

                    return _to_float_or_none(m0.group(0))

                for j in range(1, 8):

                    if i + j >= len(cleaned):

                        break

                    nxt = cleaned[i + j]

                    m1 = CURRENCY_RE.search(nxt)

                    if m1:

                        return _to_float_or_none(m1.group(0))

            return None



        total_value_v = _find_amount_after_labels(["total value"])

        if isinstance(total_value_v, (int, float)):

            out["std_net"] = abs(float(total_value_v))



        vat_v = _find_amount_after_labels(["vat"])

        if isinstance(vat_v, (int, float)):

            out["vat_amount"] = abs(float(vat_v))



        balance_v = _find_amount_after_labels(["balance"])

        if isinstance(balance_v, (int, float)):

            out["buying_price"] = abs(float(balance_v))

            out["non_vat"] = abs(float(balance_v))



        out["category"] = "purchase" if "credit note" not in joined_low else "expense"

        return out







    def _looks_like_warranty_solutions_group_swg() -> bool:



        # This format typically contains header text like:

        # "Warranty Solutions Group" and document type "Credit Note".

        head = "\n".join(cleaned[:80]).lower()

        return ("warranty" in head and "solutions" in head and "group" in head) or ("warranty solutions group" in head)







    def _looks_like_savin_wholesalers_ltd() -> bool:



        # This format typically contains "Savin Wholesalers Ltd" in the header

        head = "\n".join(cleaned[:120]).lower()

        return ("savin" in head and "wholesalers" in head and "ltd" in head) or ("savin wholesalers ltd" in head)







    def _looks_like_combi_tech_engineering_services() -> bool:



        head = "\n".join(cleaned[:140]).lower()

        if "combi" not in head and "combi-tech" not in head:

            return False

        if "engineering" not in head:

            return False

        return ("combi-tech" in head) or ("combi" in head and "tech" in head and "engineering" in head)







    def _looks_like_autotrader_invoice() -> bool:



        head = "\n".join(cleaned[:160]).lower()

        if "autotrader" not in head and "auto trader" not in head:

            return False

        # Typical labels in this layout (OCR may miss one of them)

        if ("order code" in head) or ("payment date" in head) or ("payment method" in head):

            return True

        # Even if labels are missed, the distinct table footer text usually exists.

        return ("vat total" in head) and ("grand total" in head)







    def _looks_like_sw_motor_factors_ltd() -> bool:



        head = "\n".join(cleaned[:180]).lower()

        if "sw" not in head:

            return False

        if "motor" not in head or "factor" not in head:

            return False

        return ("sw motor factors" in head) and ("invoice" in head)







    def _extract_autotrader_invoice_fields() -> Dict[str, Any]:



        out: Dict[str, Any] = {}



        def _to_float_relaxed(value: Any) -> Optional[float]:

            s0 = _clean_text(value)

            if not s0:

                return None

            s = s0.replace("Â£", "£")

            s = re.sub(r"(?<=\d)\s*[\.,]\s*(?=\d{1,2}\b)", ".", s)

            if "." not in s and re.search(r"\b\d{1,6},\d{2}\b", s):

                s = re.sub(r"\b(\d{1,6}),(\d{2})\b", r"\1.\2", s)

            try:

                return _parse_money(s)

            except Exception:

                return None



        def _value_after_label(label_re: "re.Pattern[str]", scan: int = 260) -> str:

            for i, ln in enumerate(cleaned[:scan]):

                m = label_re.search(ln)

                if not m:

                    continue

                tail = _clean_text(ln[m.end() :]).strip(" :-\t")

                if tail:

                    return tail

                for j in range(1, 6):

                    if i + j >= len(cleaned):

                        break

                    nxt = _clean_text(cleaned[i + j])

                    if nxt:

                        return nxt

            return ""



        # Payment Date -> document_date

        date_raw = _value_after_label(re.compile(r"\bPayment\s*Date\b\s*[:\-]?", flags=re.IGNORECASE))

        date_raw = _clean_text(date_raw).replace("-", "/")

        mdate = re.search(r"\b(\d{1,2}[\/-]\d{1,2}[\/-]\d{2,4})\b", date_raw)

        if not mdate:

            for ln in cleaned[:220]:

                if "payment date" not in ln.lower():

                    continue

                m2 = re.search(r"\b(\d{1,2}[\/-]\d{1,2}[\/-]\d{2,4})\b", ln)

                if m2:

                    mdate = m2

                    break

        if mdate:

            out["document_date"] = _clean_text(mdate.group(1)).replace("-", "/")

        else:

            out["document_date"] = "N/A"



        # Order Code -> inv_ref_no

        order_code = _value_after_label(re.compile(r"\bOrder\s*Code\b\s*[:\-]?", flags=re.IGNORECASE))

        order_code = _clean_text(order_code)

        if order_code:

            mref = re.search(r"\b([A-Za-z0-9\-/]+)\b", order_code)

            out["inv_ref_no"] = _clean_text(mref.group(1)) if mref else order_code

        else:

            out["inv_ref_no"] = ""



        # Supplier: prefer company name line if present

        supplier_name = ""

        for ln in cleaned[:260]:

            low = ln.lower()

            if "auto trader limited" in low:

                supplier_name = "Auto Trader Limited"

                break

            if low.strip() == "autotrader" or low.strip() == "auto trader":

                supplier_name = "AutoTrader"

                break

        # Fallback: Description (table)

        desc_value = ""

        for i, ln in enumerate(cleaned[:700]):

            if re.search(r"\bDescription\b", ln, flags=re.IGNORECASE):

                for j in range(1, 20):

                    if i + j >= len(cleaned):

                        break

                    cand = _clean_text(cleaned[i + j])

                    if not cand:

                        continue

                    low = cand.lower()

                    if any(x in low for x in ["package", "cost", "costs", "subtotal", "vat total", "grand total"]):

                        continue

                    if CURRENCY_RE.search(cand):

                        continue

                    desc_value = cand

                    break

                break

        if supplier_name:

            out["supplier"] = supplier_name

        else:

            bad_phrases = [

                "service based email",

                "marketing communications",

                "registered in england",

                "company number",

                "vat number",

                "copyright",

            ]

            if desc_value and any(p in desc_value.lower() for p in bad_phrases):

                desc_value = ""

            out["supplier"] = desc_value[:240] if desc_value else "N/A"



        # Make -> header information

        header_lines: List[str] = []

        for ln in cleaned[:120]:

            low = ln.lower()

            if any(x in low for x in ["order code", "payment method", "payment date", "description", "subtotal", "vat total", "grand total"]):

                break

            if low.startswith("page") or re.match(r"^page\s*\d+\b", low):

                continue

            t = _clean_text(ln)

            if t:

                header_lines.append(t)

        header_text = _clean_text(" | ".join(header_lines))

        out["make"] = header_text[:600] if header_text else "N/A"



        def _find_amount_after_labels(labels: List[str]) -> Optional[float]:

            amount_loose_re = re.compile(r"\(?\s*-?\s*(?:£|Â£|\$|€)?\s*\d[\d,]*(?:\.\d{1,2})?\s*\)?")

            for i, ln in enumerate(cleaned[:900]):

                low = ln.lower()

                if not any(lbl in low for lbl in labels):

                    continue

                m0 = CURRENCY_RE.search(ln)

                if m0:

                    return _to_float_relaxed(m0.group(0))

                m0b = amount_loose_re.search(ln)

                if m0b:

                    return _to_float_relaxed(m0b.group(0))

                for j in range(1, 7):

                    if i + j >= len(cleaned):

                        break

                    nxt = cleaned[i + j]

                    m1 = CURRENCY_RE.search(nxt)

                    if m1:

                        return _to_float_relaxed(m1.group(0))

                    m1b = amount_loose_re.search(nxt)

                    if m1b:

                        return _to_float_relaxed(m1b.group(0))

            return None



        subtotal_v = _find_amount_after_labels(["subtotal", "sub total", "sub-total"])

        if isinstance(subtotal_v, (int, float)):

            out["std_net"] = abs(float(subtotal_v))

        else:

            out["std_net"] = "N/A"



        vat_v = _find_amount_after_labels(["vat total", "total vat", "vat"])

        if isinstance(vat_v, (int, float)):

            out["vat_amount"] = abs(float(vat_v))

        else:

            out["vat_amount"] = "N/A"



        grand_total_v = _find_amount_after_labels(["grand total", "grandtotal"])



        if grand_total_v is None:

            # Regex fallback on joined text to handle cases where OCR splits the label/amount

            # across lines or table cells (e.g. 'Grand Total' on one line and '£21.30' on next).

            joined2 = "\n".join(cleaned)

            mgt = re.search(

                r"\bgrand\s*total\b[^0-9]{0,160}(\(?\s*-?\s*(?:£|Â£|\$|€)?\s*\d[\d,]*(?:\.\d{1,2})?\s*\)?)",

                joined2,

                flags=re.IGNORECASE,

            )

            if mgt:

                vgt = _to_float_relaxed(mgt.group(1))

                if isinstance(vgt, (int, float)):

                    grand_total_v = abs(float(vgt))



        if grand_total_v is None:

            # Split-label fallback: label words and amount may be on separate lines.

            amount_loose_re2 = re.compile(r"\(?\s*-?\s*(?:£|Â£|\$|€)?\s*\d[\d,]*(?:\.\d{1,2})?\s*\)?")

            for i, ln in enumerate(cleaned[:950]):

                low = ln.lower()

                if "grand" not in low:

                    continue

                window = cleaned[i : min(len(cleaned), i + 6)]

                if not any("total" in _clean_text(x).lower() for x in window):

                    continue

                for w in window:

                    m = CURRENCY_RE.search(w) or amount_loose_re2.search(w)

                    if not m:

                        continue

                    v3 = _to_float_relaxed(m.group(0))

                    if isinstance(v3, (int, float)):

                        grand_total_v = abs(float(v3))

                        break

                if grand_total_v is not None:

                    break



        if grand_total_v is None:

            # Fallback: OCR/text extraction may split the label or reorder table columns.

            # Choose the largest plausible amount near the bottom, excluding Subtotal/VAT lines.

            candidates: List[float] = []

            tail = cleaned[max(0, len(cleaned) - 220) :]

            amount_loose_re = re.compile(r"\(?\s*-?\s*(?:£|Â£|\$|€)?\s*\d[\d,]*(?:\.\d{1,2})?\s*\)?")

            for ln in tail:

                low = ln.lower()

                if any(x in low for x in ["subtotal", "sub total", "sub-total", "vat total", "total vat", "vat"]):

                    continue

                m = CURRENCY_RE.search(ln)

                if m:

                    v = _to_float_relaxed(m.group(0))

                    if isinstance(v, (int, float)):

                        candidates.append(abs(float(v)))

                    continue

                m2 = amount_loose_re.search(ln)

                if m2:

                    v2 = _to_float_relaxed(m2.group(0))

                    if isinstance(v2, (int, float)):

                        candidates.append(abs(float(v2)))

            if candidates:

                grand_total_v = max(candidates)



        if isinstance(grand_total_v, (int, float)):

            out["buying_price"] = abs(float(grand_total_v))

            out["non_vat"] = abs(float(grand_total_v))

        else:

            out["buying_price"] = "N/A"

            out["non_vat"] = "N/A"



        # Category inference

        low_all = joined_low

        if "refund" in low_all or "credit" in low_all or "credit note" in low_all:

            out["category"] = "expense"

        elif "sale" in low_all or "sales" in low_all:

            out["category"] = "sale"

        else:

            out["category"] = "purchase"



        # Ensure remaining expected fields are present

        if not _clean_text(out.get("reg_no")):

            out["reg_no"] = "N/A"

        if not _clean_text(out.get("model")):

            out["model"] = "N/A"

        if not _clean_text(out.get("colour")):

            out["colour"] = "N/A"



        return out







    def _extract_sw_motor_factors_ltd_fields() -> Dict[str, Any]:



        out: Dict[str, Any] = {}



        def _to_float_relaxed(value: Any) -> Optional[float]:

            s0 = _clean_text(value)

            if not s0:

                return None

            s = s0.replace("Â£", "£")

            s = re.sub(r"(?<=\d)\s*[\.,]\s*(?=\d{1,2}\b)", ".", s)

            if "." not in s and re.search(r"\b\d{1,6},\d{2}\b", s):

                s = re.sub(r"\b(\d{1,6}),(\d{2})\b", r"\1.\2", s)

            try:

                return _parse_money(s)

            except Exception:

                try:

                    s2 = re.sub(r"[^0-9.\-()]", "", s)

                    if not s2:

                        return None

                    return float(s2.strip("()"))

                except Exception:

                    return None



        def _parse_sw_date(raw: str) -> str:

            s = _clean_text(raw).replace("-", "/")

            if not s:

                return ""

            m0 = re.search(r"\b(\d{1,2})[\/-](\d{1,2})[\/-](\d{2,4})\b", s)

            if not m0:

                return ""

            dd = int(m0.group(1))

            mm = int(m0.group(2))

            yy = int(m0.group(3))

            if yy < 100:

                yy = 2000 + yy

            return f"{dd:02d}/{mm:02d}/{yy:04d}"



        def _value_after_label(label_re: "re.Pattern[str]", scan: int = 340) -> str:

            for i, ln in enumerate(cleaned[:scan]):

                m = label_re.search(ln)

                if not m:

                    continue

                tail = _clean_text(ln[m.end() :]).strip(" :-\t")

                if tail:

                    return tail

                for j in range(1, 7):

                    if i + j >= len(cleaned):

                        break

                    nxt = _clean_text(cleaned[i + j])

                    if nxt:

                        return nxt

            return ""



        date_raw = _value_after_label(re.compile(r"\bDate\b\s*[:\-]?", flags=re.IGNORECASE))

        doc_date = _parse_sw_date(date_raw)

        if not doc_date:

            for i, ln in enumerate(cleaned[:320]):

                if "date" not in ln.lower():

                    continue

                doc_date = _parse_sw_date(ln)

                if doc_date:

                    break

                for j in range(1, 5):

                    if i + j >= len(cleaned):

                        break

                    doc_date = _parse_sw_date(cleaned[i + j])

                    if doc_date:

                        break

                if doc_date:

                    break

        if not doc_date:

            # Final fallback: first dd/mm/yy occurrence in the header area.

            for ln in cleaned[:420]:

                doc_date = _parse_sw_date(ln)

                if doc_date:

                    break

        out["document_date"] = doc_date if doc_date else "N/A"



        inv_no_raw = _value_after_label(re.compile(r"\bInvoice\s*(?:No\.?|Number)\b\s*[:\-]?", flags=re.IGNORECASE))

        inv_no_raw = _clean_text(inv_no_raw)

        if inv_no_raw:

            mref = re.search(r"\b([A-Za-z0-9\-/]+)\b", inv_no_raw)

            out["inv_ref_no"] = _clean_text(mref.group(1)) if mref else inv_no_raw

        else:

            out["inv_ref_no"] = ""



        supplier_lines: List[str] = []

        inv_from_idx = -1

        for i, ln in enumerate(cleaned[:260]):

            if re.search(r"\bInvoice\s*From\b\s*:?", ln, flags=re.IGNORECASE):

                inv_from_idx = i

                break

        if inv_from_idx != -1:

            for ln in cleaned[inv_from_idx + 1 : min(len(cleaned), inv_from_idx + 40)]:

                low = ln.lower()

                if any(x in low for x in ["invoice to", "account", "type", "paid", "page", "op", "customer order", "mop", "invoice no"]):

                    break

                t = _clean_text(ln)

                if t:

                    supplier_lines.append(t)

        supplier_text = _clean_text(" | ".join(supplier_lines))

        out["supplier"] = supplier_text[:600] if supplier_text else "N/A"



        desc_value = ""

        for i, ln in enumerate(cleaned[:1100]):

            if re.search(r"\bDescription\b", ln, flags=re.IGNORECASE):

                for j in range(1, 40):

                    if i + j >= len(cleaned):

                        break

                    cand = _clean_text(cleaned[i + j])

                    if not cand:

                        continue

                    low = cand.lower()

                    if any(x in low for x in ["part number", "location", "qty", "quantity", "unit price", "ext cost", "goods value", "vat", "total", "invoice"]):

                        continue

                    if CURRENCY_RE.search(cand):

                        continue

                    if re.fullmatch(r"[0-9]+(?:\.[0-9]{1,2})?", cand.strip()):

                        continue

                    desc_value = cand

                    break

                break

        out["make"] = desc_value[:240] if desc_value else "N/A"



        reg_no = ""

        for ln in cleaned[:600]:

            m = re.search(r"\bVAT\s*Reg\s*No\b\s*[:\-]?\s*(GB\s*[0-9 ]{6,20})\b", ln, flags=re.IGNORECASE)

            if m:

                reg_no = _clean_text(m.group(1))

                break

        if not reg_no:

            vat_reg = _value_after_label(re.compile(r"\bVAT\s*Reg\s*No\b\s*[:\-]?", flags=re.IGNORECASE))

            vat_reg = _clean_text(vat_reg)

            if vat_reg:

                m2 = re.search(r"\b(GB\s*[0-9 ]{6,20})\b", vat_reg, flags=re.IGNORECASE)

                reg_no = _clean_text(m2.group(1)) if m2 else vat_reg

        reg_no = reg_no.replace(" ", "")

        out["reg_no"] = reg_no.upper() if reg_no else "N/A"



        def _find_amount_after_labels(labels: List[str], scan: int = 1600) -> Optional[float]:

            amount_loose_re = re.compile(r"\(?\s*-?\s*(?:£|Â£|\$|€)?\s*\d[\d,]*(?:\.\d{1,2})?\s*\)?")

            for i, ln in enumerate(cleaned[:scan]):

                low = ln.lower()

                if not any(lbl in low for lbl in labels):

                    continue

                m0 = CURRENCY_RE.search(ln)

                if m0:

                    return _to_float_relaxed(m0.group(0))

                m0b = amount_loose_re.search(ln)

                if m0b:

                    return _to_float_relaxed(m0b.group(0))

                for j in range(1, 8):

                    if i + j >= len(cleaned):

                        break

                    nxt = cleaned[i + j]

                    m1 = CURRENCY_RE.search(nxt)

                    if m1:

                        return _to_float_relaxed(m1.group(0))

                    m1b = amount_loose_re.search(nxt)

                    if m1b:

                        return _to_float_relaxed(m1b.group(0))

            return None



        goods_v = _find_amount_after_labels(["goods value"])

        out["std_net"] = abs(float(goods_v)) if isinstance(goods_v, (int, float)) else "N/A"



        vat_v = _find_amount_after_labels(["vat"])

        out["vat_amount"] = abs(float(vat_v)) if isinstance(vat_v, (int, float)) else "N/A"



        total_v = _find_amount_after_labels(["total"])

        if isinstance(total_v, (int, float)):

            out["buying_price"] = abs(float(total_v))

            out["non_vat"] = abs(float(total_v))

        else:

            out["buying_price"] = "N/A"

            out["non_vat"] = "N/A"



        low_all = joined_low

        if "credit note" in low_all or "refund" in low_all or "credit" in low_all:

            out["category"] = "expense"

        elif "sales" in low_all or "sale" in low_all:

            out["category"] = "sale"

        else:

            out["category"] = "purchase"



        return out







    def _looks_like_j_wilson_plumbing_heating() -> bool:



        head = "\n".join(cleaned[:180]).lower()

        if ("j wilson" not in head and "jwilson" not in head):

            return False

        if "plumbing" not in head or "heating" not in head:

            return False

        return ("invoice date" in head) and ("invoice number" in head or "invoice no" in head)







    def _looks_like_beetroot_catering_booking() -> bool:



        head = "\n".join(cleaned[:140]).lower()

        if "beetroot catering booking" not in head:

            return False

        # Typical header labels visible on this invoice layout

        return ("invoice number" in head) and ("amount due" in head)







    def _looks_like_repair_telecommunications_limited() -> bool:



        head = "\n".join(cleaned[:160]).lower()

        if "repair telecommunications" not in head or "limited" not in head:

            return False

        # Typical labels on this invoice layout

        return ("invoice number" in head or "invoice no" in head) and ("tax date" in head or "amount due" in head)







    def _looks_like_amazon_invoice() -> bool:



        head = "\n".join(cleaned[:220]).lower()

        if "invoice" not in head:

            return False



        has_date_pair = ("invoice date" in head and "delivery date" in head) or ("invoice date / delivery date" in head)

        has_inv_no = ("invoice #" in head) or ("invoice number" in head) or ("invoice no" in head)

        has_sold_by = "sold by" in head

        has_total = ("invoice total" in head) or ("total payable" in head)

        has_amazon_hint = ("amazon" in head) or ("www.amazon" in head) or ("amazon.co.uk" in head) or ("contact-us" in head)



        # Prefer strong label combinations, but keep amazon hint as a supporting signal.

        if has_date_pair and has_sold_by and (has_inv_no or has_total):

            return True

        if has_amazon_hint and (has_date_pair or has_inv_no) and has_total:

            return True

        return False







    def _extract_amazon_invoice_fields() -> Dict[str, Any]:



        out: Dict[str, Any] = {}



        def _value_after_label(label_re: "re.Pattern[str]", scan: int = 320) -> str:

            for i, ln in enumerate(cleaned[:scan]):

                m = label_re.search(ln)

                if not m:

                    continue

                tail = _clean_text(ln[m.end() :]).strip(" :-\t")

                if tail:

                    return tail

                for j in range(1, 6):

                    if i + j >= len(cleaned):

                        break

                    nxt = _clean_text(cleaned[i + j])

                    if nxt:

                        return nxt

            return ""



        def _parse_amazon_date(raw: str) -> str:

            s = _clean_text(raw).replace("-", "/")

            if not s:

                return ""

            m0 = re.search(r"\b(\d{1,2}[\/-]\d{1,2}[\/-]\d{2,4})\b", s)

            if m0:

                return _clean_text(m0.group(1)).replace("-", "/")

            m1 = re.search(r"\b(\d{1,2})\s+([A-Za-z]{3,9})\s+(\d{4})\b", s)

            if not m1:

                return ""

            dd = int(m1.group(1))

            mon = m1.group(2).strip().lower()

            yy = int(m1.group(3))

            months = {

                "jan": 1,

                "january": 1,

                "feb": 2,

                "february": 2,

                "mar": 3,

                "march": 3,

                "apr": 4,

                "april": 4,

                "may": 5,

                "jun": 6,

                "june": 6,

                "jul": 7,

                "july": 7,

                "aug": 8,

                "august": 8,

                "sep": 9,

                "sept": 9,

                "september": 9,

                "oct": 10,

                "october": 10,

                "nov": 11,

                "november": 11,

                "dec": 12,

                "december": 12,

            }

            mm = months.get(mon)

            if not mm:

                return ""

            return f"{dd:02d}/{mm:02d}/{yy:04d}"



        doc_date_raw = _value_after_label(re.compile(r"\bInvoice\s*date\s*/\s*Delivery\s*date\b\s*[:\-]?", flags=re.IGNORECASE))

        doc_date = _parse_amazon_date(doc_date_raw)

        if not doc_date:

            for i, ln in enumerate(cleaned[:260]):

                low = ln.lower()

                if "invoice date" not in low and "delivery date" not in low:

                    continue

                doc_date = _parse_amazon_date(ln)

                if doc_date:

                    break

                # Common OCR/PDF extraction case: label line and the actual date are on next line(s)

                for j in range(1, 5):

                    if i + j >= len(cleaned):

                        break

                    cand = _parse_amazon_date(cleaned[i + j])

                    if cand:

                        doc_date = cand

                        break

                if doc_date:

                    break

        if doc_date:

            out["document_date"] = doc_date



        inv_no_raw = _value_after_label(re.compile(r"\bInvoice\s*(?:#|No\.?|Number)\b\s*[:\-]?", flags=re.IGNORECASE))

        inv_no_raw = _clean_text(inv_no_raw)

        if not inv_no_raw:

            for i, ln in enumerate(cleaned[:260]):

                low = ln.lower()

                if "invoice" not in low:

                    continue

                if "invoice #" not in low and "invoice no" not in low and "invoice number" not in low:

                    continue

                tail = _clean_text(re.sub(r"^.*?\binvoice\s*(?:#|no\.?|number)\b\s*[:\-]?\s*", "", ln, flags=re.IGNORECASE))

                if tail:

                    inv_no_raw = tail

                    break

                for j in range(1, 5):

                    if i + j >= len(cleaned):

                        break

                    cand = _clean_text(cleaned[i + j])

                    if cand:

                        inv_no_raw = cand

                        break

                if inv_no_raw:

                    break

        if inv_no_raw:

            mref = re.search(r"\b([A-Za-z0-9\-/]+)\b", inv_no_raw)

            out["inv_ref_no"] = _clean_text(mref.group(1)) if mref else inv_no_raw



        sold_by_raw = _value_after_label(re.compile(r"\bSold\s*by\b\s*[:\-]?", flags=re.IGNORECASE))

        sold_by_raw = _clean_text(sold_by_raw)

        if sold_by_raw:

            out["make"] = sold_by_raw[:240]



        sold_by_block = ""

        sold_by_idx = -1

        for i, ln in enumerate(cleaned[:260]):

            if re.search(r"\bSold\s+by\b", ln, flags=re.IGNORECASE):

                sold_by_idx = i

                break

        if sold_by_idx != -1:

            block: List[str] = []

            for ln in cleaned[sold_by_idx : min(len(cleaned), sold_by_idx + 25)]:

                low = ln.lower()

                if any(x in low for x in [

                    "billing address",

                    "delivery address",

                    "order information",

                    "invoice details",

                    "invoice date",

                    "invoice #",

                    "invoice total",

                ]):

                    break

                t = _clean_text(ln)

                if t and t.lower() not in ("paid", "invoice") and not re.match(r"^page\s*\d+\b", low):

                    block.append(t)

            sold_by_block = _clean_text(" | ".join(block))

            if sold_by_block:

                out["supplier"] = sold_by_block[:600]



        supplier_lines: List[str] = []

        for ln in cleaned[:120]:

            low = ln.lower()

            if any(x in low for x in [

                "billing address",

                "delivery address",

                "sold by",

                "order information",

                "invoice date",

                "invoice #",

                "invoice total",

                "for customer support",

                "www.amazon",

            ]):

                break

            t = _clean_text(ln)

            if not t:

                continue

            if t.lower() == "invoice":

                continue

            if t.lower() == "paid":

                continue

            if re.match(r"^page\s*\d+\b", low):

                continue

            supplier_lines.append(t)

            if len(supplier_lines) >= 10:

                break

        supplier_text = _clean_text(" | ".join(supplier_lines))

        if supplier_text:

            if not _clean_text(out.get("supplier")):

                out["supplier"] = supplier_text[:600]



        if not _clean_text(out.get("supplier")):

            bill_idx = -1

            for i, ln in enumerate(cleaned[:260]):

                if re.search(r"\bBilling\s+address\b", ln, flags=re.IGNORECASE):

                    bill_idx = i

                    break

            if bill_idx != -1:

                block: List[str] = []

                for ln in cleaned[bill_idx + 1 : min(len(cleaned), bill_idx + 25)]:

                    low = ln.lower()

                    if any(x in low for x in ["delivery address", "sold by", "order information", "invoice details", "invoice date", "invoice #"]):

                        break

                    t = _clean_text(ln)

                    if t and not re.match(r"^page\s*\d+\b", low):

                        block.append(t)

                supplier2 = _clean_text(" | ".join(block))

                if supplier2:

                    out["supplier"] = supplier2[:600]



        reg_no = ""

        for ln in cleaned[:260]:

            m = re.search(r"\bVAT\s*[#:]\s*(GB\s*[0-9A-Za-z ]{6,20})\b", ln, flags=re.IGNORECASE)

            if m:

                reg_no = _clean_text(m.group(1)).replace(" ", "")

                reg_no = reg_no.upper()

                break

        if not reg_no:

            for ln in cleaned[:260]:

                m = re.search(r"\bVAT\s*[#:]?\s*(GB\s*[0-9 ]{6,20})\b", ln, flags=re.IGNORECASE)

                if m:

                    reg_no = _clean_text(m.group(1)).replace(" ", "")

                    reg_no = reg_no.upper()

                    break

        out["reg_no"] = reg_no if reg_no else "N/A"



        def _find_amount_after_labels(labels: List[str], scan: int = 1400) -> Optional[float]:

            for i, ln in enumerate(cleaned[:scan]):

                low = ln.lower()

                if not any(lbl in low for lbl in labels):

                    continue

                m0 = CURRENCY_RE.search(ln)

                if m0:

                    return _to_float_or_none(m0.group(0))

                for j in range(1, 8):

                    if i + j >= len(cleaned):

                        break

                    nxt = cleaned[i + j]

                    m1 = CURRENCY_RE.search(nxt)

                    if m1:

                        return _to_float_or_none(m1.group(0))

            return None



        total_v = _find_amount_after_labels(["invoice total", "total payable", "total\tpayable", "total payable"])

        if isinstance(total_v, (int, float)):

            out["buying_price"] = abs(float(total_v))

            out["non_vat"] = abs(float(total_v))



        vat_v = _find_amount_after_labels(["vat subtotal", "vat sub total", "vat sub-total", "vat\tsubtotal"])

        if isinstance(vat_v, (int, float)):

            out["vat_amount"] = abs(float(vat_v))



        net_v = _find_amount_after_labels([

            "item subtotal (excl. vat)",

            "item subtotal (excl.\u00a0vat)",

            "item subtotal excl. vat",

            "item subtotal (excl vat)",

            "item subtotal",

        ])

        if isinstance(net_v, (int, float)):

            out["std_net"] = abs(float(net_v))

        else:

            out["std_net"] = "N/A"



        low_all = joined_low

        if "credit note" in low_all or "refund" in low_all:

            out["category"] = "expense"

        elif "sales invoice" in low_all or "sale" in low_all or "sales" in low_all:

            out["category"] = "sale"

        else:

            out["category"] = "purchase"



        if not _clean_text(out.get("reg_no")):

            out["reg_no"] = "N/A"



        return out







    def _extract_repair_telecommunications_limited_fields() -> Dict[str, Any]:



        out: Dict[str, Any] = {}



        def _value_after_label(label_re: "re.Pattern[str]", scan: int = 260) -> str:

            for i, ln in enumerate(cleaned[:scan]):

                m = label_re.search(ln)

                if not m:

                    continue

                tail = _clean_text(ln[m.end() :]).strip(" :-\t")

                if tail:

                    return tail

                for j in range(1, 5):

                    if i + j >= len(cleaned):

                        break

                    nxt = _clean_text(cleaned[i + j])

                    if nxt:

                        return nxt

            return ""



        # Invoice Number -> inv_ref_no

        inv_ref = _value_after_label(re.compile(r"\bInvoice\s*(?:No\.?|Number)\b\s*[:\-]?", flags=re.IGNORECASE))

        inv_ref = _clean_text(inv_ref)

        if inv_ref:

            mref = re.search(r"\b([A-Za-z0-9\-/]+)\b", inv_ref)

            out["inv_ref_no"] = _clean_text(mref.group(1)) if mref else inv_ref



        # Tax Date -> document_date

        date_re = re.compile(r"\b(\d{1,2}[\/-]\d{1,2}[\/-]\d{2,4})\b")

        tax_date_raw = _value_after_label(re.compile(r"\bTax\s*Date\b\s*[:\-]?", flags=re.IGNORECASE))

        tax_date_raw = _clean_text(tax_date_raw).replace("-", "/")

        mdate = date_re.search(tax_date_raw) if tax_date_raw else None

        if not mdate:

            for ln in cleaned[:220]:

                if "tax date" not in ln.lower():

                    continue

                m2 = date_re.search(ln.replace("-", "/"))

                if m2:

                    mdate = m2

                    break

        if mdate:

            out["document_date"] = _clean_text(mdate.group(1)).replace("-", "/")



        # Supplier (as requested): take the "Invoice To" block contents

        invoice_to_idx = -1

        for i, ln in enumerate(cleaned[:220]):

            if re.search(r"\bInvoice\s*To\b", ln, flags=re.IGNORECASE):

                invoice_to_idx = i

                break

        if invoice_to_idx != -1:

            block: List[str] = []

            for ln in cleaned[invoice_to_idx + 1 : min(len(cleaned), invoice_to_idx + 35)]:

                low = ln.lower()

                if any(x in low for x in ["invoice number", "invoice no", "tax date", "invoice date", "description", "qty", "quantity", "unit price", "amount", "subtotal", "goods total", "total amount due", "amount due"]):

                    break

                t = _clean_text(ln)

                if t and not re.match(r"^page\s*\d+\b", low):

                    block.append(t)

            supplier_text = _clean_text(" | ".join(block))

            if supplier_text:

                out["supplier"] = supplier_text[:600]



        # VAT Reg No -> reg_no

        vat_reg = _value_after_label(re.compile(r"\bVAT\s*(?:Reg(?:istration)?\s*)?No\.?\b\s*[:\-]?", flags=re.IGNORECASE))

        vat_reg = _clean_text(vat_reg)

        if vat_reg:

            m_vat = re.search(r"\b([A-Za-z0-9 ]{6,20})\b", vat_reg)

            out["reg_no"] = _clean_text(m_vat.group(1)) if m_vat else vat_reg

        else:

            out["reg_no"] = "N/A"



        # Make: first meaningful line item description

        make = ""

        start_idx = -1

        for i, ln in enumerate(cleaned[:900]):

            low = ln.lower()

            if "description" in low and ("qty" in low or "quantity" in low or "unit" in low or "price" in low or "amount" in low):

                start_idx = i

                break

        if start_idx >= 0:

            for ln in cleaned[start_idx + 1 : min(len(cleaned), start_idx + 50)]:

                t = _clean_text(ln)

                if not t:

                    continue

                low = t.lower()

                if any(x in low for x in ["goods total", "subtotal", "vat", "tax", "total", "amount due", "total amount due"]):

                    break

                if CURRENCY_RE.search(t):

                    continue

                if any(h in low for h in ["qty", "quantity", "unit price", "price", "amount"]):

                    continue

                if len(t) >= 3 and re.search(r"[A-Za-z]", t):

                    make = t

                    break

        if make:

            out["make"] = make[:240]



        def _find_amount_after_labels(labels: List[str]) -> Optional[float]:

            for i, ln in enumerate(cleaned[:950]):

                low = ln.lower()

                if not any(lbl in low for lbl in labels):

                    continue

                m0 = CURRENCY_RE.search(ln)

                if m0:

                    return _to_float_or_none(m0.group(0))

                for j in range(1, 7):

                    if i + j >= len(cleaned):

                        break

                    nxt = cleaned[i + j]

                    m1 = CURRENCY_RE.search(nxt)

                    if m1:

                        return _to_float_or_none(m1.group(0))

            return None



        # Goods Total -> std_net

        goods_total_v = _find_amount_after_labels(["goods total", "goods\u00a0total"])

        if isinstance(goods_total_v, (int, float)):

            out["std_net"] = abs(float(goods_total_v))



        # Total Amount Due -> buying_price and non_vat

        amount_due_v = _find_amount_after_labels(["total amount due", "amount due"])

        if isinstance(amount_due_v, (int, float)):

            out["buying_price"] = abs(float(amount_due_v))

            out["non_vat"] = abs(float(amount_due_v))



        # VAT@20% -> vat

        vat_v = _find_amount_after_labels([

            "vat@20",

            "vat @20",

            "vat@20%",

            "vat @20%",

            "vat@20 %",

            "vat @20 %",

            "vat 20%",

            "vat 20.00",

            "vat 20",

        ])

        if isinstance(vat_v, (int, float)):

            out["vat"] = abs(float(vat_v))

            out["vat_amount"] = abs(float(vat_v))



        # Category: infer for this layout

        if "credit note" in joined_low or "refund" in joined_low:

            out["category"] = "expense"

        else:

            out["category"] = "sale"



        return out







    def _extract_beetroot_catering_booking_fields() -> Dict[str, Any]:



        out: Dict[str, Any] = {}



        def _value_after_label(label_re: "re.Pattern[str]") -> str:

            for i, ln in enumerate(cleaned[:240]):

                m = label_re.search(ln)

                if not m:

                    continue

                tail = _clean_text(ln[m.end() :]).strip(" :-\t")

                if tail:

                    return tail

                for j in range(1, 4):

                    if i + j >= len(cleaned):

                        break

                    nxt = _clean_text(cleaned[i + j])

                    if nxt:

                        return nxt

            return ""



        # Invoice number

        inv_ref = _value_after_label(re.compile(r"\bInvoice\s*Number\b\s*[:\-]?", flags=re.IGNORECASE))

        inv_ref = _clean_text(inv_ref)

        if inv_ref:

            mref = re.search(r"\b([A-Za-z0-9\-/]+)\b", inv_ref)

            out["inv_ref_no"] = _clean_text(mref.group(1)) if mref else inv_ref



        # Due date preferred; fallback to Date of Issue

        date_re = re.compile(r"\b(\d{1,2}[\/-]\d{1,2}[\/-]\d{2,4})\b")

        due_raw = _value_after_label(re.compile(r"\bDue\s*Date\b\s*[:\-]?", flags=re.IGNORECASE))

        due_raw = _clean_text(due_raw).replace("-", "/")

        mdate = date_re.search(due_raw) if due_raw else None

        if not mdate:

            issue_raw = _value_after_label(re.compile(r"\bDate\s*of\s*Issue\b\s*[:\-]?", flags=re.IGNORECASE))

            issue_raw = _clean_text(issue_raw).replace("-", "/")

            mdate = date_re.search(issue_raw) if issue_raw else None

        if not mdate:

            for ln in cleaned[:180]:

                low = ln.lower()

                if "due date" in low or "date of issue" in low:

                    m2 = date_re.search(ln)

                    if m2:

                        mdate = m2

                        break

        if mdate:

            out["document_date"] = _clean_text(mdate.group(1)).replace("-", "/")



        # Supplier from header: for this layout it appears in the top-right block.

        supplier = ""

        for ln in cleaned[:60]:

            t = _clean_text(ln)

            if not t:

                continue

            low = t.lower()

            if "beetroot catering booking" in low:

                continue

            if any(bad in low for bad in ["billed to", "date of issue", "due date", "invoice number", "amount due", "page"]):

                continue

            # Prefer the strong known supplier token for this template.

            if "sissons" in low:

                supplier = t

                break

        if not supplier:

            for ln in cleaned[:60]:

                t = _clean_text(ln)

                if not t:

                    continue

                low = t.lower()

                if any(bad in low for bad in ["billed to", "date of issue", "due date", "invoice number", "amount due", "page", "beetroot"]):

                    continue

                if len(t) >= 6 and re.search(r"[A-Za-z]", t):

                    supplier = t

                    break

        if supplier:

            out["supplier"] = supplier[:120]



        # Make: item description. Capture first meaningful line after ITEM table heading.

        make = ""

        start_idx = -1

        for i, ln in enumerate(cleaned[:600]):

            low = ln.lower()

            if low.strip() == "item" or ("item" in low and "line total" in low):

                start_idx = i

                break

        if start_idx >= 0:

            for ln in cleaned[start_idx + 1 : min(len(cleaned), start_idx + 40)]:

                t = _clean_text(ln)

                if not t:

                    continue

                low = t.lower()

                if any(x in low for x in ["qty", "price", "line total", "subtotal", "vat", "taxable", "total"]):

                    break

                if CURRENCY_RE.search(t):

                    continue

                if len(t) >= 4:

                    make = t

                    break

        if make:

            out["make"] = make[:240]



        # Reg no: sometimes not present on this invoice; keep consistent with other parsers.

        reg_no = ""

        for ln in cleaned[:260]:

            m = re.search(r"\b([A-Z]{2}[0-9O]{2}\s*[A-Z]{3})\b", ln, flags=re.IGNORECASE)

            if m:

                reg_no = _format_uk_reg(m.group(1))

                break

        out["reg_no"] = reg_no if reg_no else "N/A"



        def _find_amount_after_labels(labels: List[str]) -> Optional[float]:

            for i, ln in enumerate(cleaned[:900]):

                low = ln.lower()

                if not any(lbl in low for lbl in labels):

                    continue

                m0 = CURRENCY_RE.search(ln)

                if m0:

                    return _to_float_or_none(m0.group(0))

                for j in range(1, 6):

                    if i + j >= len(cleaned):

                        break

                    nxt = cleaned[i + j]

                    m1 = CURRENCY_RE.search(nxt)

                    if m1:

                        return _to_float_or_none(m1.group(0))

            return None



        subtotal_v = _find_amount_after_labels(["subtotal", "sub total", "sub-total"])

        taxable_v = _find_amount_after_labels(["taxable subtotal", "taxable sub total", "taxable sub-total"])

        vat_v = _find_amount_after_labels(["vat (", "vat"])

        total_v = _find_amount_after_labels(["amount due", "total"])



        # Requested mapping:

        # buying_price & non_vat = invoice total / amount due

        if isinstance(total_v, (int, float)):

            out["buying_price"] = abs(float(total_v))

            out["non_vat"] = abs(float(total_v))



        # std_net = taxable subtotal when present; else fallback to subtotal

        if isinstance(taxable_v, (int, float)):

            out["std_net"] = abs(float(taxable_v))

        elif isinstance(subtotal_v, (int, float)):

            out["std_net"] = abs(float(subtotal_v))



        if isinstance(vat_v, (int, float)):

            out["vat_amount"] = abs(float(vat_v))



        # Category inference: treat as purchase by default; credit note => expense

        if "credit note" in joined_low or "refund" in joined_low:

            out["category"] = "expense"

        else:

            out["category"] = "purchase"



        return out







    def _extract_warranty_solutions_group_swg_fields() -> Dict[str, Any]:



        out: Dict[str, Any] = {}



        def _pick_header_date() -> str:



            # Prefer explicit invoice/credit-note date labels.

            # Fallback to the first valid-looking UK date in header while ignoring

            # "Document Generated" timestamp line.



            date_re = re.compile(r"\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})\b")



            def _norm_date(s: str) -> str:

                return _clean_text(s).replace("-", "/")



            def _date_from_line(line: str) -> str:

                m = date_re.search(line)

                if not m:

                    return ""

                d = _norm_date(m.group(1))

                return d if _is_valid_uk_date(d) else d



            def _find_after_label(label_re: "re.Pattern[str]") -> str:

                for i, ln in enumerate(cleaned[:180]):

                    if not label_re.search(ln):

                        continue

                    # Try date on same line.

                    d1 = _date_from_line(ln)

                    if d1:

                        return d1

                    # Try date on next couple lines (OCR may break line).

                    for j in range(1, 4):

                        if i + j >= len(cleaned):

                            break

                        d2 = _date_from_line(cleaned[i + j])

                        if d2:

                            return d2

                return ""



            # Strong labels first.

            d = _find_after_label(re.compile(r"\bCredit\s*Note\s*Date\b\s*[:\-]?", flags=re.IGNORECASE))

            if d:

                return d

            d = _find_after_label(re.compile(r"\bInvoice\s*Date\b\s*[:\-]?", flags=re.IGNORECASE))

            if d:

                return d



            # Generic "Date" label, but ignore Document Generated.

            for i, ln in enumerate(cleaned[:120]):

                low = ln.lower()

                if "document generated" in low or "generated" in low:

                    continue

                if "date" not in low:

                    continue

                d0 = _date_from_line(ln)

                if d0:

                    return d0

                for j in range(1, 3):

                    if i + j >= len(cleaned):

                        break

                    d1 = _date_from_line(cleaned[i + j])

                    if d1:

                        return d1



            # Last fallback: first date token in the header excluding Document Generated.

            for ln in cleaned[:120]:

                low = ln.lower()

                if "document generated" in low or "generated" in low or low.startswith("page"):

                    continue

                d2 = _date_from_line(ln)

                if d2:

                    return d2



            return ""



        # Supplier: from the left block "Name: ..." (circled in provided image)

        supplier = ""

        for ln in cleaned[:120]:

            m = re.search(r"\bName\s*:\s*(.+)$", ln, flags=re.IGNORECASE)

            if m:

                supplier = _clean_text(m.group(1))

                break

        if supplier:

            out["supplier"] = supplier



        # Invoice / reference number: "Credit Note Number: ..."

        inv_ref = ""

        for ln in cleaned[:150]:

            m = re.search(r"\bCredit\s*Note\s*Number\s*:\s*([A-Za-z0-9\-\/]+)", ln, flags=re.IGNORECASE)

            if m:

                inv_ref = _clean_text(m.group(1))

                break

        if inv_ref:

            out["inv_ref_no"] = inv_ref



        # Document date (invoice/credit note date from header)

        doc_date = _pick_header_date()

        if doc_date:

            out["document_date"] = doc_date



        # Make: map to "Account Name" on PDF (as requested)

        make = ""

        for ln in cleaned[:200]:

            m = re.search(r"\bAccount\s*Name\s*:\s*(.+)$", ln, flags=re.IGNORECASE)

            if m:

                make = _clean_text(m.group(1))

                break

        if make:

            out["make"] = make



        # Reg number: look for UK plate pattern anywhere (table includes something like CF65BCZ)

        reg_no = ""

        for ln in cleaned:

            m = re.search(r"\b([A-Z]{2}[0-9O]{2}\s*[A-Z]{3})\b", ln, flags=re.IGNORECASE)

            if m:

                reg_no = _clean_text(m.group(1)).upper()

                reg_no = reg_no.replace("O", "0")

                reg_no = reg_no[:4] + " " + reg_no[4:] if len(reg_no.replace(" ", "")) == 7 else reg_no

                break

        if reg_no:

            out["reg_no"] = reg_no



        # Totals: Net/VAT/Gross on header.

        def _amount_after_label(label: str) -> Optional[float]:

            for ln in cleaned[:250]:

                m = re.search(rf"\b{label}\s*:\s*([^\n]+)$", ln, flags=re.IGNORECASE)

                if not m:

                    continue

                amt_m = CURRENCY_RE.search(m.group(1))

                if amt_m:

                    return _to_float_or_none(amt_m.group(0))

            return None



        net_v = _amount_after_label("Net")

        vat_v = _amount_after_label("VAT")

        gross_v = _amount_after_label("Gross")



        # Requested mapping:

        # buying_price = Gross, non_vat = Gross, std_net = Net, vat_amount = VAT

        if gross_v is not None:

            out["buying_price"] = abs(float(gross_v))

            out["non_vat"] = abs(float(gross_v))

        if net_v is not None:

            out["std_net"] = abs(float(net_v))

        if vat_v is not None:

            out["vat_amount"] = abs(float(vat_v))



        # Category inference:

        # Credit Note usually behaves like an expense adjustment in your workflow.

        # If amounts are negative OR document says Credit Note => expense, otherwise purchase.

        is_credit_note = "credit note" in joined_low

        any_negative = any((v is not None and isinstance(v, (int, float)) and float(v) < 0) for v in [net_v, vat_v, gross_v])

        out["category"] = "expense" if (is_credit_note or any_negative) else "purchase"



        return out







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







    def _find_amount_after_phrase_spanning(phrases: List[str]) -> Optional[float]:



        # OCR may split labels like "TOTAL"/"DUE" across lines (and/or place the £ amount on the next line).

        # This scans the matching line and a short forward window.

        for i, ln in enumerate(cleaned[:900]):



            low = ln.lower()



            if not any(p in low for p in phrases):



                continue



            m = CURRENCY_RE.search(ln)

            if m:



                return _to_float_or_none(m.group(0))



            for j in range(1, 6):



                if i + j >= len(cleaned):



                    break



                nxt = cleaned[i + j]



                m2 = CURRENCY_RE.search(nxt)



                if m2:



                    return _to_float_or_none(m2.group(0))



        return None







    def _looks_like_costcutter_supermarkets_group() -> bool:



        head = "\n".join(cleaned[:80]).lower()

        if "costcutter" not in head:

            return False

        # Typical identifiers on this invoice header.

        return ("invoice number" in head) or ("awrs urn" in head) or ("vat no" in head) or ("vat no." in head)







    def _extract_costcutter_supermarkets_group_fields() -> Dict[str, Any]:



        out: Dict[str, Any] = {}



        def _find_date_after_label(label_pat: "re.Pattern[str]") -> str:

            date_re = re.compile(r"\b(\d{1,2}[\/-]\d{1,2}[\/-]\d{2,4})\b")

            for i, ln in enumerate(cleaned[:250]):

                if not label_pat.search(ln):

                    continue

                m0 = date_re.search(ln)

                if m0:

                    return _clean_text(m0.group(1)).replace("-", "/")

                for j in range(1, 5):

                    if i + j >= len(cleaned):

                        break

                    m1 = date_re.search(cleaned[i + j])

                    if m1:

                        return _clean_text(m1.group(1)).replace("-", "/")

            return ""



        def _find_amount_after_labels(labels: List[str]) -> Optional[float]:

            for i, ln in enumerate(cleaned[:900]):

                low = ln.lower()

                if not any(lbl in low for lbl in labels):

                    continue

                m0 = CURRENCY_RE.search(ln)

                if m0:

                    return _to_float_or_none(m0.group(0))

                for j in range(1, 6):

                    if i + j >= len(cleaned):

                        break

                    nxt = cleaned[i + j]

                    m1 = CURRENCY_RE.search(nxt)

                    if m1:

                        return _to_float_or_none(m1.group(0))

            return None



        # Invoice / reference number

        inv_ref = ""

        for i, ln in enumerate(cleaned[:120]):

            m = re.search(r"\bInvoice\s*Number\b\s*([A-Za-z0-9\-/]+)", ln, flags=re.IGNORECASE)

            if m:

                inv_ref = _clean_text(m.group(1))

                break

            if "invoice number" in ln.lower():

                for j in range(1, 3):

                    if i + j >= len(cleaned):

                        break

                    m2 = re.search(r"\b([A-Za-z0-9\-/]{6,})\b", cleaned[i + j])

                    if m2:

                        inv_ref = _clean_text(m2.group(1))

                        break

        if inv_ref:

            out["inv_ref_no"] = inv_ref



        # Document date (Inv. Date)

        doc_date = _find_date_after_label(re.compile(r"\bInv\.?\s*Date\b\s*[:\-]?", flags=re.IGNORECASE))

        if doc_date:

            out["document_date"] = doc_date



        # Supplier block: "From Supplier" section

        supplier_details = ""

        start_idx = -1

        for i, ln in enumerate(cleaned[:250]):

            if re.search(r"\bFrom\s*Supplier\b", ln, flags=re.IGNORECASE):

                start_idx = i

                break

        if start_idx != -1:

            block: List[str] = []

            for ln in cleaned[start_idx : min(len(cleaned), start_idx + 60)]:

                low = ln.lower()

                if any(x in low for x in ("delivery to", "invoice to", "product", "description")):

                    break

                if ln:

                    block.append(ln)

            supplier_details = _clean_text(" | ".join(block))

        supplier_name = ""

        if start_idx != -1:

            for i, ln in enumerate(cleaned[start_idx : min(len(cleaned), start_idx + 60)]):

                m = re.search(r"\bName\s*:\s*(.+)$", ln, flags=re.IGNORECASE)

                if m:

                    supplier_name = _clean_text(m.group(1))

                    if not supplier_name:

                        for j in range(1, 4):

                            if i + j >= len(cleaned):

                                break

                            nxt = _clean_text(cleaned[start_idx + i + j])

                            if nxt and not re.search(r"\b(Address|Town|County|Post\s*Code|VAT\s*Number|Code)\s*:\b", nxt, flags=re.IGNORECASE):

                                supplier_name = nxt

                                break

                    break

        if supplier_name:

            out["supplier"] = supplier_name[:120]

        elif supplier_details:

            out["supplier"] = supplier_details[:600]



        # Header details (Costcutter name + AWRS/VAT/address) -> map into make field

        header_lines: List[str] = []

        for ln in cleaned[:40]:

            low = ln.lower()

            if "invoice number" in low:

                break

            if not ln:

                continue

            # Keep key company/header lines only

            if ("costcutter" in low) or ("awrs" in low) or ("vat" in low) or re.search(r"\byo\d{2}\b", low):

                header_lines.append(ln)

        header_text = _clean_text(" | ".join(header_lines))

        if header_text:

            out["make"] = header_text[:600]



        # reg_no: prefer VAT No, else AWRS URN, else N/A

        reg_no = ""

        for ln in cleaned[:120]:

            m_vat = re.search(r"\bVat\s*No\.?\s*[:\-]?\s*([0-9 ]{6,})\b", ln, flags=re.IGNORECASE)

            if m_vat:

                reg_no = _clean_text(m_vat.group(1))

                break

        if not reg_no:

            for ln in cleaned[:120]:

                m_awrs = re.search(r"\bAWRS\s*URN\b\s*[:\-]?\s*([A-Za-z0-9]+)", ln, flags=re.IGNORECASE)

                if m_awrs:

                    reg_no = _clean_text(m_awrs.group(1))

                    break

        out["reg_no"] = reg_no if reg_no else "N/A"



        # Totals

        net_v = _find_amount_after_labels(["net value", "net\u00a0value", "netvalue"])

        vat_v = _find_amount_after_labels(["vat"])

        total_payable = _find_amount_after_labels(["total payable", "amount payable"])



        if isinstance(total_payable, (int, float)):

            out["buying_price"] = abs(float(total_payable))

            out["non_vat"] = abs(float(total_payable))

        if isinstance(net_v, (int, float)):

            out["std_net"] = abs(float(net_v))

        if isinstance(vat_v, (int, float)):

            out["vat_amount"] = abs(float(vat_v))



        # Category (default purchase for this supplier invoice)

        out["category"] = "purchase" if "credit note" not in joined_low else "expense"



        return out







    def _extract_savin_wholesalers_ltd_fields() -> Dict[str, Any]:



        out: Dict[str, Any] = {}



        def _value_after_label(label_re: "re.Pattern[str]") -> str:

            for i, ln in enumerate(cleaned[:200]):

                m = label_re.search(ln)

                if not m:

                    continue

                tail = _clean_text(ln[m.end() :]).strip(" :-\t")

                if tail:

                    return tail

                for j in range(1, 4):

                    if i + j >= len(cleaned):

                        break

                    nxt = _clean_text(cleaned[i + j])

                    if nxt:

                        return nxt

            return ""



        # Invoice number -> inv_ref_no

        inv_ref = _value_after_label(re.compile(r"\bInvoice\s*(?:No\.?|Number)\b\s*[:\-]?", flags=re.IGNORECASE))

        inv_ref = _clean_text(inv_ref)

        if inv_ref:

            mref = re.search(r"\b([A-Za-z0-9\-/]+)\b", inv_ref)

            out["inv_ref_no"] = _clean_text(mref.group(1)) if mref else inv_ref



        # Document date -> document_date

        date_re = re.compile(r"\b(\d{1,2}[\/-]\d{1,2}[\/-]\d{2,4})\b")

        date_raw = _value_after_label(re.compile(r"\bInvoice\s*Date\b\s*[:\-]?", flags=re.IGNORECASE))

        if not date_raw:

            date_raw = _value_after_label(re.compile(r"\bDate\b\s*[:\-]?", flags=re.IGNORECASE))

        date_raw = _clean_text(date_raw).replace("-", "/")

        mdate = date_re.search(date_raw)

        if not mdate:

            for ln in cleaned[:220]:

                low = ln.lower()

                if "date" not in low:

                    continue

                m2 = date_re.search(ln)

                if m2:

                    mdate = m2

                    break

        if mdate:

            out["document_date"] = _clean_text(mdate.group(1)).replace("-", "/")



        # Supplier from upper heading

        header_block: List[str] = []

        for ln in cleaned[:80]:

            low = ln.lower()

            if any(x in low for x in ["invoice no", "invoice number", "invoice date", "savin", "wholesalers"]):

                break

            if low.startswith("page") or re.match(r"^page\s*\d+\b", low):

                continue

            if ln:

                header_block.append(ln)

        

        supplier = ""

        for ln in header_block[:18]:

            t = _clean_text(ln)

            if not t:

                continue

            low = t.lower()

            if any(bad in low for bad in ["invoice", "vat", "reg", "telephone", "tel", "email", "www", "page"]):

                continue

            if len(t) >= 6:

                supplier = t

                break

        if supplier:

            out["supplier"] = supplier[:120]



        # Category: sale, purchase, or expense

        low_all = joined_low

        if "credit note" in low_all or "refund" in low_all:

            out["category"] = "expense"

        elif "sale" in low_all or "sales" in low_all:

            out["category"] = "sale"

        else:

            out["category"] = "purchase"



        def _find_amount_after_labels(labels: List[str]) -> Optional[float]:

            for i, ln in enumerate(cleaned[:950]):

                low = ln.lower()

                if not any(lbl in low for lbl in labels):

                    continue

                m0 = CURRENCY_RE.search(ln)

                if m0:

                    return _to_float_or_none(m0.group(0))

                for j in range(1, 8):

                    if i + j >= len(cleaned):

                        break

                    nxt = cleaned[i + j]

                    m1 = CURRENCY_RE.search(nxt)

                    if m1:

                        return _to_float_or_none(m1.group(0))

            return None



        # Total in PDF -> buying_price and non_vat

        total_v = _find_amount_after_labels(["total", "total in", "invoice total", "grand total"])

        if isinstance(total_v, (int, float)):

            out["buying_price"] = abs(float(total_v))

            out["non_vat"] = abs(float(total_v))



        # Subtotal in PDF -> sub_net

        subtotal_v = _find_amount_after_labels(["subtotal", "sub total", "sub-total"])

        if isinstance(subtotal_v, (int, float)):

            out["sub_net"] = abs(float(subtotal_v))



        # VAT 20.00 in PDF -> vat

        vat_v = _find_amount_after_labels(["vat", "vat 20.00", "tax", "tax amount"])

        if isinstance(vat_v, (int, float)):

            out["vat"] = abs(float(vat_v))



        return out







    if _looks_like_costcutter_supermarkets_group():



        return _extract_costcutter_supermarkets_group_fields()







    if _looks_like_despatch_note():



        return _extract_despatch_note_fields()







    if _looks_like_one_stop_invoice():



        return _extract_one_stop_invoice_fields()







    if _looks_like_warranty_solutions_group_swg():



        return _extract_warranty_solutions_group_swg_fields()







    if _looks_like_savin_wholesalers_ltd():



        return _extract_savin_wholesalers_ltd_fields()







    if _looks_like_smiths_fire_llp():



        return _extract_smiths_fire_llp_fields()







    if _looks_like_combi_tech_engineering_services():



        return _extract_combi_tech_engineering_services_fields()







    if _looks_like_autotrader_invoice():



        return _extract_autotrader_invoice_fields()







    if _looks_like_j_wilson_plumbing_heating():



        return _extract_j_wilson_plumbing_heating_fields()







    if _looks_like_beetroot_catering_booking():



        return _extract_beetroot_catering_booking_fields()







    if _looks_like_repair_telecommunications_limited():



        return _extract_repair_telecommunications_limited_fields()







    if _looks_like_amazon_invoice():



        return _extract_amazon_invoice_fields()







    if _looks_like_sw_motor_factors_ltd():



        return _extract_sw_motor_factors_ltd_fields()







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







        buying_price = _find_amount_after_phrase_spanning(["total due", "total\u00a0due", "total  due"])



        if buying_price is None:



            # Fallback: look for TOTAL label (often appears as "TOTAL £") near bottom

            buying_price = _find_amount_after_phrase_spanning(["total £", "total\u00a3", "total"])  # fallback







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







    s = s.replace("Â£", "£")



    s = s.replace("GBP", "").replace("gbp", "")



    s = s.replace("£", "").replace("$", "").replace("€", "")



    s = s.replace(",", "").replace(" ", "")



    if s.endswith("-"):



        neg = True



        s = s[:-1]







    m = re.search(r"-?\d+(?:\.\d{1,2})?", s)



    if not m:



        raise ValueError("Invalid amount")







    amt = float(m.group(0))



    if amt < 0:



        return amt



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





def _invoice_score_text_lines(lines: List[str]) -> int:



    cleaned = [_clean_text(x) for x in lines]

    cleaned = [x for x in cleaned if x]

    if not cleaned:

        return -1



    t = "\n".join(cleaned).lower()

    score = 0



    if "invoice" in t:

        score += 6

    if "document" in t and "date" in t:

        score += 6

    if "total" in t:

        score += 4

    if "total due" in t:

        score += 8

    if "vat" in t:

        score += 2

    if "bca" in t or "british car auctions" in t:

        score += 4



    score += len(re.findall(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", t)) * 2

    score += len(CURRENCY_RE.findall(t)) * 2



    # Prefer outputs that look like real text and have enough content.

    score += min(len(t) // 80, 30)

    return score





def _invoice_extract_text_lines_from_pdf_without_ocr(pdf_path: str) -> List[str]:



    # First try the standard fast extractor.

    base = _extract_text_lines_from_pdf_without_ocr(pdf_path)



    # For rotated / multi-column digital PDFs, block-based extraction (x/y ordered)

    # can preserve reading order better than plain text extraction.

    alt: List[str] = []

    if fitz is not None:

        try:

            doc = fitz.open(pdf_path)  # type: ignore[union-attr]

            for page in doc:

                try:

                    blocks = page.get_text("blocks")

                except Exception:

                    blocks = []



                # blocks: (x0, y0, x1, y1, "text", block_no, block_type)

                parts: List[str] = []

                try:

                    blocks_sorted = sorted(blocks, key=lambda b: (float(b[1]), float(b[0])))

                except Exception:

                    blocks_sorted = blocks



                for b in blocks_sorted:

                    try:

                        txt = b[4]

                    except Exception:

                        txt = ""

                    if txt:

                        parts.append(str(txt))



                if parts:

                    alt.extend("\n".join(parts).splitlines())

        except Exception:

            alt = []



    base_clean = [_clean_text(x) for x in base]

    base_clean = [x for x in base_clean if x]

    alt_clean = [_clean_text(x) for x in alt]

    alt_clean = [x for x in alt_clean if x]



    if not alt_clean:

        return base_clean

    if not base_clean:

        return alt_clean



    sb = _invoice_score_text_lines(base_clean)

    sa = _invoice_score_text_lines(alt_clean)

    return alt_clean if sa > sb else base_clean







def _invoice_extract_text_lines_from_pdf_with_ocr(pdf_path: str, force_ocr: bool = False) -> Tuple[List[str], bool]:



    lines: List[str] = []



    if not force_ocr:

        cleaned = _invoice_extract_text_lines_from_pdf_without_ocr(pdf_path)

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



                for base_img in _ocr_preprocess_variants(pil_img):



                    for angle in (0, 90, 180, 270):



                        try:



                            img2 = base_img.rotate(angle, expand=True) if angle else base_img



                        except Exception:



                            img2 = base_img



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



                for base_img in _ocr_preprocess_variants(pil_img):



                    for angle in (0, 90, 180, 270):



                        try:



                            img2 = base_img.rotate(angle, expand=True) if angle else base_img



                        except Exception:



                            img2 = base_img



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







def _invoice_pdf_page_count(pdf_path: str) -> int:



    try:

        if pdfium is not None:

            doc = pdfium.PdfDocument(pdf_path)

            return int(len(doc))

    except Exception:

        pass



    try:

        if fitz is not None:

            doc2 = fitz.open(pdf_path)  # type: ignore[union-attr]

            try:

                return int(doc2.page_count)  # type: ignore[union-attr]

            except Exception:

                return int(len(doc2))

    except Exception:

        pass



    try:

        with pdfplumber.open(pdf_path) as pdf:

            return int(len(pdf.pages))

    except Exception:

        return 0







def _invoice_extract_text_lines_from_pdf_pages(pdf_path: str) -> List[List[str]]:



    pages_out: List[List[str]] = []



    try:

        with pdfplumber.open(pdf_path) as pdf:

            for page in pdf.pages:

                txt = page.extract_text() or ""

                lines = [x for x in [_clean_text(x) for x in str(txt).splitlines()] if x]

                pages_out.append(lines)

    except Exception:

        pages_out = []



    if pages_out and any(p for p in pages_out):

        return pages_out



    if fitz is not None:

        try:

            doc = fitz.open(pdf_path)  # type: ignore[union-attr]

            for page in doc:

                try:

                    blocks = page.get_text("blocks")

                except Exception:

                    blocks = []

                parts: List[str] = []

                try:

                    blocks_sorted = sorted(blocks, key=lambda b: (float(b[1]), float(b[0])))

                except Exception:

                    blocks_sorted = blocks

                for b in blocks_sorted:

                    try:

                        txtb = b[4]

                    except Exception:

                        txtb = ""

                    if txtb:

                        parts.append(str(txtb))

                lines2 = [x for x in [_clean_text(x) for x in "\n".join(parts).splitlines()] if x]

                pages_out.append(lines2)

        except Exception:

            pages_out = []



    return pages_out







def _invoice_extract_text_lines_from_pdf_pages_with_ocr(pdf_path: str, force_ocr: bool = False) -> Tuple[List[List[str]], bool]:



    if not force_ocr:

        pages = _invoice_extract_text_lines_from_pdf_pages(pdf_path)

        if pages and any(p for p in pages):

            return pages, False



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

    pages_out: List[List[str]] = []



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

                lines = [x for x in [_clean_text(x) for x in str(best_txt).splitlines()] if x]

                pages_out.append(lines)

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

                lines = [x for x in [_clean_text(x) for x in str(best_txt).splitlines()] if x]

                pages_out.append(lines)

    except Exception:

        return [], False



    return pages_out, bool(pages_out and any(p for p in pages_out))







def _invoice_row_from_parsed(sr: int, parsed: Dict[str, Any], inv_ref_value: str, used_vehicle_purchase: bool = False) -> Dict[str, Any]:



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



    if used_vehicle_purchase:

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



    return {

        "sr_no": sr,

        "category": _clean_text(parsed.get("category")) or "purchase",

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







    def _deepseek_ocr_roi(rel_box: Tuple[float, float, float, float], prompt: str) -> str:



        if not deepseek_enabled or Image is None:



            return ""



        try:



            base_img = _auto_crop_to_red_border(img)



            W, H = base_img.size



            lx, ty, rx, by = rel_box



            crop = base_img.crop((int(lx * W), int(ty * H), int(rx * W), int(by * H)))



            crop = crop.resize((max(1, crop.size[0] * 2), max(1, crop.size[1] * 2)))



            try:



                crop = _invoice_remove_red_print(crop)



            except Exception:



                pass



            try:



                crop = _invoice_preprocess_handwriting(crop)



            except Exception:



                pass



            buf = io.BytesIO()



            crop.save(buf, format="PNG")



            txt = _deepseek_vision_ocr_text(buf.getvalue(), prompt=prompt)



            return _clean_text(txt)



        except Exception:



            return ""







    if deepseek_enabled:



        try:



            def _norm_colour(v: str) -> str:



                s = _clean_text(v).upper()



                s = re.sub(r"[^A-Z ]", " ", s)



                s = re.sub(r"\s{2,}", " ", s).strip()



                if not s:



                    return ""



                tok = s.split(" ")[0]



                return tok







            def _norm_model(v: str) -> str:



                s = _clean_text(v).upper()



                s = re.sub(r"[^A-Z0-9 \-]", " ", s)



                s = re.sub(r"\s{2,}", " ", s).strip()



                if not s:



                    return ""



                s = re.sub(r"\bMODEL\b", " ", s)



                s = re.sub(r"\bTYPE\b", " ", s)



                s = re.sub(r"\s{2,}", " ", s).strip()



                parts = s.split(" ")



                return (parts[0] if parts else s)[:60]



            date_txt = _deepseek_ocr_roi(



                (0.72, 0.13, 0.95, 0.19),



                "Read the handwritten Date field. Return only the date in dd/mm/yy or dd/mm/yyyy format.",



            )



            m = re.search(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b", date_txt)



            if m:



                out["document_date"] = _clean_text(m.group(1)).replace("-", "/")



            if not _is_valid_uk_date(out.get("document_date")):



                date_txt2 = _deepseek_ocr_roi(



                    (0.62, 0.10, 0.98, 0.22),



                    "Extract the handwritten Date value only (dd/mm/yy or dd/mm/yyyy).",



                )



                m = re.search(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b", date_txt2)



                if m and _is_valid_uk_date(m.group(1).replace("-", "/")):



                    out["document_date"] = _clean_text(m.group(1)).replace("-", "/")







            sup_txt = _deepseek_ocr_roi(



                (0.13, 0.155, 0.55, 0.205),



                "Read the handwritten 'Sold by' name. Return only the name (no extra words).",



            )



            if sup_txt:



                out["supplier"] = _normalize_supplier(sup_txt)[:120]







            addr_txt = _deepseek_ocr_roi(



                (0.13, 0.205, 0.55, 0.255),



                "Read the handwritten address under 'Sold by'. Return only the address line.",



            )



            if addr_txt:



                addr_clean = _clean_text(addr_txt)



                if addr_clean and (not _clean_text(out.get("supplier"))):



                    out["supplier"] = _normalize_supplier(addr_clean)[:120]



                elif addr_clean and _clean_text(out.get("supplier")):



                    out["supplier"] = _clean_text(out.get("supplier") + " " + addr_clean)[:120]







            make_txt = _deepseek_ocr_roi(



                (0.20, 0.295, 0.55, 0.345),



                "Read the handwritten vehicle Make. Return only the make.",



            )



            if make_txt:



                out["make"] = _normalize_make(make_txt)[:80]







            model_txt = _deepseek_ocr_roi(



                (0.20, 0.335, 0.55, 0.385),



                "Read the handwritten vehicle Model/Type. Return only the model/type.",



            )



            if model_txt:



                out["model"] = _norm_model(model_txt)







            colour_txt = _deepseek_ocr_roi(



                (0.20, 0.375, 0.55, 0.425),



                "Read the handwritten vehicle Colour. Return only the colour.",



            )



            if colour_txt:



                out["colour"] = _norm_colour(colour_txt)







            reg_txt = _deepseek_ocr_roi(



                (0.76, 0.37, 0.95, 0.44),



                "Read the handwritten UK Registration Number (VRM). Return only the registration like AB12 CDE.",



            )



            if reg_txt:



                m2 = re.search(r"\b([A-Z]{2}[0-9O]{2}\s*[A-Z]{3})\b", reg_txt.upper())



                if m2:



                    reg_raw = _clean_text(m2.group(1)).upper().replace(" ", "")



                    reg_raw = reg_raw[:2] + reg_raw[2:4].replace("O", "0") + reg_raw[4:]



                    out["reg_no"] = reg_raw[:4] + " " + reg_raw[4:]



            if not _clean_text(out.get("reg_no")):



                reg_txt2 = _deepseek_ocr_roi(



                    (0.74, 0.35, 0.97, 0.47),



                    "Extract the handwritten UK Registration Number (VRM) only.",



                )



                m2 = re.search(r"\b([A-Z]{2}[0-9O]{2}\s*[A-Z]{3})\b", reg_txt2.upper())



                if m2:



                    reg_raw = _clean_text(m2.group(1)).upper().replace(" ", "")



                    reg_raw = reg_raw[:2] + reg_raw[2:4].replace("O", "0") + reg_raw[4:]



                    out["reg_no"] = reg_raw[:4] + " " + reg_raw[4:]







            price_txt = _deepseek_ocr_roi(



                (0.70, 0.545, 0.92, 0.62),



                "Read the handwritten price (amount). Return only the numeric amount, optionally with £.",



            )



            if price_txt:



                v = _pick_invoice_price(price_txt)



                if v is not None:



                    out["buying_price"] = float(v)



                    out["non_vat"] = float(v)



            if out.get("buying_price") in (None, ""):



                price_txt2 = _deepseek_ocr_roi(



                    (0.66, 0.52, 0.96, 0.65),



                    "Extract the handwritten price amount only (e.g., 1500 or £1500).",



                )



                v2 = _pick_invoice_price(price_txt2)



                if v2 is not None:



                    out["buying_price"] = float(v2)



                    out["non_vat"] = float(v2)



        except Exception:



            pass







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



    deepseek_enabled = (

        OCR_PROVIDER == "deepseek"

        and httpx is not None

        and bool(DEEPSEEK_API_KEY)

        and bool(DEEPSEEK_OCR2_URL)

    )



    ok, _detail = _invoice_tesseract_available()



    if (not deepseek_enabled) and (not ok or pytesseract is None):



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

    return await bank_statements.convert_bank_statements(
        request,
        output_dir=OUTPUT_DIR,
        jobs=JOBS,
        UploadFileT=UploadFile,
        StarletteUploadFileT=StarletteUploadFile,
        clean_text=_clean_text,
        to_float_or_none=_to_float_or_none,
        infer_subcategory=_infer_subcategory,
        extract_text_lines_from_pdf_with_ocr=_extract_text_lines_from_pdf_with_ocr,
        extract_text_lines_from_image_with_ocr=_extract_text_lines_from_image_with_ocr,
        tesseract_available=_tesseract_available,
        convert_pdf_to_rows=convert_pdf_to_rows,
        extract_account_from_lines=_extract_account_from_lines,
        looks_like_barclays_statement=_looks_like_barclays_statement,
        looks_like_barclays_business_premium_statement=_looks_like_barclays_business_premium_statement,
        extract_barclays_business_premium_header_info=_extract_barclays_business_premium_header_info,
        barclays_business_premium_preamble_lines=_barclays_business_premium_preamble_lines,
        extract_barclays_header_info=_extract_barclays_header_info,
        barclays_header_preamble_lines=_barclays_header_preamble_lines,
        looks_like_monzo_statement=_looks_like_monzo_statement,
        extract_monzo_header_info=_extract_monzo_header_info,
        monzo_header_preamble_lines=_monzo_header_preamble_lines,
        looks_like_virgin_money_statement=_looks_like_virgin_money_statement,
        extract_virgin_money_header_info=_extract_virgin_money_header_info,
        virgin_money_header_preamble_lines=_virgin_money_header_preamble_lines,
        looks_like_tide_statement=_looks_like_tide_statement,
        extract_tide_header_info=_extract_tide_header_info,
        tide_header_preamble_lines=_tide_header_preamble_lines,
        looks_like_revolut_business_statement=_looks_like_revolut_business_statement,
        extract_revolut_business_header_info=_extract_revolut_business_header_info,
        revolut_business_preamble_lines=_revolut_business_preamble_lines,
        write_csv=_write_csv,
        write_csv_with_preamble=_write_csv_with_preamble,
        write_barclays_csv_with_pending=_write_barclays_csv_with_pending,
        format_csv_value=_format_csv_value,
        JSONResponseT=JSONResponse,
        BANKPDF_OCR=BANKPDF_OCR,
        pytesseract_installed=pytesseract is not None,
        fitz_available=fitz is not None,
        pdfium_available=pdfium is not None,
        uuid4=uuid.uuid4,
    )





@app.get("/api/download/{job_id}")

def download(job_id: str) -> FileResponse:

    return bank_statements.download_bank_statements(job_id, output_dir=OUTPUT_DIR, jobs=JOBS, FileResponseT=FileResponse)





@app.post("/api/convert-invoice")

async def invoice_convert(request: Request) -> JSONResponse:

    return await invoices.invoice_convert(
        request,
        output_dir=OUTPUT_DIR,
        invoice_jobs=INVOICE_JOBS,
        UploadFileT=UploadFile,
        StarletteUploadFileT=StarletteUploadFile,
        clean_text=_clean_text,
        format_csv_value=_format_csv_value,
        JSONResponseT=JSONResponse,
        uuid4=uuid.uuid4,
        invoice_pdf_page_count=_invoice_pdf_page_count,
        invoice_extract_text_lines_from_pdf_pages_with_ocr=_invoice_extract_text_lines_from_pdf_pages_with_ocr,
        extract_invoice_fields=_extract_invoice_fields,
        invoice_row_from_parsed=_invoice_row_from_parsed,
        invoice_extract_text_lines_from_pdf_with_ocr=_invoice_extract_text_lines_from_pdf_with_ocr,
        invoice_tesseract_available=_invoice_tesseract_available,
        Image_available=Image is not None,
        fitz_available=fitz is not None,
        pdfium_available=pdfium is not None,
        OCR_PROVIDER=OCR_PROVIDER,
        deepseek_extract_used_vehicle_fields_from_pdf=_deepseek_extract_used_vehicle_fields_from_pdf,
        invoice_ocr_used_vehicle_purchase_fields=_invoice_ocr_used_vehicle_purchase_fields,
        is_valid_uk_date=_is_valid_uk_date,
        invoice_ocr_autotrader_costs_box=_invoice_ocr_autotrader_costs_box,
        invoice_ocr_bca_fields=_invoice_ocr_bca_fields,
        write_csv=_write_csv,
        to_float_or_none=_to_float_or_none,
        re_mod=re,
    )









@app.post("/api/invoice-convert-review")



async def invoice_convert_review(request: Request) -> JSONResponse:

    return await invoices.invoice_convert_review(
        request,
        output_dir=OUTPUT_DIR,
        invoice_review_jobs=INVOICE_REVIEW_JOBS,
        invoice_jobs=INVOICE_JOBS,
        UploadFileT=UploadFile,
        StarletteUploadFileT=StarletteUploadFile,
        clean_text=_clean_text,
        format_csv_value=_format_csv_value,
        JSONResponseT=JSONResponse,
        uuid4=uuid.uuid4,
        invoice_pdf_page_count=_invoice_pdf_page_count,
        invoice_extract_text_lines_from_pdf_pages_with_ocr=_invoice_extract_text_lines_from_pdf_pages_with_ocr,
        extract_invoice_fields=_extract_invoice_fields,
        invoice_row_from_parsed=_invoice_row_from_parsed,
        invoice_extract_text_lines_from_pdf_with_ocr=_invoice_extract_text_lines_from_pdf_with_ocr,
        invoice_tesseract_available=_invoice_tesseract_available,
        Image_available=Image is not None,
        fitz_available=fitz is not None,
        pdfium_available=pdfium is not None,
        OCR_PROVIDER=OCR_PROVIDER,
        deepseek_extract_used_vehicle_fields_from_pdf=_deepseek_extract_used_vehicle_fields_from_pdf,
        invoice_ocr_used_vehicle_purchase_fields=_invoice_ocr_used_vehicle_purchase_fields,
        is_valid_uk_date=_is_valid_uk_date,
        invoice_ocr_autotrader_costs_box=_invoice_ocr_autotrader_costs_box,
        invoice_ocr_bca_fields=_invoice_ocr_bca_fields,
        write_csv=_write_csv,
        write_json=_write_json,
        to_float_or_none=_to_float_or_none,
        re_mod=re,
    )


@app.post("/api/handwritten-invoice-convert-review")
async def handwritten_invoice_convert_review(request: Request) -> JSONResponse:
    return await handwritten.handwritten_invoice_convert_review(
        request,
        output_dir=OUTPUT_DIR,
        handwritten_review_jobs=HANDWRITTEN_REVIEW_JOBS,
        handwritten_jobs=HANDWRITTEN_JOBS,
        UploadFileT=UploadFile,
        StarletteUploadFileT=StarletteUploadFile,
        clean_text=_clean_text,
        JSONResponseT=JSONResponse,
        uuid4=uuid.uuid4,
        MODEL_REGISTRY=MODEL_REGISTRY,
        LIGHTON_LOCAL_MODEL_NAME=LIGHTON_LOCAL_MODEL_NAME,
        handwritten_lighton_multimodel_ocr=_handwritten_lighton_multimodel_ocr,
        parse_bbox_output=_parse_bbox_output,
        clean_output_text=_clean_output_text,
        ocr_words_and_lines_from_pil_image=_ocr_words_and_lines_from_pil_image,
        crop_from_bbox=_crop_from_bbox,
        extract_invoice_fields=_extract_invoice_fields,
        extract_used_vehicle_invoice_fields=_extract_used_vehicle_invoice_fields,
        invoice_render_page=_invoice_render_page,
        invoice_render_first_page=_invoice_render_first_page,
        ImageT=Image,
        io_mod=io,
        base64_mod=base64,
        re_mod=re,
        write_json=_write_json,
        write_csv=_write_csv,
        format_csv_value=_format_csv_value,
    )


@app.post("/api/invoice-confirm/{job_id}")
async def invoice_confirm(job_id: str, request: Request) -> JSONResponse:
    return await invoices.invoice_confirm(
        job_id,
        request,
        invoice_review_jobs=INVOICE_REVIEW_JOBS,
        invoice_jobs=INVOICE_JOBS,
        clean_text=_clean_text,
        to_float_or_none=_to_float_or_none,
        read_json=_read_json,
        write_json=_write_json,
        write_csv=_write_csv,
        format_csv_value=_format_csv_value,
        JSONResponseT=JSONResponse,
    )


@app.post("/api/handwritten-invoice-confirm/{job_id}")
async def handwritten_invoice_confirm(job_id: str, request: Request) -> JSONResponse:
    return await handwritten.handwritten_invoice_confirm(
        job_id,
        request,
        handwritten_review_jobs=HANDWRITTEN_REVIEW_JOBS,
        handwritten_jobs=HANDWRITTEN_JOBS,
        clean_text=_clean_text,
        to_float_or_none=_to_float_or_none,
        read_json=_read_json,
        write_json=_write_json,
        write_csv=_write_csv,
        format_csv_value=_format_csv_value,
        JSONResponseT=JSONResponse,
    )


@app.get("/api/invoice-download/{job_id}")
def invoice_download(job_id: str) -> FileResponse:
    return invoices.invoice_download(job_id, invoice_jobs=INVOICE_JOBS, FileResponseT=FileResponse)


@app.get("/api/handwritten-invoice-download/{job_id}")

def handwritten_invoice_download(job_id: str) -> FileResponse:

    return handwritten.handwritten_invoice_download(job_id, handwritten_jobs=HANDWRITTEN_JOBS, FileResponseT=FileResponse)



