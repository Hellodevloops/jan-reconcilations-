
import os
import io
import base64
import re
from typing import Any, Dict, List, Optional, Tuple
import traceback

print("DEBUG: Loading ocr_utils.py v3...", flush=True)

try:
    import numpy as np
except ImportError:
    np = None
    print("DEBUG: numpy not available", flush=True)

try:
    import httpx
except ImportError:
    httpx = None

try:
    import torch
except ImportError:
    torch = None

try:
    from PIL import Image
except ImportError:
    Image = None

# OCR Configuration
LIGHTON_LOCAL_ENABLED = os.getenv("LIGHTON_LOCAL_ENABLED", "").strip().lower() in {"1", "true", "yes", "y", "on"}
MODEL_REGISTRY = {
    "LightOnOCR-2-1B (Best OCR)": {
        "model_id": "lightonai/LightOnOCR-2-1B",
        "vllm_endpoint": "", 
        "has_bbox": True
    }
}

_EASYOCR_READER = None

def get_easyocr_reader():
    global _EASYOCR_READER
    if _EASYOCR_READER is None:
        try:
            import easyocr
            # Use GPU/CUDA if available
            use_gpu = torch.cuda.is_available() if torch else False
            print(f"DEBUG: Initializing EasyOCR (GPU={use_gpu}, CUDA available={torch.cuda.is_available() if torch else 'N/A'})...", flush=True)
            _EASYOCR_READER = easyocr.Reader(['en'], gpu=use_gpu, verbose=True)
            print(f"DEBUG: EasyOCR initialized successfully", flush=True)
        except Exception as e:
            traceback.print_exc()
            print(f"DEBUG: EasyOCR Init Failed: {e}", flush=True)
            _EASYOCR_READER = None
    return _EASYOCR_READER

def clean_output_text(text: str) -> str:
    if not text: return ""
    return text.strip()

def _easyocr_fallback(image_png_bytes: bytes) -> str:
    print("DEBUG: Inside _easyocr_fallback", flush=True)
    
    if np is None:
        print("DEBUG: numpy not available", flush=True)
        return ""
        
    reader = get_easyocr_reader()
    if not reader:
        print("DEBUG: No EasyOCR reader returned", flush=True)
        return ""
    if not Image:
        print("DEBUG: No PIL Image module", flush=True)
        return ""
    
    try:
        img = Image.open(io.BytesIO(image_png_bytes)).convert('RGB')
        print(f"DEBUG: Image opened, size: {img.size}", flush=True)
        
        img_np = np.array(img)
        print(f"DEBUG: Converted to numpy array, shape: {img_np.shape}", flush=True)
        
        # Use detail=1 to get bounding boxes and text, then extract just text
        print("DEBUG: Running easyocr readtext (detail=1)...", flush=True)
        results = reader.readtext(img_np, detail=1, paragraph=False)
        print(f"DEBUG: EasyOCR found {len(results)} text regions", flush=True)
        
        # Extract text from results: each result is (bbox, text, confidence)
        texts = []
        for r in results:
            if len(r) >= 2:
                txt = str(r[1]).strip()
                if txt:
                    texts.append(txt)
        
        full_text = " ".join(texts)
        print(f"DEBUG: Combined text length: {len(full_text)} chars", flush=True)
        return full_text
        
    except Exception as e:
        traceback.print_exc()
        print(f"DEBUG: EasyOCR processing failed: {e}", flush=True)
        return ""

def handwritten_lighton_multimodel_ocr(image_png_bytes: bytes, model_name: str = "LightOnOCR-2-1B (Best OCR)", temperature: float = 0.2, max_tokens: int = 2048) -> str:
    print("DEBUG: handwritten_lighton_multimodel_ocr called", flush=True)
    
    # Try EasyOCR as primary method for now (LightOn requires model download)
    print("DEBUG: Calling EasyOCR...", flush=True)
    txt = _easyocr_fallback(image_png_bytes)
    if txt:
        return txt
        
    print("DEBUG: OCR returned empty text.", flush=True)
    return ""
