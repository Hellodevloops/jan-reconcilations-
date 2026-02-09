# Run project & libraries/models summary

## Commands to run (in order)

Run these in **PowerShell** from the project folder `c:\ai-reconcilations`:

### 1. Create virtual environment (optional but recommended)
```powershell
cd c:\ai-reconcilations
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies
```powershell
pip install -r requirements.txt
```

### 3. Start the server

**If "running scripts is disabled" in PowerShell**, use the venv’s Python directly (no activation):
```powershell
cd c:\ai-reconcilations
.\.venv\Scripts\python.exe -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

**If the venv activates successfully**, you can use:
```powershell
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

**Optional:** Allow PowerShell to run scripts (once per user) so activation works:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 4. Open in browser
http://127.0.0.1:8000

---

## If Python is not found

- Install Python from https://www.python.org/downloads/ (e.g. 3.10 or 3.11).
- During install, check **“Add Python to PATH”**.
- Restart the terminal, then run the commands above.

---

## Optional: Tesseract OCR (for scanned PDFs)

- Download and install **Tesseract OCR** for Windows.
- Add Tesseract to PATH or set:
  ```powershell
  $env:TESSERACT_CMD = "C:\Program Files\Tesseract-OCR\tesseract.exe"
  ```

---

## Libraries used (from `requirements.txt`)

| Library | Version | Purpose |
|--------|---------|--------|
| **fastapi** | 0.110.0 | Web API framework |
| **uvicorn** | 0.27.1 | ASGI server to run the app |
| **python-multipart** | 0.0.9 | Form/file uploads |
| **jinja2** | 3.1.3 | HTML templates |
| **pdfplumber** | 0.10.3 | Extract text/tables from PDFs |
| **pypdfium2** | 4.30.0 | PDF rendering (images) |
| **pytesseract** | 0.3.10 | Tesseract OCR wrapper |
| **Pillow** | 11.1.0 | Image processing |
| **httpx** | 0.27.0 | HTTP client (e.g. for APIs) |
| **python-dotenv** | 1.0.1 | Load `.env` config |
| **easyocr** | 1.7.2 | Deep-learning OCR (English) |
| **opencv-python-headless** | 4.11.0.86 | Image processing for OCR |

Optional (imported in code if installed): **torch**, **transformers** (for LightOn OCR models).

---

## Models used (OCR / AI)

### 1. **Tesseract OCR**
- **Type:** Local OCR engine (not a Python package; separate install).
- **Use:** Default OCR for scanned PDFs and invoices when no AI model is used.
- **Config:** `OCR_PROVIDER=tesseract` (default), or set `TESSERACT_CMD` to the executable path.

### 2. **EasyOCR**
- **Type:** Deep-learning OCR (runs locally).
- **Model:** English (`en`) reader; can use GPU if available.
- **Use:** Alternative OCR when `OCR_PROVIDER=easyocr` or when used alongside other engines.

### 3. **LightOn OCR (Hugging Face)**
- **Type:** Vision/language models for OCR (via `transformers` + optional `torch`).
- **Model IDs used in the app:**
  - `lightonai/LightOnOCR-2-1B` – main OCR
  - `lightonai/LightOnOCR-2-1B-bbox` – with bounding boxes
  - `lightonai/LightOnOCR-2-1B-base`
  - `lightonai/LightOnOCR-2-1B-bbox-base`
  - `lightonai/LightOnOCR-2-1B-ocr-soup`
  - `lightonai/LightOnOCR-2-1B-bbox-soup`
- **Use:** Handwritten and document OCR when “LightOn” or vLLM is selected in the UI/settings.

### 4. **vLLM-served models**
- **Type:** Same LightOn model IDs but served via a **vLLM** server (optional).
- **Use:** When `VLLM_ENDPOINT_OCR` / `VLLM_ENDPOINT_BBOX` are set in env; the app sends images to that API instead of loading the model in-process.

### 5. **DeepSeek (API)**
- **Type:** Cloud vision API (OpenAI-compatible).
- **Model:** e.g. `deepseek-vl` (configurable via env).
- **Use:** Optional OCR for invoices when DeepSeek API key and endpoint are configured.

---

## What this project does

- **Bank statements:** Upload PDF bank statements → extract transactions → download as CSV.
- **Invoices:** Upload invoice PDFs (including scanned) → extract fields (date, supplier, model, reg no, price, etc.) using OCR (Tesseract, EasyOCR, or LightOn/DeepSeek when configured).
- **Handwritten:** Handwritten invoice/images supported via EasyOCR and LightOn (local or vLLM).

The app is a **FastAPI** web app; you run it with **uvicorn** and use the browser at **http://127.0.0.1:8000** to upload files and get results.
