# Bank Statement PDF → CSV (Local Web App)

## Run

1) Create venv (optional) and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2) Start server:

```powershell
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

3) Open:

http://127.0.0.1:8000

## Notes

- Works best with text-based PDFs (tables embedded as text).
- If your PDFs are scanned images, OCR support can be added (Tesseract).

## OCR (Scanned PDFs)

If your PDF is scanned (you cannot select/copy text), install Tesseract OCR and enable OCR fallback.

1) Install Tesseract for Windows:

- Download and install "Tesseract OCR" (Windows installer)

2) Ensure your app can find `tesseract.exe`:

- Option A: add Tesseract install folder to your PATH
- Option B: set environment variable `TESSERACT_CMD` to the full path, e.g.:

```powershell
$env:TESSERACT_CMD = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
```

3) Install Python OCR dependency:

```powershell
pip install -r requirements.txt
```

4) (Optional) disable OCR fallback:

```powershell
$env:BANKPDF_OCR = "0"
```
