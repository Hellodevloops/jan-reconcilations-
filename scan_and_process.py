
import os
import json
import asyncio
from typing import List, Dict, Any
from ocr_utils import handwritten_lighton_multimodel_ocr

# Directory definitions
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INVOICES_DIR = os.path.join(BASE_DIR, "handwritten_invoices")
OUTPUT_DIR = os.path.join(INVOICES_DIR, "output")

try:
    from PIL import Image
except ImportError:
    Image = None

def convert_to_json_format(filename: str, raw_text: str) -> Dict[str, Any]:
    """
    Convert raw OCR text into a structured JSON format.
    """
    words = raw_text.split()
    return {
        "filename": filename,
        "full_text": raw_text,
        "word_count": len(words),
        "words": words
    }

def process_image_bytes(filename: str, png_bytes: bytes) -> Dict[str, Any]:
    raw_text = handwritten_lighton_multimodel_ocr(png_bytes)
    if not raw_text:
        return {"filename": filename, "error": "OCR returned empty text"}
    return convert_to_json_format(filename, raw_text)

def main():
    os.makedirs(INVOICES_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Scanning for invoices in: {INVOICES_DIR}")
    
    files = [f for f in os.listdir(INVOICES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.pdf'))]
    all_results = []
    
    if not files:
        print("No image/PDF files found.")
        return

    for f in files:
        filepath = os.path.join(INVOICES_DIR, f)
        print(f"Processing: {f}")
        
        # PDF Handling (All Pages)
        if f.lower().endswith('.pdf'):
            try:
                import pypdfium2 as pdfium
                doc = pdfium.PdfDocument(filepath)
                
                pdf_text = []
                for i, page in enumerate(doc):
                    print(f"  - Rendering page {i+1}...")
                    bitmap = page.render(scale=2)
                    pil_image = bitmap.to_pil()
                    
                    import io
                    buf = io.BytesIO()
                    pil_image.save(buf, format="PNG")
                    png_bytes = buf.getvalue()
                    
                    page_res = process_image_bytes(f"page_{i+1}", png_bytes)
                    if "full_text" in page_res:
                        pdf_text.append(page_res["full_text"])
                
                full_pdf_text = "\n\n".join(pdf_text)
                if full_pdf_text:
                    res = convert_to_json_format(f, full_pdf_text)
                    all_results.append(res)
                else:
                    all_results.append({"filename": f, "error": "No text extracted from PDF"})
                    
            except ImportError:
                 all_results.append({"filename": f, "error": "pypdfium2 not installed"})
            except Exception as e:
                 all_results.append({"filename": f, "error": str(e)})
            continue

        # Image Handling
        try:
            with Image.open(filepath) as img:
                img = img.convert("RGB")
                import io
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                png_bytes = buf.getvalue()
                
            res = process_image_bytes(f, png_bytes)
            all_results.append(res)
            
        except Exception as e:
            all_results.append({"filename": f, "error": str(e)})

        # Save individual JSON
        # (For the last processed item - careful with loop logic if needed per-file)
        # Better to save here inside the loop for the current file
        if all_results:
             current_result = all_results[-1]
             json_name = f"{os.path.splitext(f)[0]}.json"
             with open(os.path.join(OUTPUT_DIR, json_name), "w", encoding="utf-8") as jf:
                 json.dump(current_result, jf, indent=2)

    # Save Combined JSON
    with open(os.path.join(OUTPUT_DIR, "all_invoices.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
        
    print(f"Processing complete. {len(all_results)} files processed.")

if __name__ == "__main__":
    main()
