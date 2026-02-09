
from PIL import Image, ImageDraw, ImageFont
import os

def create_dummy_invoice(filepath):
    # Create white image
    img = Image.new('RGB', (800, 1000), color='white')
    d = ImageDraw.Draw(img)
    
    # Try to load a font, otherwise default
    try:
        # Looking for a font that might look slightly less "perfect" or just standard arial
        font = ImageFont.truetype("arial.ttf", 24)
        header_font = ImageFont.truetype("arial.ttf", 36)
    except IOError:
        font = ImageFont.load_default()
        header_font = ImageFont.load_default()

    # Draw "Handwritten" content
    d.text((300, 50), "INVOICE", fill=(0, 0, 0), font=header_font)
    
    d.text((50, 150), "Date: 12/05/2024", fill=(0, 0, 200), font=font)
    d.text((50, 200), "From: John Smith Services", fill=(0, 0, 200), font=font)
    d.text((50, 250), "To: Jane Doe", fill=(0, 0, 200), font=font)
    
    d.text((50, 350), "Description              Amount", fill=(0, 0, 0), font=font)
    d.line((50, 380, 750, 380), fill=(0, 0, 0), width=2)
    
    d.text((50, 400), "Garden Maintenance       $150.00", fill=(0, 0, 200), font=font)
    d.text((50, 450), "Materials                $50.00", fill=(0, 0, 200), font=font)
    
    d.line((50, 500, 750, 500), fill=(0, 0, 0), width=2)
    d.text((400, 520), "Total: $200.00", fill=(0, 0, 200), font=font)
    
    d.text((50, 600), "Thank you for your business!", fill=(0, 0, 100), font=font)

    # Save
    img.save(filepath)
    print(f"Created dummy invoice at {filepath}")

if __name__ == "__main__":
    os.makedirs("handwritten_invoices", exist_ok=True)
    create_dummy_invoice("handwritten_invoices/test_invoice.png")
