#!/usr/bin/env python3
"""Test script for Amazon invoice format"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import main
    print("✓ Successfully imported main module")
except ImportError as e:
    print(f"✗ Failed to import main module: {e}")
    sys.exit(1)


def test_amazon_detection_and_fields() -> bool:
    # Sample lines based on the provided Amazon invoice layout
    test_lines = [
        "Invoice",
        "Paid",
        "Sold by MH STAR UK LTD",
        "VAT # GB103973325",
        "Invoice date / Delivery date 01 April 2024",
        "Invoice # INV-GB-124542991-2024-256656",
        "Total payable £124.99",
        "SONNY ANDERSON",
        "72, CEDAR ROAD",
        "ROCHESTER, ME2 2JP",
        "GB",
        "Invoice total £124.99",
        "VAT subtotal £20.83",
        "Item subtotal (excl. VAT) £104.16",
    ]

    cleaned = [main._clean_text(ln) for ln in test_lines]
    cleaned = [ln for ln in cleaned if ln]

    parsed = main._extract_invoice_fields(cleaned)

    if (main._clean_text(parsed.get("document_date")) or "").strip() != "01/04/2024":
        print(f"✗ document_date mismatch: {parsed.get('document_date')}")
        return False

    if (main._clean_text(parsed.get("inv_ref_no")) or "").strip() != "INV-GB-124542991-2024-256656":
        print(f"✗ inv_ref_no mismatch: {parsed.get('inv_ref_no')}")
        return False

    supplier = (main._clean_text(parsed.get("supplier")) or "")
    if "SONNY" not in supplier.upper() or "CEDAR" not in supplier.upper() or "ROCHESTER" not in supplier.upper():
        print(f"✗ supplier mismatch: {parsed.get('supplier')}")
        return False

    if (main._clean_text(parsed.get("make")) or "").strip() != "MH STAR UK LTD":
        print(f"✗ make mismatch: {parsed.get('make')}")
        return False

    if parsed.get("buying_price") != 124.99 or parsed.get("non_vat") != 124.99:
        print(f"✗ total mismatch: buying_price={parsed.get('buying_price')} non_vat={parsed.get('non_vat')}")
        return False

    if parsed.get("vat_amount") != 20.83:
        print(f"✗ vat_amount mismatch: {parsed.get('vat_amount')}")
        return False

    if parsed.get("std_net") != 104.16:
        print(f"✗ std_net mismatch: {parsed.get('std_net')}")
        return False

    if (main._clean_text(parsed.get("reg_no")) or "") != "GB103973325":
        print(f"✗ reg_no mismatch: {parsed.get('reg_no')}")
        return False

    if parsed.get("category") not in ("purchase", "expense", "sale"):
        print(f"✗ category missing/invalid: {parsed.get('category')}")
        return False

    print("✓ Amazon detection and field extraction passed")
    return True


def main_test() -> bool:
    print("Testing Amazon invoice format implementation...")
    print("=" * 60)

    ok = test_amazon_detection_and_fields()

    print("\n" + "=" * 60)
    if ok:
        print("All tests passed! Amazon format is ready.")
        return True
    print("Some tests failed. Please check the implementation.")
    return False


if __name__ == "__main__":
    raise SystemExit(0 if main_test() else 1)
