#!/usr/bin/env python3
"""Test script for J Wilson Plumbing & Heating invoice format"""

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


def test_jwilson_detection_and_fields():
    # Sample lines based on the provided J Wilson invoice layout
    test_lines = [
        "INVOICE",
        "Invoice Date",
        "25 Aug 2023",
        "Invoice Number",
        "INV-12467",
        "Reference",
        "49756 - 1 Athena Way, Oundle",
        "J Wilson Plumbing & Heating",
        "Ltd",
        "Unit 8 Caserton Business",
        "Park",
        "The Old Great North Road",
        "STAMFORD",
        "Lincolnshire",
        "PE9 4DE",
        "GBR",
        "Subtotal",
        "60.00",
        "Total VAT 20%",
        "12.00",
        "Amount Due GBP",
        "72.00",
    ]

    cleaned = [main._clean_text(ln) for ln in test_lines]
    cleaned = [ln for ln in cleaned if ln]

    parsed = main._extract_invoice_fields(cleaned)

    # document_date
    if (main._clean_text(parsed.get("document_date")) or "").strip() != "25/08/2023":
        print(f"✗ document_date mismatch: {parsed.get('document_date')}")
        return False

    # inv_ref_no
    if (main._clean_text(parsed.get("inv_ref_no")) or "").strip() != "INV-12467":
        print(f"✗ inv_ref_no mismatch: {parsed.get('inv_ref_no')}")
        return False

    # make (reference)
    if (main._clean_text(parsed.get("make")) or "").strip() != "49756 - 1 Athena Way, Oundle":
        print(f"✗ make mismatch: {parsed.get('make')}")
        return False

    # totals
    if parsed.get("buying_price") != 72.00 or parsed.get("non_vat") != 72.00:
        print(f"✗ amount due mismatch: buying_price={parsed.get('buying_price')} non_vat={parsed.get('non_vat')}")
        return False

    if parsed.get("vat_amount") != 12.00:
        print(f"✗ vat_amount mismatch: {parsed.get('vat_amount')}")
        return False

    if parsed.get("std_net") != 60.00:
        print(f"✗ std_net mismatch: {parsed.get('std_net')}")
        return False

    supplier = (main._clean_text(parsed.get("supplier")) or "").lower()
    if "j wilson" not in supplier or "plumbing" not in supplier or "heating" not in supplier:
        print(f"✗ supplier mismatch: {parsed.get('supplier')}")
        return False

    print("✓ J Wilson detection and field extraction passed")
    return True


def main_test():
    print("Testing J Wilson Plumbing & Heating invoice format implementation...")
    print("=" * 70)

    ok = test_jwilson_detection_and_fields()

    print("\n" + "=" * 70)
    if ok:
        print("🎉 All tests passed! J Wilson format is ready.")
        return True

    print("❌ Test failed. Please check the implementation.")
    return False


if __name__ == "__main__":
    success = main_test()
    sys.exit(0 if success else 1)
