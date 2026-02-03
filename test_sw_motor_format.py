#!/usr/bin/env python3
"""Test script for SW Motor Factors Ltd invoice format"""

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


def test_sw_motor_detection_and_fields() -> bool:
    # Sample lines based on provided SW Motor Factors invoice layout
    test_lines = [
        "SW MOTOR FACTORS LTD",
        "Invoice",
        "Invoice From:",
        "SW MOTOR FACTORS LTD",
        "UNIT B",
        "DORRINGTON FARM",
        "RYE HILL ROAD",
        "HARLOW",
        "ESSEX",
        "CM18 7JF",
        "Date 10/10/24",
        "Invoice No. H1020146",
        "Part Number Description Location Qty Unit Price Ext Cost",
        "VE360271 EXHAUST PRESSURE 1 45.48 45.48",
        "VAT Reg No: GB 573 1078 45",
        "GOODS VALUE: 45.48",
        "VAT: 9.10",
        "TOTAL: 54.58",
    ]

    cleaned = [main._clean_text(ln) for ln in test_lines]
    cleaned = [ln for ln in cleaned if ln]

    parsed = main._extract_invoice_fields(cleaned)

    if (main._clean_text(parsed.get("document_date")) or "").strip() != "10/10/2024":
        print(f"✗ document_date mismatch: {parsed.get('document_date')}")
        return False

    if (main._clean_text(parsed.get("inv_ref_no")) or "").strip() != "H1020146":
        print(f"✗ inv_ref_no mismatch: {parsed.get('inv_ref_no')}")
        return False

    supplier = (main._clean_text(parsed.get("supplier")) or "")
    if "SW MOTOR FACTORS LTD" not in supplier.upper():
        print(f"✗ supplier mismatch: {parsed.get('supplier')}")
        return False

    make = (main._clean_text(parsed.get("make")) or "")
    if "EXHAUST" not in make.upper():
        print(f"✗ make mismatch: {parsed.get('make')}")
        return False

    if parsed.get("buying_price") != 54.58 or parsed.get("non_vat") != 54.58:
        print(f"✗ total mismatch: buying_price={parsed.get('buying_price')} non_vat={parsed.get('non_vat')}")
        return False

    if parsed.get("vat_amount") != 9.10:
        print(f"✗ vat_amount mismatch: {parsed.get('vat_amount')}")
        return False

    if (main._clean_text(parsed.get("reg_no")) or "") != "GB573107845":
        print(f"✗ reg_no mismatch: {parsed.get('reg_no')}")
        return False

    if parsed.get("category") not in ("purchase", "expense", "sale"):
        print(f"✗ category missing/invalid: {parsed.get('category')}")
        return False

    print("✓ SW Motor Factors detection and field extraction passed")
    return True


def main_test() -> bool:
    print("Testing SW Motor Factors Ltd invoice format implementation...")
    print("=" * 60)

    ok = test_sw_motor_detection_and_fields()

    print("\n" + "=" * 60)
    if ok:
        print("All tests passed! SW Motor Factors format is ready.")
        return True
    print("Some tests failed. Please check the implementation.")
    return False


if __name__ == "__main__":
    raise SystemExit(0 if main_test() else 1)
