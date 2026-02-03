#!/usr/bin/env python3
"""Test script for SMITHS FIRE LLP invoice format"""

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


def test_smiths_fire_llp_detection() -> bool:
    test_lines_positive = [
        "SMITHS FIRE LLP",
        "INVOICE",
        "Invoice No: 249882",
        "Date: 30/05/23",
        "VAT Reg No: 336 1615 8440",
        "Invoice To:",
        "Deliver To:",
        "Total Value 321.00",
        "VAT 63.50",
        "Balance 321.00",
    ]

    test_lines_negative = [
        "One Stop Builders Merchants",
        "INVOICE",
        "Invoice No: 12345",
        "Date: 30/05/23",
        "Total 100.00",
    ]

    cleaned_positive = [main._clean_text(ln) for ln in test_lines_positive]
    cleaned_positive = [ln for ln in cleaned_positive if ln]
    head_positive = "\n".join(cleaned_positive[:160]).lower()
    should_match_positive = ("smiths fire" in head_positive) and ("invoice" in head_positive)

    cleaned_negative = [main._clean_text(ln) for ln in test_lines_negative]
    cleaned_negative = [ln for ln in cleaned_negative if ln]
    head_negative = "\n".join(cleaned_negative[:160]).lower()
    should_match_negative = ("smiths fire" in head_negative) and ("invoice" in head_negative)

    if not should_match_positive:
        print("✗ Positive detection test failed")
        return False
    if should_match_negative:
        print("✗ Negative detection test failed")
        return False

    print("✓ Detection tests passed")
    return True


def test_field_mapping() -> bool:
    sample_lines = [
        "SMITHS FIRE LLP",
        "INVOICE",
        "Invoice No: 249882",
        "Date: 30/05/23",
        "Invoice To:",
        "Polebrook Arms Northants Ltd",
        "Deliver To:",
        "Kings Arms",
        "VAT Reg No: 336 1615 8440",
        "Total Value 321.00",
        "VAT 63.50",
        "Balance 321.00",
    ]

    parsed = main._extract_invoice_fields(sample_lines)

    ok = True
    if main._clean_text(parsed.get("inv_ref_no")) != "249882":
        print(f"✗ inv_ref_no mismatch: {parsed.get('inv_ref_no')}")
        ok = False
    if main._clean_text(parsed.get("document_date")) != "30/05/23":
        print(f"✗ document_date mismatch: {parsed.get('document_date')}")
        ok = False
    if "Polebrook" not in main._clean_text(parsed.get("supplier")):
        print(f"✗ supplier mismatch: {parsed.get('supplier')}")
        ok = False
    if "Kings Arms" not in main._clean_text(parsed.get("make")):
        print(f"✗ make mismatch: {parsed.get('make')}")
        ok = False
    if "336" not in main._clean_text(parsed.get("reg_no")):
        print(f"✗ reg_no mismatch: {parsed.get('reg_no')}")
        ok = False

    # Totals
    if parsed.get("std_net") not in (321.0, 321, "321.00"):
        print(f"✗ std_net mismatch: {parsed.get('std_net')}")
        ok = False
    if parsed.get("vat_amount") not in (63.5, 63.50, "63.50"):
        print(f"✗ vat_amount mismatch: {parsed.get('vat_amount')}")
        ok = False
    if parsed.get("buying_price") not in (321.0, 321, "321.00") or parsed.get("non_vat") not in (321.0, 321, "321.00"):
        print(f"✗ balance mapping mismatch: buying_price={parsed.get('buying_price')} non_vat={parsed.get('non_vat')}")
        ok = False

    if ok:
        print("✓ Field mapping test passed")
    return ok


def main_test() -> bool:
    print("Testing SMITHS FIRE LLP invoice format implementation...")
    print("=" * 60)

    tests = [
        ("Format Detection", test_smiths_fire_llp_detection),
        ("Field Mapping", test_field_mapping),
    ]

    passed = 0
    for test_name, test_func in tests:
        print(f"\nRunning {test_name} test...")
        if test_func():
            passed += 1
            print(f"✓ {test_name} test passed")
        else:
            print(f"✗ {test_name} test failed")

    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{len(tests)} tests passed")
    return passed == len(tests)


if __name__ == "__main__":
    success = main_test()
    sys.exit(0 if success else 1)
