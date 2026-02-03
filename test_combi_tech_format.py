#!/usr/bin/env python3
"""Test script for Combi-tech engineering services invoice format"""

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


def test_combi_tech_detection_and_fields():
    # Sample lines based on the provided Combi-tech invoice layout
    test_lines = [
        "INVOICE",
        "Invoice Date",
        "29 Dec 2022",
        "Invoice Number",
        "INV-1062",
        "Reference",
        "fryer repairs",
        "Combi-tech engineering services",
        "Attention: Paul Lessitter",
        "39b Station Road",
        "Thorney",
        "PETERBOROUGH",
        "PE6 0QE",
        "UNITED KINGDOM",
        "VAT registration number",
        ": 388968701",
        "Subtotal",
        "135.38",
        "TOTAL VAT 20%",
        "27.08",
        "TOTAL GBP",
        "162.46",
    ]

    cleaned = [main._clean_text(ln) for ln in test_lines]
    cleaned = [ln for ln in cleaned if ln]

    parsed = main._extract_invoice_fields(cleaned)

    # document_date
    if (main._clean_text(parsed.get("document_date")) or "").strip() != "29/12/2022":
        print(f"✗ document_date mismatch: {parsed.get('document_date')}")
        return False

    # inv_ref_no
    if (main._clean_text(parsed.get("inv_ref_no")) or "").strip() != "INV-1062":
        print(f"✗ inv_ref_no mismatch: {parsed.get('inv_ref_no')}")
        return False

    # make (reference)
    if (main._clean_text(parsed.get("make")) or "").lower() != "fryer repairs":
        print(f"✗ make mismatch: {parsed.get('make')}")
        return False

    # totals
    if parsed.get("buying_price") != 162.46 or parsed.get("non_vat") != 162.46:
        print(f"✗ total mismatch: buying_price={parsed.get('buying_price')} non_vat={parsed.get('non_vat')}")
        return False

    if parsed.get("vat_amount") != 27.08:
        print(f"✗ vat_amount mismatch: {parsed.get('vat_amount')}")
        return False

    # std_net from subtotal
    if parsed.get("std_net") != 135.38:
        print(f"✗ std_net mismatch: {parsed.get('std_net')}")
        return False

    print("✓ Combi-tech detection and field extraction passed")
    return True


def main_test():
    print("Testing Combi-tech engineering services invoice format implementation...")
    print("=" * 65)

    ok = test_combi_tech_detection_and_fields()

    print("\n" + "=" * 65)
    if ok:
        print("🎉 All tests passed! Combi-tech format is ready.")
        return True

    print("❌ Test failed. Please check the implementation.")
    return False


if __name__ == "__main__":
    success = main_test()
    sys.exit(0 if success else 1)
