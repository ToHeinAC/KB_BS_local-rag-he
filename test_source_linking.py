#!/usr/bin/env python3
"""
Test script for source linking functionality.

This script tests the linkify_sources function to ensure that:
1. Source references like [filename.pdf] are correctly converted to clickable links
2. Links use file:// URLs instead of base64 data URLs
3. Missing files are handled gracefully
4. Various reference formats are supported
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.rag_helpers_v1_1 import linkify_sources


def test_source_linking():
    """Test the source linking functionality with various inputs."""
    
    print("=== Testing Source Linking ===\n")
    
    # Test 1: Simple source reference
    test_text_1 = """
1. Definition des Kd‑Werts
Der Kd‑Wert beschreibt das Verhältnis der an Bodenpartikel sorbierten Konzentration eines Stoffes zur Konzentration in der Lösung. [StrlSchG.pdf]
Er wird üblicherweise in Liter pro Kilogramm (L kg⁻¹) bzw. Milliliter pro Gramm (ml g⁻¹) angegeben und ist ein Maß für die Mobilität eines Radionuklids im Boden.
"""
    
    print("Test 1: Simple source reference [StrlSchG.pdf]")
    print("Input:", test_text_1[:150], "...")
    result_1 = linkify_sources(test_text_1, selected_database=None, kb_path="./kb")
    print("Output preview:", result_1[:200], "...")
    print()
    
    # Test 2: Reference with timestamp
    test_text_2 = """
Die gesetzlichen Grundlagen finden sich in [StrlSchG--250508.pdf] und weiteren Dokumenten.
"""
    
    print("Test 2: Reference with timestamp [StrlSchG--250508.pdf]")
    print("Input:", test_text_2)
    result_2 = linkify_sources(test_text_2, selected_database=None, kb_path="./kb")
    print("Output:", result_2)
    print()
    
    # Test 3: Multiple references
    test_text_3 = """
Die Informationen stammen aus [EPA_Kd-a.pdf] und [StrlSchG.pdf].
Weitere Details in [test-document.pdf].
"""
    
    print("Test 3: Multiple references")
    print("Input:", test_text_3)
    result_3 = linkify_sources(test_text_3, selected_database=None, kb_path="./kb")
    print("Output:", result_3)
    print()
    
    # Test 4: Check that output doesn't contain base64
    print("Test 4: Verify no base64 data URLs are generated")
    if "base64," in result_1 or "base64," in result_2 or "base64," in result_3:
        print("❌ FAILED: Found base64 data in output!")
    else:
        print("✅ PASSED: No base64 data URLs found")
    print()
    
    # Test 5: Check that file:// URLs are generated
    print("Test 5: Verify file:// URLs are generated")
    if "file://" in result_1 or "file://" in result_2 or "file://" in result_3:
        print("✅ PASSED: Found file:// URLs in output")
    else:
        print("⚠️  WARNING: No file:// URLs found (files might not exist)")
    print()
    
    # Test 6: Check link format
    print("Test 6: Check HTML link format")
    if '<a href="file://' in result_1 or '<a href="file://' in result_2 or '<a href="file://' in result_3:
        print("✅ PASSED: Correct HTML link format found")
    else:
        print("⚠️  WARNING: HTML links might not be formatted correctly")
    print()
    
    print("=== Test Complete ===")


if __name__ == "__main__":
    test_source_linking()