#!/usr/bin/env python3
"""
Diagnostic script for source linking functionality.

This script checks:
1. If linkify_sources is using file:// URLs (not base64)
2. If PDF files can be found
3. If the links are properly formatted
4. If there are any issues with the implementation
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.rag_helpers_v1_1 import linkify_sources, resolve_source_directory


def main():
    print("=" * 70)
    print("SOURCE LINKING DIAGNOSTIC TOOL")
    print("=" * 70)
    print()
    
    # Test 1: Check basic source reference conversion
    print("TEST 1: Basic Source Reference Conversion")
    print("-" * 70)
    test_text = "See [StrlSchG.pdf] for details."
    result = linkify_sources(test_text, selected_database=None)
    
    print(f"Input:  {test_text}")
    print(f"Output: {result}")
    print()
    
    if "base64," in result:
        print("✅ PASS: Using base64 data URLs (required for browser compatibility)")
        print("   Note: file:// URLs don't work in browsers due to security restrictions")
    elif "file://" in result:
        print("⚠️  WARN: Using file:// URLs (may not work in browsers)")
    else:
        print("❌ FAIL: No valid URL scheme found")
    
    if "<a href=" in result:
        print("✅ PASS: Generating HTML links")
    else:
        print("❌ FAIL: No HTML links generated")
    
    print()
    
    # Test 2: Check if links have proper attributes
    print("TEST 2: Link Attributes")
    print("-" * 70)
    
    if 'onclick="window.open' in result:
        print("✅ PASS: Links use JavaScript window.open (opens in new window)")
    elif 'target="_blank"' in result:
        print("✅ PASS: Links open in new tab (target='_blank')")
    else:
        print("❌ FAIL: Links may not open in new window/tab")
    
    if 'text-decoration: underline' in result:
        print("✅ PASS: Links have underline styling")
    else:
        print("⚠️  WARN: Links may not have underline styling")
    
    if '📄' in result:
        print("✅ PASS: Using document emoji icon")
    else:
        print("⚠️  WARN: No document icon")
    
    print()
    
    # Test 3: Check KB directory structure
    print("TEST 3: Knowledge Base Structure")
    print("-" * 70)
    kb_path = Path("./kb")
    
    if kb_path.exists():
        print(f"✅ PASS: KB directory exists at {kb_path.resolve()}")
        
        # List subdirectories
        subdirs = [d for d in kb_path.iterdir() if d.is_dir()]
        print(f"   Found {len(subdirs)} subdirectories:")
        for subdir in subdirs[:5]:  # Show first 5
            pdf_count = len(list(subdir.glob("*.pdf")))
            print(f"   - {subdir.name} ({pdf_count} PDFs)")
        if len(subdirs) > 5:
            print(f"   ... and {len(subdirs) - 5} more")
    else:
        print(f"❌ FAIL: KB directory not found at {kb_path.resolve()}")
    
    print()
    
    # Test 4: Test with a database selection
    print("TEST 4: Database-Specific Source Resolution")
    print("-" * 70)
    
    # Try with StrlSch database
    test_db = "StrlSch__Qwen--Qwen3-Embedding-0.6B--3000--600"
    test_text_db = "According to [StrlSchG--250508.pdf], the regulations state..."
    
    try:
        source_dir = resolve_source_directory(test_db)
        print(f"Database: {test_db}")
        print(f"Resolved source directory: {source_dir}")
        print(f"Directory exists: {source_dir.exists()}")
        
        if source_dir.exists():
            pdf_files = list(source_dir.glob("*.pdf"))
            print(f"PDF files found: {len(pdf_files)}")
            if pdf_files:
                print("Sample PDFs:")
                for pdf in pdf_files[:3]:
                    print(f"   - {pdf.name}")
        
        result_db = linkify_sources(test_text_db, selected_database=test_db)
        print()
        print(f"Input:  {test_text_db}")
        print(f"Output: {result_db[:200]}...")
        print()
        
        if "file://" in result_db:
            print("✅ PASS: Database-specific linking works with file:// URLs")
        else:
            print("⚠️  WARN: Database-specific linking may have issues")
            
    except Exception as e:
        print(f"⚠️  WARN: Error testing database-specific resolution: {str(e)}")
    
    print()
    
    # Test 5: Verify implementation
    print("TEST 5: Implementation Check")
    print("-" * 70)
    
    try:
        import inspect
        source_code = inspect.getsource(linkify_sources)
        
        if "base64.b64encode" in source_code:
            print("✅ PASS: Found base64.b64encode in linkify_sources")
            print("   (This is correct - base64 is needed for browser compatibility)")
        else:
            print("⚠️  WARN: No base64.b64encode found - links may not work in browsers")
        
        if "window.open" in source_code:
            print("✅ PASS: Found window.open JavaScript for new window/tab")
        else:
            print("⚠️  WARN: No window.open found in code")
            
    except Exception as e:
        print(f"⚠️  WARN: Could not inspect source code: {str(e)}")
    
    print()
    
    # Final summary
    print("=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)
    print()
    print("To test source linking interactively, run:")
    print("   make test-sources")
    print("   # or")
    print("   uv run streamlit run dev/basic_report-source-tester_app.py")
    print()
    print("This will open a browser where you can click on source links")
    print("and verify they open PDFs correctly.")
    print()


if __name__ == "__main__":
    main()
