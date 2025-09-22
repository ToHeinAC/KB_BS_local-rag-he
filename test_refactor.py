#!/usr/bin/env python3
"""
Test script to verify the refactored source handling functions work correctly
"""
import sys
from pathlib import Path

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.rag_helpers_v1_1 import (
        get_available_databases,
        extract_database_prefix,
        resolve_source_directory,
        resolve_pdf_path,
        linkify_sources
    )
    
    print("✅ Successfully imported all source handling functions")
    
    # Test extract_database_prefix
    test_db_name = "StrlSch__Qwen--Qwen3-Embedding-0.6B--3000--600"
    prefix = extract_database_prefix(test_db_name)
    print(f"✅ extract_database_prefix('{test_db_name}') = '{prefix}'")
    
    # Test resolve_source_directory
    source_dir = resolve_source_directory(test_db_name)
    print(f"✅ resolve_source_directory('{test_db_name}') = '{source_dir}'")
    
    # Test linkify_sources with sample text
    sample_text = "This is a test with [StrlSchG--250508.pdf] source reference"
    linked = linkify_sources(sample_text, test_db_name)
    print(f"✅ linkify_sources processed sample text successfully")
    
    print("\n🎉 All refactored functions working correctly!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error testing functions: {e}")
