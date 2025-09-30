#!/usr/bin/env python3
"""
Test script for empty response handling.

This script tests that the system properly handles cases where the LLM
returns an empty response, providing clear error messages to the user.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unittest.mock import Mock, patch
from src.utils_v1_1 import invoke_ollama


def test_empty_response_handling():
    """Test that empty LLM responses are properly caught and reported."""
    
    print("=== Testing Empty Response Handling ===\n")
    
    # Test 1: Empty message content
    print("Test 1: Empty message content")
    with patch('src.utils_v1_1.chat') as mock_chat:
        # Simulate empty response
        mock_response = Mock()
        mock_response.message.content = ""
        mock_chat.return_value = mock_response
        
        try:
            result = invoke_ollama(
                model="test-model",
                system_prompt="Test system prompt",
                user_prompt="Test user prompt"
            )
            print("❌ FAILED: Should have raised ValueError for empty response")
        except ValueError as e:
            if "returned an empty response" in str(e):
                print("✅ PASSED: Empty response correctly caught")
                print(f"   Error message: {str(e)[:100]}...")
            else:
                print(f"❌ FAILED: Wrong error message: {str(e)}")
        except Exception as e:
            print(f"❌ FAILED: Unexpected exception: {str(e)}")
    print()
    
    # Test 2: None message
    print("Test 2: None message")
    with patch('src.utils_v1_1.chat') as mock_chat:
        # Simulate None message
        mock_response = Mock()
        mock_response.message = None
        mock_chat.return_value = mock_response
        
        try:
            result = invoke_ollama(
                model="test-model",
                system_prompt="Test system prompt",
                user_prompt="Test user prompt"
            )
            print("❌ FAILED: Should have raised ValueError for None message")
        except ValueError as e:
            if "returned an empty response" in str(e):
                print("✅ PASSED: None message correctly caught")
            else:
                print(f"❌ FAILED: Wrong error message: {str(e)}")
        except Exception as e:
            print(f"❌ FAILED: Unexpected exception: {str(e)}")
    print()
    
    # Test 3: Whitespace-only content
    print("Test 3: Whitespace-only content")
    with patch('src.utils_v1_1.chat') as mock_chat:
        # Simulate whitespace-only response
        mock_response = Mock()
        mock_response.message.content = "   \n  \t  "
        mock_chat.return_value = mock_response
        
        try:
            result = invoke_ollama(
                model="test-model",
                system_prompt="Test system prompt",
                user_prompt="Test user prompt"
            )
            print("❌ FAILED: Should have raised ValueError for whitespace-only response")
        except ValueError as e:
            if "returned an empty response" in str(e):
                print("✅ PASSED: Whitespace-only response correctly caught")
            else:
                print(f"❌ FAILED: Wrong error message: {str(e)}")
        except Exception as e:
            print(f"❌ FAILED: Unexpected exception: {str(e)}")
    print()
    
    # Test 4: Valid response
    print("Test 4: Valid response")
    with patch('src.utils_v1_1.chat') as mock_chat:
        # Simulate valid response
        mock_response = Mock()
        mock_response.message.content = "This is a valid response"
        mock_chat.return_value = mock_response
        
        try:
            result = invoke_ollama(
                model="test-model",
                system_prompt="Test system prompt",
                user_prompt="Test user prompt"
            )
            if result == "This is a valid response":
                print("✅ PASSED: Valid response correctly returned")
            else:
                print(f"❌ FAILED: Got unexpected result: {result}")
        except Exception as e:
            print(f"❌ FAILED: Should not have raised exception: {str(e)}")
    print()
    
    print("=== Test Complete ===")


if __name__ == "__main__":
    test_empty_response_handling()
