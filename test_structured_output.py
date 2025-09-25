#!/usr/bin/env python3
"""
Test script to verify structured output functionality for final report generation.
This script tests both the structured output approach and the fallback JSON parsing.
"""

import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.utils_v1_1 import invoke_ollama
from pydantic import BaseModel
import json

# Pydantic model for structured final report output
class FinalReportOutput(BaseModel):
    """Structured output model for final report generation."""
    content: str  # The final report content in markdown format

def test_structured_output():
    """Test the structured output functionality."""
    print("🧪 Testing Structured Output for Final Report Generation")
    print("=" * 60)
    
    # Test prompt
    test_prompt = """
    Based on the following information, generate a brief research report:
    
    Topic: Renewable Energy
    Key Points:
    - Solar energy is becoming more cost-effective
    - Wind power capacity is growing globally
    - Battery storage technology is improving
    
    Please provide a concise report in markdown format.
    """
    
    # Test 1: Try structured output with Pydantic model
    print("\n📋 Test 1: Structured Output with Pydantic Model")
    print("-" * 50)
    
    try:
        print("Attempting structured output...")
        structured_result = invoke_ollama(
            system_prompt="You are an expert research analyst. Your response must be a JSON object with a single 'content' key containing the markdown-formatted report.",
            user_prompt=test_prompt,
            model="qwen3:1.7b",  # Use a lighter model for testing
            output_format=FinalReportOutput
        )
        
        print("✅ Structured output successful!")
        print(f"Type: {type(structured_result)}")
        print(f"Content type: {type(structured_result.content)}")
        print(f"Content length: {len(structured_result.content)} characters")
        print(f"Content preview: {structured_result.content[:200]}...")
        
        return True, structured_result.content
        
    except Exception as e:
        print(f"❌ Structured output failed: {str(e)}")
        return False, None

def test_fallback_json_parsing():
    """Test the fallback JSON parsing functionality."""
    print("\n📋 Test 2: Fallback JSON Parsing")
    print("-" * 50)
    
    try:
        print("Testing fallback JSON parsing...")
        
        json_system_prompt = """You are an expert research analyst.

IMPORTANT: Your response MUST be a valid JSON object with exactly this structure:
{
    "thinking": "your reasoning process here",
    "final": "your markdown-formatted report here"
}

Do not include any text outside the JSON object. The 'thinking' field should contain your reasoning process, and the 'final' field should contain the complete report in markdown format."""
        
        test_prompt = """
        Based on the following information, generate a brief research report:
        
        Topic: Renewable Energy
        Key Points:
        - Solar energy is becoming more cost-effective
        - Wind power capacity is growing globally
        - Battery storage technology is improving
        
        Please provide a concise report in markdown format.
        """
        
        raw_response = invoke_ollama(
            system_prompt=json_system_prompt,
            user_prompt=test_prompt,
            model="qwen3:1.7b"
        )
        
        print(f"Raw response length: {len(raw_response)} characters")
        print(f"Raw response preview: {raw_response[:200]}...")
        
        # Parse JSON manually
        try:
            # Clean the response - remove any markdown code blocks if present
            cleaned_response = raw_response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            print(f"Cleaned response: {cleaned_response[:200]}...")
            
            parsed_json = json.loads(cleaned_response)
            thinking_content = parsed_json.get('thinking', '')
            final_content = parsed_json.get('final', raw_response)
            
            print("✅ JSON parsing successful!")
            print(f"Thinking content type: {type(thinking_content)}")
            print(f"Thinking content length: {len(thinking_content)} characters")
            print(f"Thinking content preview: {thinking_content[:200]}...")
            print(f"Final content type: {type(final_content)}")
            print(f"Final content length: {len(final_content)} characters")
            print(f"Final content preview: {final_content[:200]}...")
            
            return True, (final_content, thinking_content)
            
        except json.JSONDecodeError as json_error:
            print(f"❌ JSON parsing failed: {str(json_error)}")
            print("Using raw response as fallback...")
            return False, (raw_response, None)
            
    except Exception as e:
        print(f"❌ Fallback test failed: {str(e)}")
        return False, (None, None)

def test_parse_structured_output():
    """Test the parse_structured_llm_output function with the specific JSON structure."""
    print("\n📋 Test 3: parse_structured_llm_output Function")
    print("-" * 50)
    
    # Import the function from the app
    try:
        from apps.app_v2_0 import parse_structured_llm_output
        
        # Test with the exact JSON structure you provided
        test_json = '''{
            "thinking": "I reviewed the highest‑ranked primary summary (score 8.9) and all supporting summaries. The user asks for a precise definition of the Kd‑Wert and a detailed depiction of its calculation and interpretation for Ra‑226, including experimental values for different soils, influencing factors, applications in transport models, exposure assessment, and regulatory guidance.",
            "final": "# Kd‑Wert (Distributionskoeffizient) – Definition und Bedeutung für Radium‑226\n\n## 1. Definition\nDer Kd‑Wert beschreibt das Gleichgewicht zwischen der Konzentration eines Stoffes in der festen Phase (Boden, Sediment) und seiner Konzentration in der flüssigen Phase (Wasser). Er wird als K_{d}=\\frac{C_{s}}{C_{w}} ausgedrückt.\n\n## 2. Berechnung im Labor\nIn Batch‑Expositions‑Methoden wird ein Boden‑Wasser‑Gemisch mit einem definierten S/L‑Verhältnis inkubiert, bis ein Gleichgewicht erreicht ist."
        }'''
        
        print("Testing with structured JSON input...")
        final_content, thinking_content = parse_structured_llm_output(test_json)
        
        print("✅ Parsing successful!")
        print(f"Thinking content found: {thinking_content is not None}")
        if thinking_content:
            print(f"Thinking length: {len(thinking_content)} characters")
            print(f"Thinking preview: {thinking_content[:100]}...")
        
        print(f"Final content found: {final_content is not None}")
        if final_content:
            print(f"Final content length: {len(final_content)} characters")
            print(f"Final content preview: {final_content[:100]}...")
        
        return True, (final_content, thinking_content)
        
    except Exception as e:
        print(f"❌ parse_structured_llm_output test failed: {str(e)}")
        return False, (None, None)

def main():
    """Run all tests."""
    print("🚀 Starting Structured Output Tests")
    print("=" * 60)
    
    # Test structured output
    structured_success, structured_content = test_structured_output()
    
    # Test fallback JSON parsing
    fallback_success, fallback_content = test_fallback_json_parsing()
    
    # Test the parse_structured_llm_output function
    parse_success, parse_content = test_parse_structured_output()
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 60)
    print(f"Structured Output: {'✅ PASS' if structured_success else '❌ FAIL'}")
    print(f"Fallback JSON Parsing: {'✅ PASS' if fallback_success else '❌ FAIL'}")
    print(f"parse_structured_llm_output: {'✅ PASS' if parse_success else '❌ FAIL'}")
    
    if structured_success or fallback_success or parse_success:
        print("\n🎉 At least one method works! The implementation should be reliable.")
    else:
        print("\n⚠️ All methods failed. Please check the implementation.")
    
    return structured_success or fallback_success or parse_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
