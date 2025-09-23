#!/usr/bin/env python3
"""
Test script for the improved structured output parsing functionality.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'apps'))

# Import the parsing function from app_v2_0.py
from app_v2_0 import parse_structured_llm_output

def test_structured_output_parsing():
    """Test various formats of structured LLM output."""
    
    print("Testing structured output parsing...")
    print("=" * 50)
    
    # Test case 1: Dictionary with thinking and final keys
    test_dict = {
        "thinking": "The user requests a comprehensive, scientifically grounded explanation of the Kd‑Wert, its mathematical definition, units, relevance for Ra‑226, typical value ranges, and key influencing factors. The highest‑ranked primary summary contains the most relevant data, so I will base the core answer on that, citing its specific statements.",
        "final": "# Kd‑Wert (Verteilungskoeffizient) – Begriff, mathematische Definition und Bedeutung für ^226Ra\n\n## Einleitung\nDer **Kd‑Wert** (Distribution‑Coefficient) quantifiziert die Adsorption eines Stoffes an feste Partikel im Verhältnis zur Konzentration in der umgebenden Flüssigkeit."
    }
    
    print("Test 1: Dictionary with 'thinking' and 'final' keys")
    final_content, thinking_content = parse_structured_llm_output(test_dict)
    print(f"✅ Thinking extracted: {thinking_content is not None}")
    print(f"✅ Final content extracted: {len(final_content) > 0}")
    print(f"Thinking preview: {thinking_content[:100] if thinking_content else 'None'}...")
    print(f"Final preview: {final_content[:100]}...")
    print()
    
    # Test case 2: JSON string format
    test_json = '{"thinking": "This is the thinking process", "final": "This is the final answer with markdown content"}'
    
    print("Test 2: JSON string format")
    final_content, thinking_content = parse_structured_llm_output(test_json)
    print(f"✅ Thinking extracted: {thinking_content is not None}")
    print(f"✅ Final content extracted: {len(final_content) > 0}")
    print(f"Thinking: {thinking_content}")
    print(f"Final: {final_content}")
    print()
    
    # Test case 3: Python dict string format
    test_dict_str = "{'reasoning': 'This is my reasoning', 'response': 'This is my response'}"
    
    print("Test 3: Python dict string format")
    final_content, thinking_content = parse_structured_llm_output(test_dict_str)
    print(f"✅ Thinking extracted: {thinking_content is not None}")
    print(f"✅ Final content extracted: {len(final_content) > 0}")
    print(f"Thinking: {thinking_content}")
    print(f"Final: {final_content}")
    print()
    
    # Test case 4: Regular string (no structure)
    test_string = "This is just a regular markdown response without any structure."
    
    print("Test 4: Regular string (no structure)")
    final_content, thinking_content = parse_structured_llm_output(test_string)
    print(f"✅ No thinking extracted: {thinking_content is None}")
    print(f"✅ Final content is original: {final_content == test_string}")
    print(f"Final: {final_content}")
    print()
    
    # Test case 5: Dictionary with alternative keys
    test_alt_dict = {
        "thought": "Alternative thinking key",
        "answer": "Alternative answer key"
    }
    
    print("Test 5: Dictionary with alternative keys")
    final_content, thinking_content = parse_structured_llm_output(test_alt_dict)
    print(f"✅ Thinking extracted: {thinking_content is not None}")
    print(f"✅ Final content extracted: {len(final_content) > 0}")
    print(f"Thinking: {thinking_content}")
    print(f"Final: {final_content}")
    print()
    
    # Test case 6: Content with <think> blocks
    test_with_think = "This is content <think>with thinking blocks</think> that should be cleaned."
    
    print("Test 6: Content with <think> blocks")
    final_content, thinking_content = parse_structured_llm_output(test_with_think)
    print(f"✅ <think> blocks removed: {'<think>' not in final_content}")
    print(f"Final: '{final_content}'")
    print(f"Original: '{test_with_think}'")
    print()
    
    print("All tests completed! ✅")

if __name__ == "__main__":
    test_structured_output_parsing()
