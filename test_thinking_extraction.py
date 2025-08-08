#!/usr/bin/env python3
"""
Test script for the extract_thinking_and_final_answer function
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'apps'))

def extract_thinking_and_final_answer(text):
    """
    Extract thinking blocks and final answer from LLM response.
    
    Handles various thinking tag formats:
    - <think>...</think>
    - </think>...</think> (malformed opening)
    - <think>...<think> (malformed closing)
    
    Returns:
        tuple: (thinking_blocks: list[str], final_answer: str)
    """
    import re
    
    if not text or not text.strip():
        return [], ""
    
    # Find all thinking blocks with various tag formats
    thinking_blocks = []
    
    # Pattern 1: Proper <think>...</think> tags
    proper_pattern = r'<think>(.*?)</think>'
    proper_matches = re.findall(proper_pattern, text, re.DOTALL | re.IGNORECASE)
    thinking_blocks.extend(proper_matches)
    
    # Pattern 2: Malformed opening </think>...</think>
    malformed_open_pattern = r'</think>(.*?)</think>'
    malformed_open_matches = re.findall(malformed_open_pattern, text, re.DOTALL | re.IGNORECASE)
    thinking_blocks.extend(malformed_open_matches)
    
    # Pattern 3: Malformed closing <think>...<think>
    malformed_close_pattern = r'<think>(.*?)<think>'
    malformed_close_matches = re.findall(malformed_close_pattern, text, re.DOTALL | re.IGNORECASE)
    thinking_blocks.extend(malformed_close_matches)
    
    # Find the position of the last thinking tag (any variation)
    last_think_pos = -1
    
    # Find all thinking tag positions
    all_patterns = [
        r'<think>.*?</think>',  # Proper tags
        r'</think>.*?</think>',  # Malformed opening
        r'<think>.*?<think>',   # Malformed closing
    ]
    
    for pattern in all_patterns:
        matches = list(re.finditer(pattern, text, re.DOTALL | re.IGNORECASE))
        for match in matches:
            last_think_pos = max(last_think_pos, match.end())
    
    # Extract final answer (content after the last thinking tag)
    if last_think_pos > -1:
        final_answer = text[last_think_pos:].strip()
    else:
        # No thinking tags found, entire text is the final answer
        final_answer = text.strip()
    
    # Clean up thinking blocks (remove extra whitespace)
    thinking_blocks = [block.strip() for block in thinking_blocks if block.strip()]
    
    return thinking_blocks, final_answer

def test_thinking_extraction():
    """Test various thinking tag scenarios"""
    
    # Test case 1: User's example - thinking process only
    test1 = """Okay, I need to figure out the detected language here. The user provided a bunch of text in German, and the task is to respond in the same language. The current position is set to 'detect_language', so I should confirm that the language is German. Let me check the text again. There are terms like 'Strahlenschutzgesetz', 'Bq/m³', and references to German sources. Yep, definitely German. So the response should just be the JSON with the detected language as German and current position as detect_language. No need for any other content."""
    
    thinking1, answer1 = extract_thinking_and_final_answer(test1)
    print("Test 1 - No thinking tags:")
    print(f"Thinking blocks: {len(thinking1)}")
    print(f"Final answer: '{answer1[:100]}...'")
    print()
    
    # Test case 2: Proper thinking tags
    test2 = """<think>I need to analyze this German text about radiation protection laws. The user is asking about specific regulations.</think>
    
Based on the German Strahlenschutzgesetz (Radiation Protection Act), the following regulations apply:

1. Measurement requirements for radon levels
2. Compliance with EU directives
3. Regular monitoring protocols"""
    
    thinking2, answer2 = extract_thinking_and_final_answer(test2)
    print("Test 2 - Proper thinking tags:")
    print(f"Thinking blocks: {len(thinking2)}")
    print(f"Thinking content: '{thinking2[0][:50]}...' " if thinking2 else "None")
    print(f"Final answer: '{answer2[:50]}...'")
    print()
    
    # Test case 3: Malformed opening tags
    test3 = """</think>Let me think about this radiation protection question. I need to check the specific requirements.</think>
    
The German radiation protection regulations require:
- Regular radon measurements
- Documentation of results
- Compliance reporting"""
    
    thinking3, answer3 = extract_thinking_and_final_answer(test3)
    print("Test 3 - Malformed opening tags:")
    print(f"Thinking blocks: {len(thinking3)}")
    print(f"Thinking content: '{thinking3[0][:50]}...' " if thinking3 else "None")
    print(f"Final answer: '{answer3[:50]}...'")
    print()
    
    # Test case 4: Malformed closing tags
    test4 = """<think>This is about German radiation protection laws. I should provide accurate information about the Strahlenschutzgesetz.<think>
    
According to the German Strahlenschutzgesetz:
- Radon levels must not exceed 300 Bq/m³
- Regular monitoring is mandatory
- Proper documentation is required"""
    
    thinking4, answer4 = extract_thinking_and_final_answer(test4)
    print("Test 4 - Malformed closing tags:")
    print(f"Thinking blocks: {len(thinking4)}")
    print(f"Thinking content: '{thinking4[0][:50]}...' " if thinking4 else "None")
    print(f"Final answer: '{answer4[:50]}...'")
    print()
    
    # Test case 5: Multiple thinking blocks
    test5 = """<think>First, I need to understand what the user is asking about.</think>
    
Some initial information here.
    
<think>Now I need to provide more detailed analysis of the German regulations.</think>
    
The comprehensive answer about German radiation protection:
- Legal framework
- Implementation guidelines
- Monitoring requirements"""
    
    thinking5, answer5 = extract_thinking_and_final_answer(test5)
    print("Test 5 - Multiple thinking blocks:")
    print(f"Thinking blocks: {len(thinking5)}")
    for i, block in enumerate(thinking5):
        print(f"  Block {i+1}: '{block[:30]}...'")
    print(f"Final answer: '{answer5[:50]}...'")
    print()

if __name__ == "__main__":
    test_thinking_extraction()
