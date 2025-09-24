import streamlit as st
import sys
import os
import json
import re
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
from typing_extensions import TypedDict
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import required modules
from src.rag_helpers_v1_1 import get_report_llm_models, clear_cuda_memory
from src.utils_v1_1 import invoke_ollama, parse_output
from src.state_v2_0 import ResearcherStateV2
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from langgraph.types import RunnableConfig
from src.prompts_v1_1 import (
    LLM_QUALITY_CHECKER_SYSTEM_PROMPT, 
    LLM_QUALITY_CHECKER_HUMAN_PROMPT,
    SUMMARY_IMPROVEMENT_SYSTEM_PROMPT,
    SUMMARY_IMPROVEMENT_HUMAN_PROMPT
)

# Pydantic model for structured final report output
class FinalReportOutput(BaseModel):
    """Structured output model for final report generation."""
    thinking: str  # The LLM's thinking process
    final: str     # The final report content in markdown format

# Page configuration
st.set_page_config(
    page_title="Basic Rerank & Reporter",
    page_icon="📊",
    layout="wide"
)

def extract_thinking_and_final_answer(text: str) -> tuple[str, str]:
    """
    Extract thinking blocks and final answer from LLM response.
    
    Handles various thinking tag formats:
    - <think>...</think>
    - </think>...</think> (malformed opening)
    - <think>...<think> (malformed closing)
    
    Args:
        text: Raw LLM response string
        
    Returns:
        tuple: (thinking_content: str, final_answer: str)
    """
    if not text or not text.strip():
        return "", ""
    
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
    
    # Combine all thinking blocks into a single string
    thinking_content = "\n\n".join([block.strip() for block in thinking_blocks if block.strip()])
    
    return thinking_content, final_answer

def clean_llm_response(response: str) -> str:
    """
    Clean LLM response by removing <think>...</think> blocks and extra whitespace.
    
    Args:
        response: Raw LLM response string
        
    Returns:
        str: Cleaned response with thinking blocks removed
    """
    if not response:
        return ""
    
    # Remove <think>...</think> blocks (handle both proper and malformed tags)
    # Pattern handles: <think>...</think>, </think>...</think>, <think>...<think>
    cleaned = re.sub(r'<think>.*?(?:</think>|<think>)', '', response, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove extra whitespace and newlines
    cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned)  # Replace multiple newlines with double newlines
    cleaned = cleaned.strip()
    
    return cleaned

def score_summary(initial_query: str, query: str, content: str, context: str, 
                  llm_model: str, language: str = "English") -> float:
    """
    Ask the LLM to score a single summary 0–10.
    """
    prompt = f"""
You are an expert evaluator of document summary relevance.

TASK: Make a deep analysis of the following summary and score it for its relevance and accuracy regarding the original query and the given context.
Ignore any scoring that is already within the SUMMARY TO ASSESS section.

ORIGINAL USER QUERY:
{initial_query}

SPECIFIC RESEARCH QUERY:
{query}

ADDITIONAL CONTEXT:
{context}

SUMMARY TO ASSESS:
{content}

SCORING CRITERIA (weights in parentheses):
1. Direct relevance to the original user query (50%) – Penalize mismatches in key terms (deduct 1-2 points if exact terms are not used, e.g., 'transportation' instead of 'local public transport').
2. Specificity and level of detail (20%) – Reward precise matches to query intent.
3. Alignment with the research query context (20%) – Ensure partial alignment doesn't inflate scores.
4. Factual accuracy and completeness (10%) – Deduct for inaccuracies or unrelated additions.
Example: "Score 10 only for perfect alignment; 7-8 for partial matches with some deviations; 5-6 for broad relevance without specifics."

INSTRUCTIONS:
Return ONLY a number between 0 and 10 using the following ranges:
- 10 = perfectly relevant and accurate
- 9-8 = very relevant with strong detail
- 7-6 = relevant but somewhat incomplete
- 5-4 = partially relevant
- 3-0 = poorly relevant or inaccurate
Be strictly critical: Reserve 10 for exact matches to the original query. 
Differentiate by prioritizing exact terms e.g. 'local public transport' from the user query and additional context over broader terms, e.g. 'transportation'. 
Compute sub-scores for each criterion, weight them, and average to final score. Output ONLY the final number (0-10, decimals allowed).
That is: Sub-score each criterion (0-10), multiply by weight, sum, and divide by 100 for final score. E.g., (Criterion1 * 0.5) + (Criterion2 * 0.2) + ...

One-Shot Example:
1. Perfect match to 'local public transport' matters: Score 10.
2. Partial relevance with public transport in general but not direct query: Score 7-8.
3. Discusses general transportation: Score 5-6 max.

Respond in {language}.
"""
    
    try:
        response = invoke_ollama(
            system_prompt="""You are a strict expert evaluator. 
            Provide only a single numerical score (0-10, with decimals) based on critical analysis; no explanations.""",
            user_prompt=prompt,
            model=llm_model
        )
        
        print(f"  [DEBUG] LLM scoring response: {response[:100]}...")
        
        # Extract numerical score from response
        match = re.search(r'\b(\d{1,2}(?:\.\d{1,2})?)\b', response)
        if match:
            score = float(match.group(1))
            # Ensure score is within valid range
            final_score = max(0.0, min(10.0, score))
            print(f"  [DEBUG] Extracted score: {final_score}")
            return final_score
        else:
            print(f"  [WARNING] Could not extract score from LLM response: {response[:100]}...")
            return 5.0  # Default to middle score instead of 1.0
    except Exception as e:
        print(f"  [ERROR] Failed to score summary: {str(e)}")
        return 5.0  # Default to middle score instead of 1.0


def reranker_node(state: ResearcherStateV2, config: RunnableConfig) -> ResearcherStateV2:
    """
    LangGraph node that reranks summaries based on relevance & accuracy.
    
    Args:
        state: ResearcherStateV2 containing search_summaries and other state
        config: RunnableConfig for the graph
    
    Returns:
        Updated ResearcherStateV2 with reranked summaries
    """
    print("--- Starting reranker node ---")
    
    # Extract state variables
    user_query = state.get("user_query", "")
    search_summaries = state.get("search_summaries", {})
    additional_context = state.get("additional_context", "")
    report_llm = state.get("report_llm", "gpt-oss:20b")
    detected_language = state.get("detected_language", "English")
    
    print(f"  [DEBUG] User query: {user_query}")
    print(f"  [DEBUG] Search summaries keys: {list(search_summaries.keys())}")
    print(f"  [DEBUG] Report LLM: {report_llm}")
    print(f"  [DEBUG] Language: {detected_language}")
    
    if not search_summaries:
        print("  [WARNING] No search summaries found for reranking")
        return state
    
    # Rerank summaries for each research query
    all_reranked_summaries = []
    
    for query, summaries in search_summaries.items():
        print(f"  [DEBUG] Processing query: {query} with {len(summaries)} summaries")
        
        if not summaries:
            print(f"  [WARNING] No summaries for query: {query}")
            continue
            
        # Rerank summaries for this query
        reranked = []
        for idx, doc in enumerate(summaries):
            content = doc.page_content
            print(f"  [DEBUG] Scoring summary {idx+1}/{len(summaries)} for query: {query[:50]}...")
            print(f"  [DEBUG] Content preview: {content[:100]}...")
            
            score = score_summary(
                initial_query=user_query,
                query=query,
                content=content,
                context=additional_context,
                llm_model=report_llm,
                language=detected_language
            )
            
            print(f"  [DEBUG] Score for summary {idx+1}: {score}")
            
            reranked.append({
                'summary': content,
                'score': score,
                'original_index': idx,
                'document': doc,
                'query': query
            })
        
        # Sort by score descending within this query
        reranked.sort(key=lambda x: x['score'], reverse=True)
        all_reranked_summaries.extend(reranked)
        print(f"  [DEBUG] Added {len(reranked)} reranked summaries for query: {query[:50]}...")
    
    # CRITICAL: Global sort by score after collecting all summaries from all queries
    all_reranked_summaries.sort(key=lambda x: x['score'], reverse=True)
    print(f"  [DEBUG] Applied global sort by score")
    
    # Store reranked summaries in state
    state["all_reranked_summaries"] = all_reranked_summaries
    state["current_position"] = "reranker"
    
    print(f"  [INFO] Reranked {len(all_reranked_summaries)} total summaries")
    print(f"  [DEBUG] Final scores after global sort: {[item['score'] for item in all_reranked_summaries]}")
    return state


def generate_final_answer_prompt(initial_query: str, reranked_summaries: List[Dict], 
                                additional_context: str = "", language: str = "English", 
                                internet_result: str = "") -> str:
    """
    Create a prompt for generating the final answer using reranked summaries.
    """
    
    prompt = f"""
# GOAL
Generate a complete, deep and extensive answer to the user's query using the provided summaries. Prioritize information from higher-ranked summaries.

# CONTEXT from research:
ORIGINAL QUERY:
{initial_query}

ADDITIONAL CONTEXT:
{additional_context}

RANKED SUMMARIES (ordered by relevance):

PRIMARY SOURCE (Highest Relevance - Score: {reranked_summaries[0]['score']:.1f}):
{reranked_summaries[0]['summary']}

SUPPORTING SOURCES:"""

    # Add remaining summaries as supporting sources
    for i, item in enumerate(reranked_summaries[1:], 2):
        prompt += f"""

Source {i} (Score: {item['score']:.1f}):
{item['summary']}"""

    # Add internet results section if available
    if internet_result and internet_result.strip():
        prompt += f"""

INTERNET SOURCES:
{internet_result}

# END OF CONTEXT from research

YOU MUST ONLY respond in {language} language.

Generate a complete, deep, and well-supported answer to the user's query using the provided summaries. Prioritize information from higher-ranked summaries. When available, support your answer with recent internet search results.
"""

    return prompt

REPORT_WRITER_SYSTEM_PROMPT_backup = """
# ROLE
You are an expert analytical assistant. Your task is to deliver comprehensive, precise, and well-cited answers based on provided ranked document summaries and, when available, recent internet search results.

# OBJECTIVE
Generate a complete, deep, and well-supported answer to the user's query using the provided ranked summaries and any supplementary web-based information. Your answer should synthesize key information, with clear prioritization given to higher-ranked summaries.

# GUIDELINES

• **ALWAYS** focus on directly addressing the user's original query.
• You **MUST ONLY** use information from the context that is provided to you explicitely. **DO NOT** use your own knowledge or information.
• **STRICTLY** preserve original wording and literal information from the research whenever possible.
• **ALWAYS** Prioritize and rely primarily on information from the highest-ranked summary.
• **ALWAYS** Supplement details and context using lower-ranked summaries.
• If web search results are provided, extract current, relevant facts and clearly label these as "Internet Sources."

## Citation Requirements
- For each factual claim, statistic, figure, definition, or quoted statement, you **MUST** cite the exact source **immediately** following the information, using:
  - `[docNAME]` for mentioned documents (e.g., `[doc1.pdf]`)
  - `[URL]` for internet results (e.g., `[https://www.example.com/result1](https://www.example.com/result1)`)
- Where applicable, include section, subsection, or paragraph numbers (e.g., "as stated in Section 4.2 [doc2.pdf]").
- For direct quotes, always use quotation marks and cite the source.

## Information Inclusion and Retention
- **DO NOT OMIT** critical facts, definitions, data, or technical specifications from any summary.
- Preserve exact figures, levels, statistics, and terminology from the source material.
- Maintain 100% fidelity to the original content’s meaning—paraphrase only for clarity, not for substance.
- If information needed to fully answer the query is **missing or incomplete**, state this explicitly.
- If internet or supplementary sources contradict the primary (highest-ranked) summary, prioritize the primary source unless there is clear evidence of error.

## Final Report Structure
- Present the main answer clearly and systematically, citing sources as above.
- At the end, include a short section labeled **Internet Sources** where you list and succinctly summarize any current information used from recent web searches, again with exact URL citation.
- End your report with:  
  `Information Fidelity Score: (X/10)`  
  where X is your self-assessment of how completely you preserved and cited all key information (10 = perfect retention, 1 = significant information loss).

# LANGUAGE CONSTRAINT
Respond **ONLY** in the specified target language: {language}
"""

REPORT_WRITER_SYSTEM_PROMPT = """
# ROLE
You are an expert assistant with deep analytical skills providing comprehensive, precise, and well-cited answers based on ranked document summaries and other available information given to you.

# GOAL
Generate a complete, deep and extensive answer to the user's query using the provided summaries. Prioritize information from higher-ranked summaries.

# OUTPUT FORMAT
You MUST provide your response in the following structured format:

1. **thinking**: Your internal reasoning process, analysis, and approach to answering the query
2. **final**: The complete, polished final answer in markdown format

# CONSTRAINTS:
• Focus on directly answering the original query
• Focus your answer mainly on the highest-ranked summary as it is most relevant to the query
• Never use your own intrinsic knowledge to answer the query
• Never make up sources for citations, instead you must cite only sources that are directly referenced in the text
• Preserve original wording and literal information from the research whenever possible
• For each factual claim, statistic, figure, definition, or quoted statement, you **MUST** cite the exact source **immediately** following the information, using:
  - `[docNAME]` for mentioned documents (e.g., `[doc1.pdf]`)
  - `[URL]` for internet results (e.g., `[https://www.example.com/result1]`)
• When referencing specific information, include section or paragraph mentions (e.g. "As stated in Section 3.2 ... [doc1.pdf]" or "According to § 21 the responsible party is... [doc6.pdf]")
• If the information is insufficient to answer parts of the query, state this explicitly
• Include exact levels, figures, numbers, statistics, and quantitative data ONLY from the source material
• Maintain precision by using direct quotes for key definitions and important statements
• Use lower ranked summaries to add context, details, or complementary information
• If internet sources are available, incorporate recent/current information to complement. Then, clearly indicate that the information comes from recent web searches by adding a short separate section called "Internet Sources" and cite the sources URLs in the form [URL].
• If supporting sources or internet sources contradict the primary source, prioritize the primary source unless there's clear evidence of error
• If information is incomplete, acknowledge limitations
• Respond in {language} language ONLY

INFORMATION RETENTION MANDATE:
• You MUST preserve ALL key information from document summaries in your final report
• You MUST maintain 100% fidelity to the original document content
• You MUST NOT omit any critical details, figures, statistics, or technical specifications
• You MUST include a self-assessment fidelity score (1-10) at the end of your report

At the end of your report, include: "Information Fidelity Score: (X/10)" where X is your self-assessment of how completely you preserved all key information (10 = perfect retention, 1 = significant information loss)
"""


def report_writer_node(state: ResearcherStateV2, config: RunnableConfig) -> ResearcherStateV2:
    """
    LangGraph node that generates the final report using reranked summaries.
    
    Args:
        state: ResearcherStateV2 containing reranked summaries and other state
        config: RunnableConfig for the graph
    
    Returns:
        Updated ResearcherStateV2 with final answer
    """
    print("--- Starting report writer node ---")
    
    # Extract state variables
    user_query = state.get("user_query", "")
    all_reranked_summaries = state.get("all_reranked_summaries", [])
    additional_context = state.get("additional_context", "")
    report_llm = state.get("report_llm", "gpt-oss:20b")
    detected_language = state.get("detected_language", "English")
    internet_result = state.get("internet_result", "")
    
    print(f"  [DEBUG] User query: {user_query}")
    print(f"  [DEBUG] Reranked summaries count: {len(all_reranked_summaries)}")
    print(f"  [DEBUG] Report LLM: {report_llm}")
    print(f"  [DEBUG] Language: {detected_language}")
    print(f"  [DEBUG] Internet result available: {bool(internet_result and internet_result.strip())}")
    if internet_result and internet_result.strip():
        print(f"  [DEBUG] Internet result length: {len(internet_result)} characters")
    
    if not all_reranked_summaries:
        print("  [WARNING] No reranked summaries found - cannot generate report")
        state["final_answer"] = "No relevant information found to generate a report."
        state["current_position"] = "report_writer"
        return state
    
    # Show sample of reranked summaries for debugging
    print(f"  [DEBUG] Sample reranked summaries:")
    for i, item in enumerate(all_reranked_summaries[:2]):
        print(f"    Summary {i+1}: Score={item.get('score', 'N/A')}, Query={item.get('query', 'N/A')[:50]}...")
        print(f"    Content preview: {str(item.get('summary', 'N/A'))[:100]}...")
    
    # Generate final report
    print("  [DEBUG] Generating final answer prompt...")
    system_prompt = REPORT_WRITER_SYSTEM_PROMPT.format(language=detected_language)
    human_prompt = generate_final_answer_prompt(
        initial_query=user_query,
        reranked_summaries=all_reranked_summaries,
        additional_context=additional_context,
        language=detected_language,
        internet_result=internet_result
    )
    
    print(f"  [DEBUG] Prompt length: {len(human_prompt)} characters")
    print(f"  [DEBUG] Prompt preview: {human_prompt[:200]}...")
    
    # Try structured output first, fallback to manual JSON parsing and thinking extraction
    thinking_content = ""
    final_answer = ""
    
    try:
        print(f"  [INFO] Attempting structured output with {report_llm}")
        
        # Try using structured output with Pydantic model
        structured_result = invoke_ollama(
            system_prompt=system_prompt,
            user_prompt=human_prompt,
            model=report_llm,
            output_format=FinalReportOutput
        )
        
        # Extract thinking and final content from structured output
        thinking_content = structured_result.thinking or ""
        final_answer = structured_result.final or ""
        print(f"  [INFO] Structured output successful - thinking: {len(thinking_content)} chars, final: {len(final_answer)} chars")
        
    except Exception as e:
        print(f"  [WARNING] Structured output failed: {str(e)}. Falling back to manual JSON parsing.")
        
        # Fallback: Use strict JSON instructions in system prompt
        json_system_prompt = system_prompt + f""" \n\n
IMPORTANT: Your response MUST be a valid JSON object with exactly this structure:
        {{
            "thinking": "your internal reasoning and analysis process here",
            "final": "your complete markdown-formatted report here"
        }}
        
Do not include any text outside the JSON object. The 'thinking' field should contain your reasoning process, and the 'final' field should contain the complete report in markdown format."""
        
        raw_response = invoke_ollama(
            system_prompt=json_system_prompt,
            user_prompt=human_prompt,
            model=report_llm
        )
        
        # Parse JSON manually
        try:
            # Check if response is empty
            if not raw_response or not raw_response.strip():
                print(f"  [ERROR] Empty response from LLM model {report_llm}")
                thinking_content = ""
                final_answer = f"Error: The LLM model {report_llm} returned an empty response. This may indicate the model is not working properly or the prompt is too complex. Please try a different model or simplify the query."
            else:
                # Clean the response - remove any markdown code blocks if present
                cleaned_response = raw_response.strip()
                if cleaned_response.startswith('```json'):
                    cleaned_response = cleaned_response[7:]
                if cleaned_response.endswith('```'):
                    cleaned_response = cleaned_response[:-3]
                cleaned_response = cleaned_response.strip()
                
                parsed_json = json.loads(cleaned_response)
                thinking_content = parsed_json.get('thinking', '')
                final_answer = parsed_json.get('final', raw_response)
                print(f"  [INFO] Manual JSON parsing successful - thinking: {len(thinking_content)} chars, final: {len(final_answer)} chars")
            
        except json.JSONDecodeError as json_error:
            print(f"  [WARNING] JSON parsing failed: {str(json_error)}. Attempting thinking extraction from raw response.")
            
            # Last resort: extract thinking from raw response using regex
            if raw_response:
                thinking_content, final_answer = extract_thinking_and_final_answer(raw_response)
                print(f"  [INFO] Thinking extraction from raw response - thinking: {len(thinking_content)} chars, final: {len(final_answer)} chars")
            else:
                thinking_content = ""
                final_answer = f"Error: Failed to generate report with model {report_llm}"
    
    print(f"  [DEBUG] Final thinking length: {len(thinking_content)} characters")
    print(f"  [DEBUG] Final answer length: {len(final_answer)} characters")
    print(f"  [DEBUG] Final answer preview: {final_answer[:200]}...")
    
    # Store both thinking and final answer in state
    state["final_answer"] = final_answer
    state["thinking_process"] = thinking_content
    print("  [INFO] Final report generated successfully with structured output approach")
    
    state["current_position"] = "report_writer"
    return state


def quality_checker_node(state: ResearcherStateV2, config: RunnableConfig) -> ResearcherStateV2:
    """
    LangGraph node that performs quality checking on the final report.
    
    Args:
        state: ResearcherStateV2 containing final answer and other state
        config: RunnableConfig for the graph
    
    Returns:
        Updated ResearcherStateV2 with quality check results
    """
    print("--- Starting quality checker node ---")
    
    # Check if quality checker is enabled
    enable_quality_checker = state.get("enable_quality_checker", False)
    if not enable_quality_checker:
        print("  [INFO] Quality checker disabled. Skipping quality check.")
        state["quality_check"] = {"enabled": False, "message": "Quality checker disabled", "score": 0}
        state["current_position"] = "quality_checker"
        return state
    
    # Extract state variables
    final_answer = state.get("final_answer", "")
    all_reranked_summaries = state.get("all_reranked_summaries", [])
    user_query = state.get("user_query", "")
    report_llm = state.get("report_llm", "gpt-oss:20b")
    detected_language = state.get("detected_language", "English")
    
    print(f"  [DEBUG] Quality checker enabled")
    print(f"  [DEBUG] Final answer length: {len(final_answer)} characters")
    print(f"  [DEBUG] Reranked summaries count: {len(all_reranked_summaries)}")
    
    # If no final answer or summaries, skip quality check
    if not final_answer or not all_reranked_summaries:
        print("  [WARNING] Missing final answer or summaries. Skipping quality check.")
        state["quality_check"] = {"enabled": True, "message": "Missing data for quality check", "score": 0}
        state["current_position"] = "quality_checker"
        return state
    
    # Format source documents for quality assessment
    formatted_docs = []
    for item in all_reranked_summaries:
        # Handle different possible structures of reranked summaries
        if isinstance(item, dict):
            query = item.get('query', item.get('research_query', 'Unknown Query'))
            summary = item.get('summary', item.get('content', ''))
            score = item.get('score', 0)
        else:
            # Fallback for unexpected structure
            query = 'Unknown Query'
            summary = str(item)
            score = 0
        
        formatted_docs.append(f"\n--- Research Query: {query} (Relevance Score: {score}/10) ---")
        formatted_docs.append(f"Content: {summary}\n")
    
    source_documents = "\n".join(formatted_docs)
    
    # Use the LLM-based quality assessment prompt
    system_prompt = LLM_QUALITY_CHECKER_SYSTEM_PROMPT.format(
        language=detected_language
    )
    
    human_prompt = LLM_QUALITY_CHECKER_HUMAN_PROMPT.format(
        final_answer=final_answer,
        all_reranked_summaries=all_reranked_summaries,
        query=user_query,
        language=detected_language
    )
    
    try:
        print(f"  [DEBUG] Calling quality checker with model: {report_llm}")
        quality_assessment = invoke_ollama(
            system_prompt=system_prompt,
            user_prompt=human_prompt,
            model=report_llm
        )
        
        print(f"  [DEBUG] Quality assessment length: {len(quality_assessment)} characters")
        print(f"  [DEBUG] Assessment preview: {quality_assessment[:200]}...")
        
        # Parse the JSON response from the quality assessment
        assessment_text = quality_assessment if isinstance(quality_assessment, str) else str(quality_assessment)
        
        try:
            # Parse the JSON response
            import json
            import re
            
            print(f"  [DEBUG] Original response length: {len(assessment_text)}")
            print(f"  [DEBUG] Original response preview: {assessment_text[:200]}...")
            
            # Clean the response by removing <think> tags and their content
            # Handle both <think>...</think> and <think>...<think> patterns
            cleaned_text = re.sub(r'<think>.*?(?:</think>|<think>)', '', assessment_text, flags=re.DOTALL | re.IGNORECASE)
            cleaned_text = cleaned_text.strip()
            
            print(f"  [DEBUG] Cleaned response length: {len(cleaned_text)}")
            print(f"  [DEBUG] Cleaned response preview: {cleaned_text[:200]}...")
            
            # Try multiple patterns to extract JSON from cleaned text
            json_patterns = [
                r'\{[^{}]*"quality_score"[^{}]*\}',  # Simple single-level JSON
                r'\{(?:[^{}]|\{[^{}]*\})*"quality_score"(?:[^{}]|\{[^{}]*\})*\}',  # Nested JSON
                r'\{.*?"quality_score".*?\}',  # More flexible pattern
            ]
            
            quality_data = None
            json_str = None
            
            for pattern in json_patterns:
                json_match = re.search(pattern, cleaned_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    try:
                        quality_data = json.loads(json_str)
                        print(f"  [DEBUG] Successfully parsed JSON with pattern: {pattern[:30]}...")
                        break
                    except json.JSONDecodeError:
                        continue
            
            # Fallback: try simple boundary detection on cleaned text
            if not quality_data:
                json_start = cleaned_text.find('{')
                json_end = cleaned_text.rfind('}') + 1
                
                if json_start != -1 and json_end > json_start:
                    json_str = cleaned_text[json_start:json_end]
                    try:
                        quality_data = json.loads(json_str)
                        print(f"  [DEBUG] Successfully parsed JSON with boundary detection")
                    except json.JSONDecodeError:
                        pass
            
            if quality_data:
                
                # Extract values directly from JSON (mapping LLM response to GUI expectations)
                overall_score = quality_data.get('quality_score', 0)
                passes_quality = quality_data.get('is_accurate', False)
                needs_improvement = quality_data.get('improvement_needed', not passes_quality)  # Map improvement_needed to needs_improvement
                improvement_suggestions = quality_data.get('improvement_suggestions', '')
                issues_found = quality_data.get('issues_found', [])
                missing_elements = quality_data.get('missing_elements', [])
                citation_issues = quality_data.get('citation_issues', [])
                
                print(f"  [DEBUG] Parsed JSON successfully")
                print(f"  [DEBUG] Quality score: {overall_score}")
                print(f"  [DEBUG] Is accurate: {passes_quality}")
                print(f"  [DEBUG] Improvement suggestions: {improvement_suggestions[:100]}...")
                
            else:
                raise ValueError("No valid JSON found in response")
                
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  [ERROR] Failed to parse JSON response: {e}")
            print(f"  [ERROR] LLM returned non-JSON format. Response preview: {assessment_text[:500]}...")
            
            # Since LLM is not following JSON format, provide a reasonable fallback
            # For now, assume quality fails and provide generic improvement suggestions
            overall_score = 0
            passes_quality = False
            needs_improvement = True  # Quality failed, so improvement is needed
            improvement_suggestions = "The LLM quality checker did not return the expected JSON format. Please ensure the report follows proper structure, includes accurate citations, and addresses the user's query comprehensively."
            issues_found = ["JSON parsing failed"]
            missing_elements = []
            citation_issues = []
        
        print(f"  [INFO] Quality Assessment Score: {overall_score}/400")
        print(f"  [INFO] Quality Assessment Result: {'PASS' if passes_quality else 'FAIL'}")
        
        # Create structured quality check result (using JSON field names for GUI compatibility)
        quality_result = {
            "enabled": True,
            "assessment_type": "llm_json_assessment",
            # New JSON structure field names (matching LLM response)
            "quality_score": overall_score,  # GUI reads this
            "is_accurate": passes_quality,  # GUI reads this
            "improvement_needed": needs_improvement,  # GUI reads this
            "improvement_suggestions": improvement_suggestions if 'improvement_suggestions' in locals() else "",
            "issues_found": issues_found if 'issues_found' in locals() else [],
            "missing_elements": missing_elements if 'missing_elements' in locals() else [],
            "citation_issues": citation_issues if 'citation_issues' in locals() else [],
            # Backward compatibility fields
            "overall_score": overall_score,
            "max_score": 400,
            "passes_quality": passes_quality,
            "needs_improvement": needs_improvement,
            "threshold": 300,
            "full_assessment": assessment_text,
            "score": overall_score  # For compatibility with quality_router
        }
        
        # Store quality check results in state
        state["quality_check"] = quality_result
        state["current_position"] = "quality_checker"
        
        # If quality assessment fails, add improvement suggestions to additional_context for report writer loop
        if not passes_quality:
            print("  [INFO] Quality assessment FAILED. Adding improvement suggestions to additional_context for report writer.")
            
            # Use the parsed improvement suggestions from JSON, or fallback
            if 'improvement_suggestions' not in locals() or not improvement_suggestions:
                improvement_suggestions = "Please improve the report based on the quality assessment feedback. Focus on better factual accuracy, semantic coherence, structural organization, and proper source attribution."
            
            print(f"  [INFO] Using improvement suggestions: {improvement_suggestions[:200]}...")
            
            # Add improvement suggestions to additional_context
            current_additional_context = state.get("additional_context", "")
            updated_additional_context = current_additional_context + f"\n\nImprovement Suggestions: {improvement_suggestions}"
            
            state["additional_context"] = updated_additional_context
            
            # Increment reflection count
            reflection_count = state.get("reflection_count", 0) + 1
            state["reflection_count"] = reflection_count
            
            # Mark that we need to loop back to report writer (but only if we haven't exceeded the limit)
            state["quality_check"]["needs_improvement"] = True
            state["quality_check"]["improvement_suggestions"] = improvement_suggestions
            state["quality_check"]["reflection_count"] = reflection_count
            
            print(f"  [INFO] Updated additional_context length: {len(updated_additional_context)} characters")
            print(f"  [INFO] Reflection count: {reflection_count}")
        else:
            print("  [INFO] Quality assessment PASSED. No improvement needed.")
            state["quality_check"]["needs_improvement"] = False
        
        return state
        
    except Exception as e:
        print(f"  [ERROR] Quality checker failed: {str(e)}")
        state["quality_check"] = {
            "enabled": True, 
            "error": str(e), 
            "message": "Quality check failed due to error", 
            "score": 0
        }
        state["current_position"] = "quality_checker_error"
        return state


def quality_router(state: ResearcherStateV2) -> str:
    """
    Router function that determines the next step after quality checking.
    Limits reflection loop to only one iteration.
    
    Args:
        state: ResearcherStateV2 containing quality check results
    
    Returns:
        str: Next node name ("report_writer" for improvement loop, END for completion)
    """
    print("--- Quality router decision ---")
    
    quality_check = state.get("quality_check", {})
    reflection_count = state.get("reflection_count", 0)
    
    # If quality checker is disabled, go to END
    if not quality_check.get("enabled", False):
        print("  [ROUTER] Quality checker disabled -> END")
        return END
    
    # If quality check failed and needs improvement, check reflection count limit
    needs_improvement = quality_check.get("needs_improvement", False)
    
    if needs_improvement and reflection_count < 1:
        print(f"  [ROUTER] Quality assessment FAILED (reflection count: {reflection_count}) -> report_writer (for improvement)")
        return "report_writer"
    elif needs_improvement and reflection_count >= 1:
        print(f"  [ROUTER] Quality assessment FAILED but reflection limit reached (count: {reflection_count}) -> END")
        return END
    else:
        print("  [ROUTER] Quality assessment PASSED -> END")
        return END


def tavily_search(query: str) -> str:
    """
    Search for information using Tavily API.
    
    Args:
        query: The search query to find information about
        
    Returns:
        str: Formatted search results or error message
    """
    try:
        print(f"  [TAVILY] Searching for: {query}")
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return "Error: Tavily API key not found in environment variables"
        
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": api_key,
            "query": query,
            "search_depth": "basic",
            "include_answer": True,
            "include_raw_content": False,
            "max_results": 5
        }
        
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            data = response.json()
            results = []
            
            # Add the direct answer if available
            if data.get("answer"):
                results.append(f"Direct Answer: {data['answer']}")
                results.append("---")
            
            # Add search results
            for i, result in enumerate(data.get("results", [])[:5], 1):
                results.append(f"Result {i}:")
                results.append(f"Title: {result.get('title', 'N/A')}")
                results.append(f"Content: {result.get('content', 'N/A')}")
                results.append(f"URL: {result.get('url', 'N/A')}")
                results.append("---")
            
            return "\n".join(results) if results else "No search results found"
        else:
            return f"Search failed with status code: {response.status_code}"
    except Exception as e:
        return f"Search error: {str(e)}"


def web_tavily_searcher_node(state: ResearcherStateV2, config: RunnableConfig) -> ResearcherStateV2:
    """
    LangGraph node that performs internet search using Tavily API.
    
    This node:
    1. Uses LLM to generate optimal search term based on user query and context
    2. Performs Tavily search with the generated term
    3. Uses LLM to summarize and wrap up the search results
    4. Updates the internet_result field in state
    
    Args:
        state: ResearcherStateV2 containing user query and context
        config: RunnableConfig for the graph
    
    Returns:
        Updated ResearcherStateV2 with internet_result field populated
    """
    print("\n=== WEB TAVILY SEARCHER NODE ===")
    
    user_query = state.get("user_query", "")
    additional_context = state.get("additional_context", "")
    report_llm = state.get("report_llm", "gpt-oss:20b")
    language = state.get("detected_language", "English")
    
    print(f"  User Query: {user_query}")
    print(f"  Additional Context: {additional_context[:100]}..." if additional_context else "  No additional context")
    print(f"  Using LLM: {report_llm}")
    
    try:
        # Step 1: Generate optimal search term using LLM
        search_term_prompt = f"""
You are an expert at formulating internet search queries. Based on the user's query and additional context, create ONE perfect search term that will yield the most relevant and recent information.

User Query: {user_query}
Additional Context: {additional_context}

Instructions:
- Create a concise, specific search term consisting of the most important keywords from the initial User Query and the Additional Context (typically 4-8 words)
- Focus on the most important keywords from the initial User Query
- Add relevant points from the Additional Context
- Consider recent developments or current information needs
- Avoid overly broad or generic terms

MOST IMPORTANT: 
- You MUST reply with a single search term in string format and no prefix or suffix.
- You MUST use {language} language to respond.

Search Term:"""
        
        print("  [STEP 1] Generating optimal search term...")
        search_term_response = invoke_ollama(
            model=report_llm,
            system_prompt="You are an expert search query optimizer. Generate concise, effective search terms.",
            user_prompt=search_term_prompt
        )
        
        # Clean the response to remove <think> blocks and extra formatting
        clean_response = clean_llm_response(search_term_response)
        search_term = clean_response.replace("Search Term:", "").strip()
        print(f"  Raw LLM response length: {len(search_term_response)} characters")
        print(f"  Cleaned search term: {search_term}")
        
        # Step 2: Perform Tavily search
        print("  [STEP 2] Performing Tavily search...")
        search_results = tavily_search(search_term)
        
        if search_results.startswith("Error:") or search_results.startswith("Search error:"):
            print(f"  Search failed: {search_results}")
            return {
                **state,
                "internet_result": f"Internet search failed: {search_results}",
                "internet_search_term": search_term
            }
        
        print(f"  Search completed. Results length: {len(search_results)} characters")
        
        # Step 3: Summarize and wrap up results using LLM
        wrap_up_prompt = f"""
You are an expert research analyst. Based on the user's query and the internet search results, create a comprehensive summary of the main findings.
MOST IMPORTANT: You MUST ONLY use the internet search results to create the summary. Do not use any internal LLM knowledge.

User Query: {user_query}
Additional Context: {additional_context}

Internet Search Results:
{search_results}

Instructions:
- Summarize the most relevant and important information
- Focus mainly on answering the User Query
- Include key facts, recent developments, and important details
- Organize the information clearly and logically
- Cite sources (mention URLs and titles)
- Keep the summary comprehensive but concise (max 500 words)
- Do not include any additional information that is not directly related to the User Query.

MOST IMPORTANT: You MUST use {language} language to respond.

Summary:"""
        
        print("  [STEP 3] Wrapping up search results...")
        wrap_up_response = invoke_ollama(
            model=report_llm,
            system_prompt="You are an expert research analyst. Create comprehensive, well-organized summaries of internet search results.",
            user_prompt=wrap_up_prompt
        )
        
        # Clean the response to remove <think> blocks
        internet_result = clean_llm_response(wrap_up_response)
        print(f"  Raw wrap-up response length: {len(wrap_up_response)} characters")
        print(f"  Cleaned internet result length: {len(internet_result)} characters")
        
        return {
            **state,
            "internet_result": internet_result,
            "internet_search_term": search_term
        }
        
    except Exception as e:
        error_msg = f"Error in web_tavily_searcher_node: {str(e)}"
        print(f"  ERROR: {error_msg}")
        # Try to include search_term if it was generated before the error
        search_term = locals().get('search_term', 'Not generated due to error')
        return {
            **state,
            "internet_result": error_msg,
            "internet_search_term": search_term
        }


def web_search_router(state: ResearcherStateV2) -> str:
    """
    Router function that determines whether to use web search or go directly to report writer.
    
    Args:
        state: ResearcherStateV2 containing web search configuration
    
    Returns:
        str: Next node name ("web_tavily_searcher" or "report_writer")
    """
    # Check if web search is enabled in the state or use a default
    web_search_enabled = state.get("web_search_enabled", False)
    
    if web_search_enabled:
        print("  [ROUTER] Web search enabled -> web_tavily_searcher")
        return "web_tavily_searcher"
    else:
        print("  [ROUTER] Web search disabled -> report_writer")
        return "report_writer"


def source_linker_node(state: ResearcherStateV2, config: RunnableConfig = None) -> ResearcherStateV2:
    """
    Convert source references in the final answer to clickable PDF links.
    
    Args:
        state: ResearcherStateV2 containing final_answer and selected_database
        config: LangGraph configuration (optional)
    
    Returns:
        Updated state with linked_final_answer containing clickable source links
    """
    print("\n=== SOURCE LINKER NODE ===")
    
    # Import the linkify_sources function
    from src.rag_helpers_v1_1 import linkify_sources
    
    try:
        # Get final answer and database context from state
        final_answer = state.get("final_answer", "")
        selected_database = state.get("selected_database", None)
        
        print(f"  [DEBUG] Final answer length: {len(final_answer)} characters")
        print(f"  [DEBUG] Selected database: {selected_database}")
        
        if not final_answer:
            print("  [WARNING] No final answer found in state")
            return {**state, "linked_final_answer": ""}
        
        # Convert source references to clickable links
        linked_answer = linkify_sources(final_answer, selected_database)
        
        print(f"  [DEBUG] Linked answer length: {len(linked_answer)} characters")
        print("  [SUCCESS] Source linking completed")
        
        # Return updated state with linked final answer
        return {**state, "linked_final_answer": linked_answer}
        
    except Exception as e:
        print(f"  [ERROR] Source linking failed: {str(e)}")
        # Fallback: return original final answer as linked answer
        return {**state, "linked_final_answer": state.get("final_answer", "")}


def quality_router_with_source_linker(state: ResearcherStateV2) -> str:
    """
    Router function that determines whether to improve the report or proceed to source linking.
    
    Args:
        state: ResearcherStateV2 containing quality check results
    
    Returns:
        str: Next node name ("report_writer" for improvement or "source_linker" to finish)
    """
    print("--- Quality router with source linker decision ---")
    
    quality_check = state.get("quality_check", {})
    
    # Ensure quality_check is not None and has get method
    if quality_check is None:
        quality_check = {}
    
    # Get quality assessment results with multiple field checks for compatibility
    quality_score = quality_check.get("quality_score", quality_check.get("score", 0))
    passes_quality = quality_check.get("passes_quality", quality_check.get("is_accurate", False))
    needs_improvement = quality_check.get("needs_improvement", quality_check.get("improvement_needed", True))
    reflection_count = state.get("reflection_count", 0)
    
    # Check if quality checker is disabled
    if not quality_check.get("enabled", True):
        print("  [QUALITY ROUTER] Quality checker disabled -> source_linker")
        return "source_linker"
    
    print(f"  [QUALITY ROUTER] Score: {quality_score}, Passes: {passes_quality}, Needs improvement: {needs_improvement}, Reflections: {reflection_count}")
    
    # CRITICAL: Limit reflection loops to prevent infinite recursion
    MAX_REFLECTIONS = 1  # Only allow 1 improvement attempt
    
    # If quality passes OR we've reached max reflections, proceed to source linker
    if passes_quality or reflection_count >= MAX_REFLECTIONS:
        if passes_quality:
            print("  [QUALITY ROUTER] Quality assessment PASSED -> source_linker")
        else:
            print(f"  [QUALITY ROUTER] Max reflections reached ({reflection_count}/{MAX_REFLECTIONS}) -> source_linker")
        return "source_linker"
    
    # If quality needs improvement and we haven't exceeded reflection limit
    if needs_improvement and reflection_count < MAX_REFLECTIONS:
        print(f"  [QUALITY ROUTER] Quality needs improvement (reflection {reflection_count + 1}/{MAX_REFLECTIONS}) -> report_writer")
        return "report_writer"
    
    # Fallback: proceed to source linker to prevent infinite loops
    print("  [QUALITY ROUTER] Fallback decision -> source_linker")
    return "source_linker"


def create_rerank_reporter_graph():
    """
    Create the LangGraph workflow for reranking and report generation with quality reflection loop.
    
    Returns:
        Compiled LangGraph workflow
    """
    print("Creating rerank-reporter graph with quality reflection loop...")
    
    clear_cuda_memory()
    # Create the workflow
    workflow = StateGraph(ResearcherStateV2)
    
    # Add nodes
    workflow.add_node("reranker", reranker_node)
    workflow.add_node("web_tavily_searcher", web_tavily_searcher_node)
    workflow.add_node("report_writer", report_writer_node)
    workflow.add_node("quality_checker", quality_checker_node)
    workflow.add_node("source_linker", source_linker_node)
    
    # Add conditional edge from reranker based on web search setting
    workflow.add_conditional_edges(
        "reranker",
        web_search_router,
        {
            "web_tavily_searcher": "web_tavily_searcher",
            "report_writer": "report_writer"
        }
    )
    
    # Add edge from web_tavily_searcher to report_writer
    workflow.add_edge("web_tavily_searcher", "report_writer")
    workflow.add_edge("report_writer", "quality_checker")
    
    # Add conditional edge from quality_checker using the router with source linker
    workflow.add_conditional_edges(
        "quality_checker",
        quality_router_with_source_linker,
        {
            "report_writer": "report_writer",  # Loop back for improvement
            "source_linker": "source_linker"  # Quality passed, proceed to source linking
        }
    )
    
    # Add edge from source_linker to END
    workflow.add_edge("source_linker", END)
    
    # Set entry point
    workflow.set_entry_point("reranker")
    
    # Compile with recursion limit to prevent infinite loops
    return workflow.compile(
        checkpointer=None,  # No checkpointing needed for this workflow
        debug=False
    )


def create_sample_documents() -> Dict[str, List[Document]]:
    """Create sample documents for testing."""
    sample_docs = {
        "What are the benefits of renewable energy?": [
            Document(
                page_content="Renewable energy sources like solar and wind power offer significant environmental benefits by reducing greenhouse gas emissions and air pollution. They help combat climate change and improve public health.",
                metadata={"source": "environmental_report.pdf", "page": 1}
            ),
            Document(
                page_content="Economic advantages of renewable energy include job creation in manufacturing, installation, and maintenance sectors. The renewable energy industry has created millions of jobs worldwide.",
                metadata={"source": "economic_analysis.pdf", "page": 3}
            ),
            Document(
                page_content="Energy independence is a key benefit of renewable sources. Countries can reduce reliance on fossil fuel imports and achieve greater energy security through domestic renewable resources.",
                metadata={"source": "energy_policy.pdf", "page": 2}
            )
        ],
        "How does solar power work?": [
            Document(
                page_content="Solar photovoltaic cells convert sunlight directly into electricity through the photovoltaic effect. When photons hit the solar cell, they knock electrons loose from atoms, generating electrical current.",
                metadata={"source": "solar_tech.pdf", "page": 1}
            ),
            Document(
                page_content="Solar thermal systems use mirrors or lenses to concentrate sunlight to heat a fluid, which then generates steam to drive turbines for electricity production.",
                metadata={"source": "solar_thermal.pdf", "page": 2}
            )
        ]
    }
    return sample_docs


def generate_graph_visualization():
    """Generate and display the LangGraph workflow visualization."""
    try:
        graph = create_rerank_reporter_graph()
        
        # Try to get Mermaid PNG with xray=True
        try:
            img = graph.get_graph(xray=True).draw_mermaid_png()
            return img, "png"
        except Exception as e:
            # Fallback to Mermaid text
            mermaid_code = graph.get_graph(xray=True).draw_mermaid()
            return mermaid_code, "mermaid"
            
    except Exception as e:
        return f"Error generating visualization: {str(e)}", "error"


def main():
    st.title("📊 LangGraph Rerank & Reporter")
    st.markdown("*Advanced reranking and report generation using LangGraph state logic*")
    
    # Initialize session state
    if "graph" not in st.session_state:
        st.session_state.graph = create_rerank_reporter_graph()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Load available models
        try:
            report_llm_models = get_report_llm_models()
        except Exception as e:
            st.error(f"Error loading models: {e}")
            report_llm_models = ["gpt-oss:20b", "qwen3:latest", "deepseek-r1:latest"]
        
        # Model selection
        selected_model = st.selectbox(
            "Select LLM Model",
            options=report_llm_models,
            index=0
        )
        
        # Language selection
        language = st.selectbox(
            "Language",
            options=["English", "German"],
            index=0
        )
        
        # Quality checker toggle
        enable_quality_checker = st.checkbox(
            "Enable Quality Checker",
            value=True,
            help="Enable LLM-based quality assessment and improvement of the final report"
        )
        
        # Web search toggle
        enable_web_search = st.checkbox(
            "Enable Web Search",
            value=False,
            help="Enable internet search using Tavily API to supplement document summaries with recent information"
        )
        
        # Database selection for source linking
        st.subheader("📚 Source Linking")
        available_databases = [
            "NORM__Qwen--Qwen3-Embedding-0.6B--3000--600",
            "StrlSch__Qwen--Qwen3-Embedding-0.6B--3000--600",
            "None (No source linking)"
        ]
        selected_database = st.selectbox(
            "Select Database for Source Linking",
            options=available_databases,
            index=0,
            help="Choose the database to use for converting source references to clickable PDF links"
        )
        
        # Convert "None" selection to None
        if selected_database == "None (No source linking)":
            selected_database = None
        
        # Additional context input
        additional_context = st.text_area(
            "Additional Context",
            placeholder="Enter any additional context or conversation history...",
            height=100
        )
        
        # Graph visualization in sidebar
        st.subheader("🕸️ Workflow Graph")
        try:
            graph_result, graph_type = generate_graph_visualization()
            
            if graph_type == "png":
                st.image(graph_result, caption="LangGraph Workflow", use_container_width=True)
            elif graph_type == "mermaid":
                st.code(graph_result, language="mermaid")
            else:
                st.error(graph_result)
                
        except Exception as e:
            st.warning(f"Could not generate graph visualization: {str(e)}")
            st.text("reranker → report_writer → END")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Input")
        
        # User query input
        initial_query = st.text_area(
            "Initial Query",
            placeholder="Enter your research question...",
            height=100,
            key="initial_query"
        )
        
        # Sample documents section
        st.subheader("📄 Sample Data")
        use_sample = st.checkbox("Use sample documents", value=True)
        
        if use_sample:
            sample_docs = create_sample_documents()
            
            # Display sample documents structure
            st.info(f"Using {len(sample_docs)} sample research queries with documents")
            for query, docs in sample_docs.items():
                with st.expander(f"Query: {query} ({len(docs)} docs)"):
                    for i, doc in enumerate(docs, 1):
                        st.write(f"**Document {i}:**")
                        st.write(doc.page_content[:200] + "...")
                        st.caption(f"Source: {doc.metadata.get('source', 'Unknown')}")
        
        else:
            st.info("Manual input mode - configure research queries and summaries below")
            
            # Manual input for research queries and summaries
            research_queries_input = st.text_area(
                "Research Queries (one per line)",
                placeholder="What are the benefits of renewable energy?\nHow does solar power work?",
                height=100,
                key="research_queries"
            )
            
            # Document content input
            st.subheader("📄 Document Summaries")
            st.write("Enter summaries for each query above:")
            
            if research_queries_input:
                research_queries = [q.strip() for q in research_queries_input.split('\n') if q.strip()]
                
                # Create input fields for each query
                custom_summaries = {}
                for i, query in enumerate(research_queries):
                    st.write(f"**Query {i+1}:** {query}")
                    summary_text = st.text_area(
                        f"Summary for query {i+1}",
                        placeholder="Enter document summary content...",
                        height=100,
                        key=f"summary_{i}"
                    )
                    if summary_text:
                        custom_summaries[query] = [Document(page_content=summary_text, metadata={"custom": True})]
    
    with col2:
        st.header("📊 Results")
        
        if st.button("🚀 Process with LangGraph", type="primary"):
            if not initial_query:
                st.error("Please enter an initial query.")
                return
            
            try:
                # Prepare data based on input method
                if use_sample:
                    search_summaries = create_sample_documents()
                else:
                    if not research_queries_input:
                        st.error("Please provide research queries.")
                        return
                    
                    research_queries = [q.strip() for q in research_queries_input.split('\n') if q.strip()]
                    search_summaries = {}
                    
                    for i, query in enumerate(research_queries):
                        summary_key = f"summary_{i}"
                        if summary_key in st.session_state and st.session_state[summary_key]:
                            docs = [Document(page_content=st.session_state[summary_key], metadata={"custom": True})]
                            search_summaries[query] = docs
                
                if not search_summaries:
                    st.error("No summaries provided. Please add document content.")
                    return
                
                # Process with LangGraph
                with st.spinner("🔄 Processing with LangGraph workflow..."):
                    # Prepare initial state
                    initial_state = {
                        "user_query": initial_query,
                        "search_summaries": search_summaries,
                        "additional_context": additional_context,
                        "report_llm": selected_model,
                        "detected_language": language,
                        "current_position": 0,
                        "research_queries": list(search_summaries.keys()),
                        "final_answer": "",
                        "all_reranked_summaries": [],
                        "enable_quality_checker": enable_quality_checker,
                        "quality_check": None,
                        "reflection_count": 0,
                        "web_search_enabled": enable_web_search,
                        "internet_result": None,
                        "internet_search_term": None,
                        "selected_database": selected_database  # Add database for source linking
                    }
                    
                    # Execute the graph with recursion limit to prevent infinite loops
                    from langgraph.types import RunnableConfig
                    config = RunnableConfig(
                        recursion_limit=10,  # Limit to 10 steps maximum
                        max_concurrency=1
                    )
                    final_state = st.session_state.graph.invoke(initial_state, config=config)
                    
                    # Display results
                    st.success("✅ LangGraph processing complete!")
                    
                    # Display processing steps
                    with st.expander("🔍 Processing Details", expanded=True):
                        st.write(f"**Current Position:** {final_state.get('current_position', 'unknown')}")
                        st.write(f"**Research Queries Processed:** {len(final_state.get('research_queries', []))}")
                        
                        # Debug state keys
                        st.write("**State Keys:**", list(final_state.keys()))
                        st.write("**Reranked summaries count:**", len(final_state.get("all_reranked_summaries", [])))
                        st.write("**Reranked summaries (alt):**", len(final_state.get("reranked_summaries", [])))
                        
                        # Display reranked documents
                        reranked_summaries = final_state.get("all_reranked_summaries", [])
                        # Also check for reranked_summaries in the state
                        if not reranked_summaries:
                            reranked_summaries = final_state.get("reranked_summaries", [])
                        if reranked_summaries:
                            st.subheader("📊 Reranked Documents")
                            
                            # Summary statistics
                            total_docs = len(reranked_summaries)
                            avg_score = sum(item.get('score', 0) for item in reranked_summaries) / total_docs if total_docs > 0 else 0
                            
                            col_stats1, col_stats2, col_stats3 = st.columns(3)
                            with col_stats1:
                                st.metric("Total Documents", total_docs)
                            with col_stats2:
                                st.metric("Average Score", f"{avg_score:.2f}/10")
                            with col_stats3:
                                max_score = max([item.get('score', 0) for item in reranked_summaries], default=0)
                                st.metric("Highest Score", f"{max_score:.2f}/10")
                            
                            # Display reranked documents in ranked order
                            for rank, item in enumerate(reranked_summaries, 1):
                                score = item.get('score', 0)
                                summary_data = item.get('summary', {})
                                
                                # Handle different summary formats
                                if isinstance(summary_data, dict):
                                    content = summary_data.get('Content', str(summary_data))
                                    source = summary_data.get('Source', 'Unknown')
                                else:
                                    # Handle Document objects or string content
                                    if hasattr(summary_data, 'page_content'):
                                        content = summary_data.page_content
                                        source = summary_data.metadata.get('source', 'Unknown') if hasattr(summary_data, 'metadata') else 'Unknown'
                                    else:
                                        content = str(summary_data)
                                        source = 'Unknown'
                                
                                with st.expander(f"🥇 Rank #{rank} (Score: {score:.2f}/10)", expanded=rank <= 3):
                                    st.markdown(f"**Score:** {score:.2f}/10")
                                    st.markdown(f"**Query:** {item.get('query', 'N/A')}")
                                    st.markdown(f"**Source:** {source}")
                                    st.markdown(f"**Original Index:** {item.get('original_index', 'N/A')}")
                                    
                                    st.markdown("**Content:**")
                                    st.markdown(content)
                                    
                                    # Add separator except for last item
                                    if rank < len(reranked_summaries):
                                        st.divider()
                        else:
                            st.warning("No reranked summaries found. Check input data.")
                    
                    # Display internet search results if available
                    internet_result = final_state.get("internet_result")
                    internet_search_term = final_state.get("internet_search_term")
                    if internet_result and internet_result.strip():
                        st.subheader("🌐 Internet Search Results")
                        
                        # Check if web search was enabled
                        web_search_enabled = final_state.get("web_search_enabled", False)
                        if web_search_enabled:
                            st.success("✅ Web search was enabled and executed successfully")
                        else:
                            st.info("ℹ️ Web search was not enabled for this query")
                        
                        # Display the generated search term if available
                        if internet_search_term:
                            st.info(f"🔍 **Generated Search Term:** `{internet_search_term}`")
                        
                        # Display the internet search results
                        with st.expander("📄 Internet Search Summary", expanded=True):
                            st.markdown(internet_result)
                    elif final_state.get("web_search_enabled", False):
                        st.subheader("🌐 Internet Search Results")
                        st.warning("⚠️ Web search was enabled but no results were obtained. Check your Tavily API key and internet connection.")
                    
                    # Display quality checker results if enabled
                    quality_check = final_state.get("quality_check", {})
                    if quality_check and quality_check.get("enabled", False):
                        st.subheader("🔍 Quality Assessment")
                        
                        # Get values with backward compatibility
                        overall_score = quality_check.get("quality_score", 0)
                        max_score = quality_check.get("max_score", 400)
                        passes_quality = quality_check.get("is_accurate", False)
                        needs_improvement = quality_check.get("improvement_needed", False)
                        improvement_suggestions = quality_check.get("improvement_suggestions", "")
                        
                        # New JSON structure fields
                        issues_found = quality_check.get("issues_found", [])
                        missing_elements = quality_check.get("missing_elements", [])
                        citation_issues = quality_check.get("citation_issues", [])
                        
                        # Display quality metrics
                        col_q1, col_q2, col_q3 = st.columns(3)
                        with col_q1:
                            st.metric("Quality Score", f"{overall_score}/{max_score}")
                        with col_q2:
                            status_color = "🟢" if passes_quality else "🔴"
                            st.metric("Assessment", f"{status_color} {'PASS' if passes_quality else 'FAIL'}")
                        with col_q3:
                            if needs_improvement:
                                improvement_status = "🔄 Needs Improvement"
                            else:
                                improvement_status = "✅ Quality Passed"
                            st.metric("Status", improvement_status)
                        
                        # Display detailed quality analysis if available
                        if issues_found or missing_elements or citation_issues:
                            st.markdown("#### 📊 Detailed Quality Analysis")
                            
                            col_detail1, col_detail2, col_detail3 = st.columns(3)
                            
                            with col_detail1:
                                if issues_found:
                                    st.markdown("**🚨 Issues Found:**")
                                    # Handle both string and list formats
                                    if isinstance(issues_found, str):
                                        st.markdown(f"• {issues_found}")
                                    elif isinstance(issues_found, list):
                                        for issue in issues_found:
                                            st.markdown(f"• {issue}")
                                    else:
                                        st.markdown(f"• {str(issues_found)}")
                                else:
                                    st.markdown("**✅ No Issues Found**")
                            
                            with col_detail2:
                                if missing_elements:
                                    st.markdown("**❓ Missing Elements:**")
                                    # Handle both string and list formats
                                    if isinstance(missing_elements, str):
                                        st.markdown(f"• {missing_elements}")
                                    elif isinstance(missing_elements, list):
                                        for element in missing_elements:
                                            st.markdown(f"• {element}")
                                    else:
                                        st.markdown(f"• {str(missing_elements)}")
                                else:
                                    st.markdown("**✅ All Elements Present**")
                            
                            with col_detail3:
                                if citation_issues:
                                    st.markdown("**📚 Citation Issues:**")
                                    # Handle both string and list formats
                                    if isinstance(citation_issues, str):
                                        st.markdown(f"• {citation_issues}")
                                    elif isinstance(citation_issues, list):
                                        for citation in citation_issues:
                                            st.markdown(f"• {citation}")
                                    else:
                                        st.markdown(f"• {str(citation_issues)}")
                                else:
                                    st.markdown("**✅ Citations OK**")
                        
                        # Show improvement suggestions if they were generated
                        if improvement_suggestions:
                            with st.expander("📝 Improvement Suggestions", expanded=False):
                                st.markdown(improvement_suggestions)
                        
                        # Show full assessment if available
                        full_assessment = quality_check.get("full_assessment", "")
                        if full_assessment:
                            with st.expander("🔍 Full Quality Assessment", expanded=False):
                                st.text(full_assessment)
                    
                    # Display final report with proper structured output parsing
                    final_answer = final_state.get("final_answer", "")
                    thinking_process = final_state.get("thinking_process", "")
                    linked_final_answer = final_state.get("linked_final_answer", "")
                    
                    if final_answer and final_answer.strip():
                        st.subheader("📋 Final Report")
                        
                        # Show improvement notice if quality checker triggered reflection loop
                        if quality_check and quality_check.get("needs_improvement", False):
                            st.info("ℹ️ This report has been regenerated based on quality assessment feedback through reflection loop.")
                        
                        # Try to parse structured output (JSON format with thinking/final)
                        display_content = ""
                        thinking_content = ""
                        
                        # First, check if we have separate thinking_process from the report writer
                        if thinking_process and thinking_process.strip():
                            thinking_content = thinking_process
                            print(f"  [DEBUG] Using separate thinking_process: {len(thinking_content)} chars")
                        
                        # Try to parse the final_answer as structured JSON
                        try:
                            import json
                            # Check if final_answer is JSON with thinking/final structure
                            if final_answer.strip().startswith('{') and final_answer.strip().endswith('}'):
                                parsed_json = json.loads(final_answer)
                                if 'thinking' in parsed_json and 'final' in parsed_json:
                                    thinking_content = parsed_json.get('thinking', '')
                                    display_content = parsed_json.get('final', '')
                                    print(f"  [DEBUG] Parsed JSON structure - thinking: {len(thinking_content)} chars, final: {len(display_content)} chars")
                                else:
                                    display_content = final_answer
                            else:
                                display_content = final_answer
                        except (json.JSONDecodeError, ValueError):
                            # Not JSON, try to extract <think> blocks
                            import re
                            think_pattern = r'<think>(.*?)(?:</think>|<think>)'
                            think_matches = re.findall(think_pattern, final_answer, re.DOTALL | re.IGNORECASE)
                            
                            if think_matches and not thinking_content:
                                thinking_content = "\n\n".join([match.strip() for match in think_matches])
                            
                            # Remove <think> blocks from the main answer
                            display_content = re.sub(r'<think>.*?(?:</think>|<think>)', '', final_answer, flags=re.DOTALL | re.IGNORECASE)
                            display_content = display_content.strip()
                            print(f"  [DEBUG] Extracted thinking blocks: {len(thinking_content)} chars, clean content: {len(display_content)} chars")
                        
                        # Use linked answer if available (with clickable source links)
                        if linked_final_answer and linked_final_answer.strip():
                            print(f"  [DEBUG] Using linked_final_answer with source links: {len(linked_final_answer)} chars")
                            # Apply the same parsing to linked answer
                            try:
                                if linked_final_answer.strip().startswith('{') and linked_final_answer.strip().endswith('}'):
                                    parsed_linked = json.loads(linked_final_answer)
                                    if 'final' in parsed_linked:
                                        display_content = parsed_linked.get('final', linked_final_answer)
                                    else:
                                        display_content = linked_final_answer
                                else:
                                    display_content = linked_final_answer
                            except (json.JSONDecodeError, ValueError):
                                display_content = linked_final_answer
                            
                            # Display with HTML support for clickable links
                            st.markdown(display_content, unsafe_allow_html=True)
                        else:
                            # No linked answer, display regular content
                            if display_content:
                                st.markdown(display_content)
                            else:
                                st.warning("The answer appears to contain only thinking process. Please check the LLM response.")
                        
                        # Show thinking process in a collapsed expander if found
                        if thinking_content and thinking_content.strip():
                            with st.expander("🧠 LLM Thinking Process", expanded=False):
                                st.markdown(thinking_content)
                        
                        # Download button for the report (use original final_answer)
                        download_content = display_content if display_content else final_answer
                        st.download_button(
                            label="📥 Download Report",
                            data=download_content,
                            file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                            mime="text/markdown"
                        )
                    else:
                        st.subheader("📄 Final Report")
                        st.warning("No report generated. This could be due to:")
                        st.markdown("- No reranked summaries found")
                        st.markdown("- All summaries scored below relevance threshold")
                        st.markdown("- Report generation failed")
                        
                        # Show debug info
                        with st.expander("🔧 Debug Information"):
                            st.json({
                                "has_final_answer": bool(final_answer),
                                "final_answer_length": len(final_answer) if final_answer else 0,
                                "reranked_summaries_count": len(reranked_summaries),
                                "research_queries": final_state.get("research_queries", []),
                                "user_query": final_state.get("user_query", ""),
                                "quality_check_enabled": quality_check.get("enabled", False) if quality_check else False
                            })
                    
            except Exception as e:
                st.error(f"Error during processing: {str(e)}")
                st.exception(e)
        
    # Debug information
    with st.expander("🔧 Debug Information"):
        st.write("**Session State Keys:**")
        st.write(list(st.session_state.keys()))
        st.write("**Graph Available:**", st.session_state.graph is not None)

if __name__ == "__main__":
    main()
