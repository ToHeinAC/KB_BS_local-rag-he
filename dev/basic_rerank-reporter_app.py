import streamlit as st
import sys
import os
import json
import re
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
from typing_extensions import TypedDict

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import required modules
from src.rag_helpers_v1_1 import get_report_llm_models
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

# Page configuration
st.set_page_config(
    page_title="Basic Rerank & Reporter",
    page_icon="📊",
    layout="wide"
)

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
    report_llm = state.get("report_llm", "qwen3:latest")
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
                                additional_context: str = "", language: str = "English") -> str:
    """
    Create a prompt for generating the final answer using reranked summaries.
    """
    
    prompt = f"""You are an expert assistant providing comprehensive answers based on ranked document summaries.

TASK: Generate a complete and accurate answer to the user's query using the provided summaries. Prioritize information from higher-ranked summaries.

ORIGINAL QUERY:
{initial_query}

CONTEXT:
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

    prompt += f"""

INSTRUCTIONS:
• Base your answer PRIMARILY on the highest-ranked summary as it is most relevant to the query
• Use supporting sources to add context, details, or complementary information
• If supporting sources contradict the primary source, prioritize the primary source unless there's clear evidence of error
• Maintain accuracy and cite relevant legal references (§ sections) when mentioned
• Structure your response clearly with bullet points as preferred
• If information is incomplete, acknowledge limitations
• Focus on directly answering the original query
• Respond in {language} language

Generate a comprehensive answer that prioritizes the most relevant information while incorporating supporting details where appropriate."""

    return prompt


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
    report_llm = state.get("report_llm", "qwen3:latest")
    detected_language = state.get("detected_language", "English")
    
    print(f"  [DEBUG] User query: {user_query}")
    print(f"  [DEBUG] Reranked summaries count: {len(all_reranked_summaries)}")
    print(f"  [DEBUG] Report LLM: {report_llm}")
    print(f"  [DEBUG] Language: {detected_language}")
    
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
    prompt = generate_final_answer_prompt(
        initial_query=user_query,
        reranked_summaries=all_reranked_summaries,
        additional_context=additional_context,
        language=detected_language
    )
    
    print(f"  [DEBUG] Prompt length: {len(prompt)} characters")
    print(f"  [DEBUG] Prompt preview: {prompt[:200]}...")
    
    try:
        print(f"  [DEBUG] Calling invoke_ollama with model: {report_llm}")
        response = invoke_ollama(
            system_prompt="You are an expert research analyst. Provide comprehensive, well-structured reports based on the provided information.",
            user_prompt=prompt,
            model=report_llm
        )
        
        print(f"  [DEBUG] LLM response length: {len(response)} characters")
        print(f"  [DEBUG] Response preview: {response[:200]}...")
        
        state["final_answer"] = response
        print("  [INFO] Final report generated successfully")
        
    except Exception as e:
        print(f"  [ERROR] Exception in final report generation: {str(e)}")
        state["final_answer"] = f"Error occurred during final report generation: {str(e)}. Check logs for details."
        state["current_position"] = "report_writer_error"
        return state
    
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
    report_llm = state.get("report_llm", "qwen3:latest")
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


def create_rerank_reporter_graph():
    """
    Create the LangGraph workflow for reranking and report generation with quality reflection loop.
    
    Returns:
        Compiled LangGraph workflow
    """
    print("Creating rerank-reporter graph with quality reflection loop...")
    
    # Create the workflow
    workflow = StateGraph(ResearcherStateV2)
    
    # Add nodes
    workflow.add_node("reranker", reranker_node)
    workflow.add_node("report_writer", report_writer_node)
    workflow.add_node("quality_checker", quality_checker_node)
    
    # Add edges
    workflow.add_edge("reranker", "report_writer")
    workflow.add_edge("report_writer", "quality_checker")
    
    # Add conditional edge from quality_checker using the router
    workflow.add_conditional_edges(
        "quality_checker",
        quality_router,
        {
            "report_writer": "report_writer",  # Loop back for improvement
            END: END  # Quality passed, finish
        }
    )
    
    # Set entry point
    workflow.set_entry_point("reranker")
    
    return workflow.compile()


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
        
        # Try to get Mermaid PNG
        try:
            img = graph.get_graph().draw_mermaid_png()
            return img, "png"
        except Exception as e:
            # Fallback to Mermaid text
            mermaid_code = graph.get_graph().draw_mermaid()
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
            report_llm_models = ["qwen3:latest", "deepseek-r1:latest"]
        
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
                        "reflection_count": 0
                    }
                    
                    # Execute the graph
                    final_state = st.session_state.graph.invoke(initial_state)
                    
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
                    
                    # Display final report
                    final_answer = final_state.get("final_answer", "")
                    if final_answer and final_answer.strip():
                        st.subheader("📋 Final Report")
                        
                        # Show improvement notice if quality checker triggered reflection loop
                        if quality_check and quality_check.get("needs_improvement", False):
                            st.info("ℹ️ This report has been regenerated based on quality assessment feedback through reflection loop.")
                        
                        # Extract <think> blocks from the final answer
                        import re
                        
                        # Find all <think> blocks (handle both proper and malformed tags)
                        think_pattern = r'<think>(.*?)(?:</think>|<think>)'
                        think_matches = re.findall(think_pattern, final_answer, re.DOTALL | re.IGNORECASE)
                        
                        # Remove <think> blocks from the main answer
                        clean_answer = re.sub(r'<think>.*?(?:</think>|<think>)', '', final_answer, flags=re.DOTALL | re.IGNORECASE)
                        clean_answer = clean_answer.strip()
                        
                        # Display the clean answer
                        if clean_answer:
                            st.markdown(clean_answer)
                        else:
                            st.warning("The answer appears to contain only thinking process. Please check the LLM response.")
                        
                        # Show thinking process in a collapsed expander if found
                        if think_matches:
                            with st.expander("🧠 LLM Thinking Process", expanded=False):
                                for i, think_content in enumerate(think_matches, 1):
                                    if len(think_matches) > 1:
                                        st.markdown(f"**Thinking Block {i}:**")
                                    st.text(think_content.strip())
                                    if i < len(think_matches):
                                        st.divider()
                        
                        # Download button for the report
                        st.download_button(
                            label="📥 Download Report",
                            data=final_answer,
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
