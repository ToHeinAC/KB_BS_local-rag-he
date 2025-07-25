import datetime
import os
import pathlib
import operator
import re
from typing_extensions import Literal, Annotated
from langgraph.constants import Send
from langgraph.graph import START, END, StateGraph
from langchain_core.runnables.config import RunnableConfig
from langchain_core.documents import Document
from src.configuration_v1_1 import Configuration, get_config_instance
from src.vector_db_v1_1 import get_or_create_vector_db, search_documents, get_embedding_model_path
from src.state_v2_0 import ResearcherStateV2, InitState
from src.prompts_v1_1 import (
    LANGUAGE_DETECTOR_SYSTEM_PROMPT, LANGUAGE_DETECTOR_HUMAN_PROMPT,
    RESEARCH_QUERY_WRITER_SYSTEM_PROMPT, RESEARCH_QUERY_WRITER_HUMAN_PROMPT,
    SUMMARIZER_SYSTEM_PROMPT, SUMMARIZER_HUMAN_PROMPT,
    REPORT_WRITER_SYSTEM_PROMPT, REPORT_WRITER_HUMAN_PROMPT,
)
from src.utils_v1_1 import format_documents_with_metadata, invoke_ollama, parse_output, tavily_search, DetectedLanguage, Queries
from src.rag_helpers_v1_1 import source_summarizer_ollama, format_documents_as_plain_text, parse_document_to_formatted_content
import time

# Get the directory path of the current file
this_path = os.path.dirname(os.path.abspath(__file__))

def extract_embedding_model_from_db_name(db_dir_name):
    """
    Extract the embedding model name from the database directory name.
    
    This function properly handles the database naming convention where model names
    are stored in directory names like "Qwen/Qwen--Qwen3-Embedding-0.6B--3000--600".
    
    Args:
        db_dir_name (str): The database directory name (e.g., "Qwen/Qwen--Qwen3-Embedding-0.6B--3000--600")
        
    Returns:
        str: The extracted embedding model name (e.g., "Qwen/Qwen3-Embedding-0.6B")
    """
    # Split by '--' and reconstruct the model name
    parts = db_dir_name.split('--')
    if len(parts) >= 2:
        # The first part should contain the model organization and name
        # Example: "Qwen/Qwen--Qwen3-Embedding-0.6B--3000--600" -> "Qwen/Qwen3-Embedding-0.6B"
        model_parts = parts[0].split('/')
        if len(model_parts) >= 2:
            # Reconstruct: "Qwen/Qwen" -> "Qwen", then add the actual model name from parts[1]
            org = model_parts[0]  # "Qwen"
            model_name = parts[1]  # "Qwen3-Embedding-0.6B"
            return f"{org}/{model_name}"
        else:
            # Fallback: try to extract from the full first part
            return parts[0].replace('--', '/')
    return db_dir_name

# Import existing nodes from graph_v1_1 and adapt them
from src.graph_v1_1 import (
    detect_language as detect_language_v1,
    display_embedding_model_info as display_embedding_model_info_v1,
    retrieve_rag_documents as retrieve_rag_documents_v1,
    summarize_query_research as summarize_query_research_v1,
    generate_final_answer as generate_final_answer_v1,
    quality_checker as quality_checker_v1,
    query_router as query_router_v1,
    update_position as update_position_v1,
    quality_router as quality_router_v1
)

# HITL Node functions
def analyse_user_feedback(state: InitState, config: RunnableConfig):
    """Analyze user feedback in the context of the research workflow."""
    print("--- Analyzing user feedback ---")
    
    query = state["user_query"]
    detected_language = state.get("detected_language", "English")
    human_feedback = state.get("human_feedback", "")
    additional_context = state.get("additional_context", "")
    
    model_to_use = state.get("report_llm", config["configurable"].get("report_llm", "deepseek-r1:latest"))
    
    if not human_feedback.strip():
        print("No human feedback provided, skipping analysis")
        return {
            "analysis": "No human feedback provided for analysis.",
            "additional_context": additional_context,
            "current_position": "analyse_user_feedback"
        }
    
    system_prompt = f"""# ROLE
You are an expert conversation analyst specializing in research query refinement and context extraction.

# GOAL
Analyze the human feedback in the context of the initial research query to extract key insights, constraints, and additional context that will improve the research process.

# OUTPUT FORMAT
Provide a structured analysis in {detected_language} covering:
- **Key Insights**: Main points from the feedback
- **Research Focus**: Specific areas to emphasize
- **Constraints**: Any limitations or boundaries mentioned
- **Additional Context**: Background information provided
- **Recommendations**: How this feedback should influence the research approach

# CRITICAL CONSTRAINTS
- Write EXCLUSIVELY in {detected_language} language - NO EXCEPTIONS
- Be concise but comprehensive
- Focus on actionable insights for research improvement"""
    
    human_prompt = f"""# RESEARCH CONTEXT
Initial Query: {query}

# HUMAN FEEDBACK TO ANALYZE
{human_feedback}

# TASK
Analyze the human feedback above in the context of the initial research query and provide structured insights that will improve the research process:"""
    
    result = invoke_ollama(
        model=model_to_use,
        system_prompt=system_prompt,
        user_prompt=human_prompt,
    )
    
    parsed_result = parse_output(result)
    analysis = parsed_result["response"]
    
    if additional_context:
        additional_context += "\n\n"
    additional_context += f"Human Feedback Analysis:\n{analysis}"
    
    return {
        "analysis": analysis,
        "additional_context": additional_context,
        "current_position": "analyse_user_feedback"
    }

def generate_follow_up_questions(state: InitState, config: RunnableConfig):
    """Generate follow-up questions based on the current state and analysis."""
    print("--- Generating follow-up questions ---")
    
    query = state["user_query"]
    detected_language = state.get("detected_language", "English")
    analysis = state.get("analysis", "")
    additional_context = state.get("additional_context", "")
    
    model_to_use = state.get("report_llm", config["configurable"].get("report_llm", "deepseek-r1:latest"))
    
    system_prompt = f"""# ROLE
You are an expert research interviewer specializing in clarifying research requirements and gathering comprehensive context.

# GOAL
Generate exactly 3 strategic follow-up questions that will help clarify the research scope, gather missing context, and ensure the research meets the user's specific needs.

# OUTPUT FORMAT
Generate exactly 3 strategic questions in numbered format:
1. [Strategic question about scope/focus]
2. [Strategic question about context/background]
3. [Strategic question about preferences/approach]

# CRITICAL CONSTRAINTS
- Write EXCLUSIVELY in {detected_language} language - NO EXCEPTIONS
- Generate EXACTLY 3 questions, no more, no less
- Make questions specific and actionable
- Avoid yes/no questions - prefer open-ended inquiries"""
    
    human_prompt = f"""# RESEARCH CONTEXT
Initial Query: {query}

# CURRENT ANALYSIS
{analysis if analysis else "No detailed analysis available."}

# TASK
Based on the research context and analysis above, generate strategic follow-up questions that will help clarify requirements and improve the research process:"""
    
    result = invoke_ollama(
        model=model_to_use,
        system_prompt=system_prompt,
        user_prompt=human_prompt,
    )
    
    parsed_result = parse_output(result)
    follow_up_questions = parsed_result["response"]
    
    if additional_context:
        additional_context += "\n\n"
    additional_context += f"AI Follow-up Questions:\n{follow_up_questions}"
    
    return {
        "follow_up_questions": follow_up_questions,
        "additional_context": additional_context,
        "current_position": "generate_follow_up_questions"
    }

def generate_knowledge_base_questions(state: InitState, config: RunnableConfig):
    """
    Generate knowledge base questions and output research_queries as list[str].
    
    This is the final node in the HITL workflow that produces research_queries
    which will be passed to the main workflow. The function processes user feedback
    and analysis to generate a list of research queries for document retrieval.
    
    Args:
        state (InitState): The current state of the HITL workflow
        config (RunnableConfig): Configuration for the workflow execution
        
    Returns:
        InitState: Updated state with research_queries as list[str]
    """
    print("--- Generating knowledge base questions ---")
    
    query = state["user_query"]
    detected_language = state.get("detected_language", "English")
    additional_context = state.get("additional_context", "")
    max_queries = config["configurable"].get("max_search_queries", 3)
    report_llm = state.get("report_llm", config["configurable"].get("report_llm", "deepseek-r1:latest"))
    summarization_llm = state.get("summarization_llm", config["configurable"].get("summarization_llm", "deepseek-r1:latest"))
    
    # Use the report writer LLM for generating research queries (same as generate_research_queries)
    model_to_use = state.get("report_llm", config["configurable"].get("report_llm", "deepseek-r1:latest"))
    print(f"  [DEBUG] Knowledge Base Query LLM (report_llm): {model_to_use}")
    
    # Calculate max_queries_minus_one to avoid arithmetic in string format
    max_queries_minus_one = max_queries - 1
    
    # Format the system prompt using the same pattern as generate_research_queries
    system_prompt = RESEARCH_QUERY_WRITER_SYSTEM_PROMPT.format(
        max_queries=max_queries,
        max_queries_minus_one=max_queries_minus_one,
        date=datetime.datetime.now().strftime("%Y/%m/%d %H:%M"),
        language=detected_language
    )
    
    # Format the human prompt with HITL context
    human_prompt = RESEARCH_QUERY_WRITER_HUMAN_PROMPT.format(
        query=query,
        language=detected_language,
        additional_context=f"Consider this HITL conversation context when generating queries: {additional_context}" if additional_context else ""
    )
    
    # Using local llm model with Ollama (same pattern as generate_research_queries)
    result = invoke_ollama(
        model=model_to_use,
        system_prompt=system_prompt,
        user_prompt=human_prompt,
        output_format=Queries
    )
    
    # Extract queries and add the original query at the beginning
    research_queries = result.queries
    
    # Deduplicate queries - ensure the original query appears only once
    # Remove any duplicates of the original query from the generated list
    research_queries = [q for q in research_queries if q.strip().lower() != query.strip().lower()]
    
    # Insert the original query at the beginning
    research_queries.insert(0, query)
    
    # Additional deduplication - remove any remaining exact duplicates (case-insensitive)
    seen = set()
    deduplicated_queries = []
    for q in research_queries:
        q_lower = q.strip().lower()
        if q_lower not in seen:
            seen.add(q_lower)
            deduplicated_queries.append(q)
    
    research_queries = deduplicated_queries
    print(f"  [DEBUG] Generated knowledge base queries (deduplicated): {research_queries}")
    assert isinstance(research_queries, list), "research_queries must be a list"
    
    # Update additional context with the generated queries
    if additional_context:
        additional_context += "\n\n"
    additional_context += f"AI Knowledge Base Questions:\n" + "\n".join([f"{i+1}. {q}" for i, q in enumerate(research_queries)])
    
    # Debug logging to show the final research queries
    print(f"=== HITL WORKFLOW COMPLETE ===")
    print(f"Generated {len(research_queries)} research queries:")
    for i, query_item in enumerate(research_queries, 1):
        print(f"  {i}. {query_item}")
    print(f"=== HANDOVER TO MAIN WORKFLOW ===")
    
    # Return updated state with research_queries for main workflow
    return {
        "user_query": query,
        "detected_language": detected_language,
        "research_queries": research_queries,  # Key output for main workflow
        "additional_context": additional_context,
        "report_llm": report_llm,
        "summarization_llm": summarization_llm,
        "current_position": "hitl_complete"
    }

# Create HITL graph
def create_hitl_graph():
    """
    Create the Human-in-the-Loop (HITL) workflow graph.
    
    This graph implements the HITL workflow with the following steps:
    1. analyse_user_feedback: Analyzes user feedback and query
    2. generate_follow_up_questions: Generates follow-up questions for user refinement
    3. generate_knowledge_base_questions: Generates final research_queries for the main workflow
    
    The HITL workflow is designed to run before the main research workflow and produces
    research_queries that will be used as input to the main workflow.
    
    Returns:
        Compiled StateGraph: The compiled HITL workflow graph
    """
    workflow = StateGraph(InitState)
    
    workflow.add_node("analyse_user_feedback", analyse_user_feedback)
    workflow.add_node("generate_follow_up_questions", generate_follow_up_questions)
    workflow.add_node("generate_knowledge_base_questions", generate_knowledge_base_questions)
    
    workflow.add_edge(START, "analyse_user_feedback")
    workflow.add_edge("analyse_user_feedback", "generate_follow_up_questions")
    workflow.add_edge("generate_follow_up_questions", "generate_knowledge_base_questions")
    workflow.add_edge("generate_knowledge_base_questions", END)
    
    return workflow.compile()

# Wrapper functions for main workflow nodes
def display_embedding_model_info(state: ResearcherStateV2):
    """
    Display embedding model information with fixed embedding model extraction.
    """
    print("--- Displaying embedding model information ---")
    
    config = get_config_instance()
    use_ext_database = config.use_ext_database if hasattr(config, 'use_ext_database') else False
    selected_database = config.selected_database if hasattr(config, 'selected_database') else None
    
    if use_ext_database and selected_database:
        embedding_model = extract_embedding_model_from_db_name(selected_database)
        print(f"Using external database: {selected_database}")
        print(f"Extracted embedding model: {embedding_model}")
        
        # Update the global configuration with the correct embedding model
        from src.configuration_v1_1 import update_embedding_model
        update_embedding_model(embedding_model)
        
        return {
            "current_position": "display_embedding_model_info",
            "additional_context": state.get("additional_context", "") + f"\nUsing embedding model: {embedding_model} from database: {selected_database}"
        }
    else:
        return display_embedding_model_info_v1(state)

def retrieve_rag_documents(state: ResearcherStateV2, config: RunnableConfig):
    """Wrapper for retrieve_rag_documents_v1 to work with extended state."""
    return retrieve_rag_documents_v1(state, config)

def summarize_query_research(state: ResearcherStateV2, config: RunnableConfig):
    """Enhanced summarization that considers additional context from HITL."""
    print("--- Starting summarize_query_research with debug file output ---")
    additional_context = state.get("additional_context", "")
    
    # Prepare state for the original function
    if additional_context:
        enhanced_state = dict(state)
        enhanced_state["human_feedback"] = additional_context
        result = summarize_query_research_v1(enhanced_state, config)
    else:
        result = summarize_query_research_v1(state, config)
    
    # Write state to debug file after summarization
    try:
        import os
        import json
        from datetime import datetime
        
        # Create a timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_filename = f"summarize_query_research_debug.md"
        
        # Get the project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        debug_filepath = os.path.join(project_root, debug_filename)
        
        print(f"  [DEBUG] Writing debug state to {debug_filepath}")
        
        # Format state as markdown
        with open(debug_filepath, 'w', encoding='utf-8') as f:
            f.write(f"# Summarize Query Research Debug Output - {timestamp}\n\n")
            
            # Write original state keys
            f.write("## Original State Keys\n\n")
            f.write(", ".join(state.keys()) + "\n\n")
            
            # Write result keys
            f.write("## Result Keys\n\n")
            f.write(", ".join(result.keys()) + "\n\n")
            
            # Write search summaries if available
            if "search_summaries" in result:
                f.write("## Search Summaries\n\n")
                search_summaries = result["search_summaries"]
                f.write(f"Number of queries with summaries: {len(search_summaries)}\n\n")
                
                for query, summaries in search_summaries.items():
                    f.write(f"### Query: {query}\n\n")
                    f.write(f"Number of summaries: {len(summaries)}\n\n")
                    
                    for i, summary in enumerate(summaries):
                        f.write(f"#### Summary {i+1}\n\n")
                        f.write(f"```\n{summary.page_content}\n```\n\n")
                        f.write(f"**Metadata:** {json.dumps(summary.metadata, indent=2, default=str)}\n\n")
            
            # Write formatted documents if available
            if "formatted_documents" in result:
                f.write("## Formatted Documents\n\n")
                f.write(f"Number of formatted document sets: {len(result['formatted_documents'])}\n\n")
            
            # Query mapping has been removed from workflow
            f.write("## Note on Query Processing\n\n")
            f.write("Queries are now processed directly without numerical mapping.\n\n")
                
            f.write("## Complete State (Safe Serializable Keys)\n\n")
            # Only include serializable data in the complete state dump
            safe_state = {}
            for key, value in result.items():
                try:
                    # Test if value is JSON serializable
                    json.dumps({key: value}, default=str)
                    safe_state[key] = value
                except (TypeError, OverflowError):
                    safe_state[key] = f"[Not serializable: {type(value).__name__}]"
            
            f.write("```json\n")
            f.write(json.dumps(safe_state, indent=2, default=str))
            f.write("\n```\n")
            
        print(f"  [DEBUG] Successfully wrote debug state to {debug_filepath}")
    except Exception as e:
        print(f"  [ERROR] Failed to write debug state to file: {str(e)}")
    
    # Return the original result
    return result


def rerank_summaries(state: ResearcherStateV2, config: RunnableConfig):
    """
    Rerank summaries based on relevance and accuracy using LLM scoring.
    Ignores any 'importance_score' in metadata and uses LLM-based evaluation.
    """
    print("--- Starting rerank_summaries ---")
    
    # Get search summaries from the previous step
    search_summaries = state.get("search_summaries", {})
    user_query = state.get("user_query", "")
    additional_context = state.get("additional_context", "")
    
    if not search_summaries:
        print("  [WARNING] No search summaries found for reranking")
        return {"search_summaries": {}}
    
    # Get the report LLM for scoring (use report_llm for consistency)
    report_llm = state.get("report_llm", "qwen3:30b-a3b")
    
    # Robust language extraction - handle both string and dict formats
    detected_language_raw = state.get("detected_language", "English")
    if isinstance(detected_language_raw, dict):
        detected_language = detected_language_raw.get("detected_language", "English")
        print(f"  [DEBUG] Extracted language from dict in reranker: {detected_language}")
    else:
        detected_language = detected_language_raw
    
    # Ensure we have a valid string
    if not isinstance(detected_language, str):
        detected_language = "English"
        print(f"  [WARNING] Invalid language format in reranker, defaulting to English")
    
    print(f"  [INFO] Reranking summaries for {len(search_summaries)} queries using {report_llm}")
    
    # Rerank summaries for each query
    reranked_summaries = {}
    
    for query, summaries in search_summaries.items():
        print(f"  [INFO] Reranking {len(summaries)} summaries for query: '{query[:50]}...'")
        
        if not summaries:
            reranked_summaries[query] = []
            continue
            
        # Convert summaries to the format expected by reranker
        summary_list = []
        for summary_doc in summaries:
            # Extract content from the formatted summary
            content = summary_doc.page_content
            # Parse the content to extract just the summary text
            if "Content: " in content:
                summary_text = content.split("Content: ")[1].split("\n")[0]
            else:
                summary_text = content
            
            summary_list.append({
                "Content": summary_text,
                "original_doc": summary_doc
            })
        
        # Rerank this query's summaries
        reranked_results = _rerank_query_summaries(
            initial_query=user_query,
            query=query,
            summaries=summary_list,
            additional_context=additional_context,
            llm_model=report_llm,
            language=detected_language
        )
        
        # Convert back to Document format, preserving original metadata but updating order
        reranked_docs = []
        for result in reranked_results:
            original_doc = result["summary"]["original_doc"]
            # Update metadata with reranking score
            original_doc.metadata["rerank_score"] = result["score"]
            original_doc.metadata["original_index"] = result["original_index"]
            reranked_docs.append(original_doc)
        
        reranked_summaries[query] = reranked_docs
        print(f"  [INFO] Reranked {len(reranked_docs)} summaries for query '{query[:30]}...'")
    
    print(f"  [INFO] Completed reranking for all {len(reranked_summaries)} queries")
    
    # Write state to debug file after reranking
    try:
        import os
        import json
        from datetime import datetime
        
        # Create a timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_filename = f"reranked_debug.md"
        
        # Get the project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        debug_filepath = os.path.join(project_root, debug_filename)
        
        print(f"  [DEBUG] Writing reranked debug state to {debug_filepath}")
        
        # Format state as markdown
        with open(debug_filepath, 'w', encoding='utf-8') as f:
            f.write(f"# Reranked Summaries Debug Output - {timestamp}\n\n")
            
            # Write original state info
            f.write("## Original State Info\n\n")
            f.write(f"- User Query: {user_query}\n")
            f.write(f"- Additional Context: {additional_context}\n")
            f.write(f"- Report LLM: {report_llm}\n")
            f.write(f"- Detected Language: {detected_language}\n")
            f.write(f"- Number of queries processed: {len(reranked_summaries)}\n\n")
            
            # Write reranked summaries details
            f.write("## Reranked Summaries\n\n")
            
            for query, summaries in reranked_summaries.items():
                f.write(f"### Query: {query}\n\n")
                f.write(f"Number of summaries: {len(summaries)}\n\n")
                
                for i, summary_doc in enumerate(summaries):
                    f.write(f"#### Summary {i+1} (Rank {i+1})\n\n")
                    
                    # Extract rerank score and original index from metadata
                    rerank_score = summary_doc.metadata.get('rerank_score', 'N/A')
                    original_index = summary_doc.metadata.get('original_index', 'N/A')
                    
                    f.write(f"**Rerank Score:** {rerank_score}\n")
                    f.write(f"**Original Index:** {original_index}\n")
                    f.write(f"**Position:** {summary_doc.metadata.get('position', 'N/A')}\n\n")
                    
                    # Extract and display the content
                    content = summary_doc.page_content
                    if "Content: " in content:
                        summary_text = content.split("Content: ")[1].split("\n")[0]
                        f.write(f"**Summary Content:**\n```\n{summary_text}\n```\n\n")
                    else:
                        f.write(f"**Summary Content:**\n```\n{content}\n```\n\n")
                    
                    # Write full metadata
                    f.write(f"**Full Metadata:**\n```json\n{json.dumps(summary_doc.metadata, indent=2, default=str)}\n```\n\n")
                    
                f.write("---\n\n")
            
            # Write scoring statistics
            f.write("## Scoring Statistics\n\n")
            all_scores = []
            for summaries in reranked_summaries.values():
                for summary_doc in summaries:
                    score = summary_doc.metadata.get('rerank_score')
                    if score is not None and score != 'N/A':
                        all_scores.append(score)
            
            if all_scores:
                f.write(f"- Total summaries scored: {len(all_scores)}\n")
                f.write(f"- Average score: {sum(all_scores)/len(all_scores):.2f}\n")
                f.write(f"- Highest score: {max(all_scores):.2f}\n")
                f.write(f"- Lowest score: {min(all_scores):.2f}\n\n")
            else:
                f.write("- No valid scores found\n\n")
            
            # Write complete reranked state (safe serializable keys)
            f.write("## Complete Reranked State (Safe Serializable Keys)\n\n")
            safe_state = {
                "search_summaries": {},
                "user_query": user_query,
                "additional_context": additional_context,
                "report_llm": report_llm,
                "detected_language": detected_language
            }
            
            # Add serializable summary info
            for query, summaries in reranked_summaries.items():
                safe_state["search_summaries"][query] = []
                for summary_doc in summaries:
                    safe_summary = {
                        "content_preview": summary_doc.page_content[:200] + "..." if len(summary_doc.page_content) > 200 else summary_doc.page_content,
                        "metadata": summary_doc.metadata
                    }
                    safe_state["search_summaries"][query].append(safe_summary)
            
            f.write("```json\n")
            f.write(json.dumps(safe_state, indent=2, default=str))
            f.write("\n```\n")
            
        print(f"  [DEBUG] Successfully wrote reranked debug state to {debug_filepath}")
    except Exception as e:
        print(f"  [ERROR] Failed to write reranked debug state to file: {str(e)}")
    
    # Return updated state with reranked summaries
    return {
        "search_summaries": reranked_summaries
    }


def _rerank_query_summaries(initial_query: str, query: str, summaries: list[dict], 
                           additional_context: str, llm_model: str, language: str) -> list[dict]:
    """
    Rerank a list of summaries based on relevance & accuracy.
    Ignores any 'importance_score' in metadata.
    
    Args:
        initial_query: The original user question.
        query: The specific research query being processed.
        summaries: List of dicts, each with a 'Content' key.
        additional_context: Conversation history or domain context.
        llm_model: LLM model to use for scoring.
        language: Detected language for the evaluation.
    
    Returns:
        A list of dicts with keys: 'summary', 'score', 'original_index',
        sorted by descending score.
    """
    results = []
    for idx, s in enumerate(summaries):
        content = s["Content"]
        score = _score_summary(initial_query, query, content, additional_context, llm_model, language)
        results.append({
            "summary": s,
            "score": score,
            "original_index": idx
        })
    # sort highest score first
    return sorted(results, key=lambda x: x["score"], reverse=True)


def _score_summary(initial_query: str, query: str, content: str, context: str, 
                  llm_model: str, language: str) -> float:
    """
    Ask the LLM to score a single summary 0–10.
    """
    prompt = f"""
You are an expert evaluator of document summary relevance.

TASK: Score the following summary for its relevance and accuracy regarding the original query and the given context.

ORIGINAL USER QUERY:
{initial_query}

SPECIFIC RESEARCH QUERY:
{query}

ADDITIONAL CONTEXT:
{context}

SUMMARY TO ASSESS:
{content}

SCORING CRITERIA (weights in parentheses):
1. Direct relevance to the original user query (40%)
2. Specificity and level of detail (25%)
3. Alignment with the research query context (20%)
4. Factual accuracy and completeness (15%)

INSTRUCTIONS:
Return ONLY a number between 0 and 10:
- 10 = perfectly relevant and accurate
- 8-9 = very relevant with strong detail
- 6-7 = relevant but somewhat incomplete
- 4-5 = partially relevant
- 0-3 = poorly relevant or inaccurate

Respond in {language}.
"""
    
    try:
        response = invoke_ollama(
            system_prompt="You are an expert document evaluator. Provide only numerical scores.",
            user_prompt=prompt,
            model=llm_model
        )
        
        # Extract numerical score from response
        import re
        match = re.search(r"\b(\d+(?:\.\d+)?)\b", response)
        if match:
            score = float(match.group(1))
            # Ensure score is within valid range
            return max(0.0, min(10.0, score))
        else:
            print(f"  [WARNING] Could not extract score from LLM response: {response[:100]}...")
            return 5.0
    except Exception as e:
        print(f"  [ERROR] Failed to score summary: {str(e)}")
        return 5.0

def generate_final_answer(state: ResearcherStateV2, config: RunnableConfig):
    """Enhanced final answer generation that makes stronger use of reranked summaries."""
    print("--- Enhanced generate_final_answer with reranked summaries prioritization ---")
    
    # Get required state variables
    user_query = state.get("user_query", "")
    search_summaries = state.get("search_summaries", {})
    additional_context = state.get("additional_context", "")
    report_llm = state.get("report_llm", "qwen3:30b-a3b")
    
    # Robust language extraction - handle both string and dict formats
    detected_language_raw = state.get("detected_language", "English")
    if isinstance(detected_language_raw, dict):
        detected_language = detected_language_raw.get("detected_language", "English")
        print(f"  [DEBUG] Extracted language from dict in final answer: {detected_language}")
    else:
        detected_language = detected_language_raw
    
    # Ensure we have a valid string
    if not isinstance(detected_language, str):
        detected_language = "English"
        print(f"  [WARNING] Invalid language format in final answer, defaulting to English")
    
    if not search_summaries:
        print("  [WARNING] No search summaries found for final answer generation")
        return {"final_answer": "Error: No summaries available for generating final answer."}
    
    try:
        # Collect all reranked summaries from all queries
        all_reranked_summaries = []
        
        for query, summaries in search_summaries.items():
            print(f"  [INFO] Processing {len(summaries)} summaries for query: '{query[:50]}...'")
            
            for summary_doc in summaries:
                # Extract rerank score and content
                rerank_score = summary_doc.metadata.get('rerank_score', 0.0)
                
                # Extract content from the formatted summary
                content = summary_doc.page_content
                if "Content: " in content:
                    summary_text = content.split("Content: ")[1].split("\n")[0]
                else:
                    summary_text = content
                
                all_reranked_summaries.append({
                    'score': rerank_score,
                    'summary': {
                        'Content': summary_text,
                        'query': query
                    },
                    'original_index': summary_doc.metadata.get('original_index', 0)
                })
        
        # Sort all summaries by rerank score (highest first)
        all_reranked_summaries.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"  [INFO] Using {len(all_reranked_summaries)} reranked summaries for final answer")
        
        if not all_reranked_summaries:
            return {"final_answer": "Error: No valid summaries found for generating final answer."}
        
        # Generate the enhanced prompt using reranked summaries
        final_answer_prompt = _generate_final_answer_prompt(
            initial_query=user_query,
            reranked_summaries=all_reranked_summaries,
            additional_context=additional_context,
            language=detected_language
        )
        
        print(f"  [INFO] Generating final answer using {report_llm} with {len(all_reranked_summaries)} prioritized summaries")
        
        # Generate final answer using the enhanced prompt
        final_answer = invoke_ollama(
            system_prompt=f"You are an expert assistant providing comprehensive answers. Respond in {detected_language}.",
            user_prompt=final_answer_prompt,
            model=report_llm
        )
        
        print(f"  [INFO] Final answer generated successfully (length: {len(final_answer)})")
        
        return {"final_answer": final_answer}
        
    except Exception as e:
        print(f"  [ERROR] Exception in enhanced generate_final_answer: {str(e)}")
        return {"final_answer": f"Error occurred during final answer generation: {str(e)}. Check logs for details."}


def _generate_final_answer_prompt(initial_query: str, reranked_summaries: list[dict], 
                                 additional_context: str = "", language: str = "English") -> str:
    """
    Create a prompt for generating the final answer using reranked summaries.
    
    Args:
        initial_query: The original user question
        reranked_summaries: List of summaries sorted by relevance score (highest first)
        additional_context: Conversation history or domain context
        language: Detected language for the response
    
    Returns:
        Formatted prompt string for the LLM
    """
    
    prompt = f"""You are an expert assistant providing comprehensive answers based on ranked document summaries.

TASK: Generate a complete and accurate answer to the user's query using the provided summaries. Prioritize information from higher-ranked summaries.

ORIGINAL QUERY:
{initial_query}

CONTEXT:
{additional_context}

RANKED SUMMARIES (ordered by relevance):

PRIMARY SOURCE (Highest Relevance - Score: {reranked_summaries[0]['score']:.1f}):
{reranked_summaries[0]['summary']['Content']}

SUPPORTING SOURCES:"""

    # Add remaining summaries as supporting sources
    for i, item in enumerate(reranked_summaries[1:], 2):
        prompt += f"""

Source {i} (Score: {item['score']:.1f}):
{item['summary']['Content']}"""

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

def quality_checker(state: ResearcherStateV2, config: RunnableConfig):
    """Enhanced quality checker that considers additional context from HITL."""
    additional_context = state.get("additional_context", "")
    if additional_context:
        enhanced_state = dict(state)
        enhanced_state["human_feedback"] = additional_context
        return quality_checker_v1(enhanced_state, config)
    else:
        return quality_checker_v1(state, config)

def query_router(state: ResearcherStateV2):
    """Wrapper for query_router_v1 to work with extended state."""
    return query_router_v1(state)

def update_position(state: ResearcherStateV2):
    """Wrapper for update_position_v1 to work with extended state."""
    return update_position_v1(state)

def quality_router(state: ResearcherStateV2):
    """Wrapper for quality_router_v1 to work with extended state."""
    return quality_router_v1(state)

def conditional_quality_router(state: ResearcherStateV2):
    """
    Router that checks if quality checking is enabled before routing to quality checker.
    If quality checking is disabled, goes directly to END.
    """
    # Check if quality checking is enabled in the state
    enable_quality_checker = state.get("enable_quality_checker", False)
    
    if enable_quality_checker:
        print("  [INFO] Quality checker enabled, routing to quality_router")
        # Use the existing quality router logic
        return quality_router_v1(state)
    else:
        print("  [INFO] Quality checker disabled, ending workflow")
        return "end"

# Create simplified main researcher graph
def create_main_graph():
    """
    Create the simplified main researcher graph that starts with retrieve_rag_documents.
    
    This graph is designed to work after the HITL workflow has completed and generated research_queries.
    The main workflow skips the detect_language and generate_research_queries nodes since those
    steps are already handled in the HITL workflow.
    
    The workflow follows these steps:
    1. retrieve_rag_documents: Retrieves documents based on research_queries from HITL
    2. update_position: Updates the current position in the research process
    3. summarize_query_research: Summarizes the retrieved documents
    4. rerank_summaries: Reranks the summaries
    5. generate_final_answer: Generates the final answer based on summaries
    6. quality_checker (optional): Checks the quality of the final answer
    
    Returns:
        Compiled StateGraph: The compiled main workflow graph
    """
    main_workflow = StateGraph(ResearcherStateV2)
    
    # Add nodes for main workflow starting with retrieve_rag_documents
    main_workflow.add_node("retrieve_rag_documents", retrieve_rag_documents)
    main_workflow.add_node("summarize_query_research", summarize_query_research)
    main_workflow.add_node("rerank_summaries", rerank_summaries)
    main_workflow.add_node("generate_final_answer", generate_final_answer)
    main_workflow.add_node("quality_checker", quality_checker)
    main_workflow.add_node("update_position", update_position)
    
    # Main workflow edges starting with retrieve_rag_documents
    main_workflow.add_edge(START, "retrieve_rag_documents")
    main_workflow.add_edge("retrieve_rag_documents", "update_position")
    main_workflow.add_edge("update_position", "summarize_query_research")
    main_workflow.add_edge("summarize_query_research", "rerank_summaries")
    main_workflow.add_edge("rerank_summaries", "generate_final_answer")
    
    # Conditional quality checker routing based on enable_quality_checker setting
    main_workflow.add_conditional_edges(
        "generate_final_answer",
        conditional_quality_router,
        {
            "quality_checker": "quality_checker",
            "end": END
        }
    )
    
    main_workflow.add_conditional_edges(
        "quality_checker",
        quality_router,
        {
            "generate_final_answer": "generate_final_answer",
            "end": END
        }
    )
    
    return main_workflow.compile()

# Create the graphs
hitl_graph = create_hitl_graph()
main_graph = create_main_graph()


# Export the main function for use in the app
researcher_main = main_graph
