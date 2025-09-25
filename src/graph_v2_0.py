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
    DEEP_ANALYSIS_SYSTEM_PROMPT, DEEP_ANALYSIS_HUMAN_PROMPT,
    KNOWLEDGE_BASE_SEARCH_SYSTEM_PROMPT, KNOWLEDGE_BASE_SEARCH_HUMAN_PROMPT,
)
from src.utils_v1_1 import format_documents_with_metadata, invoke_ollama, parse_output, tavily_search, DetectedLanguage, Queries
from pydantic import BaseModel
from src.rag_helpers_v1_1 import source_summarizer_ollama, format_documents_as_plain_text, parse_document_to_formatted_content
import time

# Pydantic model for structured final report output
class FinalReportOutput(BaseModel):
    """Structured output model for final report generation."""
    content: str  # The final report content in markdown format

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
    Generate knowledge base questions using a two-step process:
    1. First LLM call: Deep analysis of user query and HITL feedback
    2. Second LLM call: Generate knowledge base questions using initial query + deep analysis
    
    This is the final node in the HITL workflow that produces research_queries
    which will be passed to the main workflow.
    
    Args:
        state (InitState): The current state of the HITL workflow with all human feedback
        config (RunnableConfig): Configuration for the workflow execution
        
    Returns:
        InitState: Updated state with knowledge_base_questions, additional_context (deep_analysis), and research_queries
    """
    print("--- Generating knowledge base questions with two-step process ---")
    
    query = state["user_query"]
    detected_language = state.get("detected_language", "English")
    human_feedback = state.get("human_feedback", "")
    additional_context = state.get("additional_context", "")
    
    # Get max_search_queries from config or state
    max_search_queries = config["configurable"].get("max_search_queries", state.get("max_search_queries", 3))
    
    # Use the report writer LLM for both analysis and question generation
    model_to_use = state.get("report_llm", config["configurable"].get("report_llm", "deepseek-r1:latest"))
    print(f"  [DEBUG] Knowledge Base Query LLM (report_llm): {model_to_use}")
    print(f"  [DEBUG] Detected language: {detected_language}")
    print(f"  [DEBUG] Max search queries: {max_search_queries}")
    
    # ========================================
    # FIRST LLM CALL: Deep Analysis
    # ========================================
    print(f"  [DEBUG] Step 1/2: Performing deep analysis of user query and HITL feedback")
    
    # Format human feedback for analysis prompt
    human_feedback_text = ""
    if human_feedback:
        if isinstance(human_feedback, list):
            for i, feedback in enumerate(human_feedback):
                human_feedback_text += f"Exchange {i+1}: {feedback}\n"
        else:
            human_feedback_text = str(human_feedback)
    else:
        human_feedback_text = "No additional human feedback exchanges.\n"
    
    # Format the system prompt for deep analysis
    analysis_system_prompt = DEEP_ANALYSIS_SYSTEM_PROMPT.format(
        language=detected_language
    )
    
    # Format the human prompt for deep analysis
    analysis_human_prompt = DEEP_ANALYSIS_HUMAN_PROMPT.format(
        query=query,
        additional_context=additional_context if additional_context else "No detailed conversation history available.",
        human_feedback=human_feedback_text,
        language=detected_language
    )
    
    # First LLM invocation: Generate deep analysis
    print(f"  [DEBUG] Invoking first LLM call for deep analysis...")
    analysis_result = invoke_ollama(
        model=model_to_use,
        system_prompt=analysis_system_prompt,
        user_prompt=analysis_human_prompt,
    )
    
    # Parse the result to get the deep analysis
    analysis_parsed_result = parse_output(analysis_result)
    deep_analysis = analysis_parsed_result["response"]
    print(f"  [DEBUG] First LLM call completed. Deep analysis generated.")
    
    # ========================================
    # SECOND LLM CALL: Knowledge Base Questions
    # ========================================
    print(f"  [DEBUG] Step 2/2: Generating knowledge base questions using initial query + deep analysis")
    
    # Format the system prompt for knowledge base questions
    kb_system_prompt = KNOWLEDGE_BASE_SEARCH_SYSTEM_PROMPT.format(
        language=detected_language
    )
    
    # Format the human prompt for knowledge base questions
    kb_human_prompt = KNOWLEDGE_BASE_SEARCH_HUMAN_PROMPT.format(
        query=query,
        deep_analysis=deep_analysis,
        language=detected_language,
        max_queries=max_search_queries
    )
    
    # Second LLM invocation: Generate knowledge base questions
    print(f"  [DEBUG] Invoking second LLM call for knowledge base questions...")
    kb_result = invoke_ollama(
        model=model_to_use,
        system_prompt=kb_system_prompt,
        user_prompt=kb_human_prompt,
    )
    
    # Parse the result
    kb_parsed_result = parse_output(kb_result)
    knowledge_base_questions = kb_parsed_result["response"]
    print(f"  [DEBUG] Second LLM call completed. Knowledge base questions generated.")
    
    # Parse knowledge base questions into a list of research queries
    import re
    generated_queries = []
    for line in knowledge_base_questions.split('\n'):
        # Extract questions using regex pattern for numbered lists (1. Question)
        match = re.match(r'\d+\.\s*(.*)', line.strip())
        if match:
            generated_queries.append(match.group(1).strip())
    
    # Limit the number of generated queries to respect max_search_queries
    # Since we always include the original query, we can add (max_search_queries - 1) additional queries
    max_additional_queries = max(0, max_search_queries - 1)  # Ensure non-negative
    if len(generated_queries) > max_additional_queries:
        generated_queries = generated_queries[:max_additional_queries]
        print(f"  [DEBUG] Limited generated queries to {max_additional_queries} (max_search_queries={max_search_queries})")
    
    # Always include the original user query as the first research query
    research_queries = [query]  # Start with original query
    research_queries.extend(generated_queries)  # Add limited generated queries
    
    print(f"  [DEBUG] Parsed {len(generated_queries)} generated queries + 1 original query = {len(research_queries)} total research queries")
    assert isinstance(research_queries, list), "research_queries must be a list"
    
    # Debug logging to show the final research queries
    print(f"=== HITL WORKFLOW COMPLETE (Two-Step Process) ===")
    print(f"Generated {len(research_queries)} research queries:")
    for i, query_item in enumerate(research_queries, 1):
        print(f"  {i}. {query_item}")
    print(f"=== HANDOVER TO MAIN WORKFLOW ===")
    
    print(f"  [DEBUG] generate_knowledge_base_questions completed successfully with two separate LLM calls")
    
    # Return updated state with all required fields
    return {
        "user_query": query,
        "detected_language": detected_language,
        "knowledge_base_questions": knowledge_base_questions,  # Generated questions text
        "additional_context": deep_analysis,  # Deep analysis as additional_context
        "research_queries": research_queries,  # Parsed list for main workflow
        "report_llm": state.get("report_llm", config["configurable"].get("report_llm", "deepseek-r1:latest")),
        "summarization_llm": state.get("summarization_llm", config["configurable"].get("summarization_llm", "deepseek-r1:latest")),
        "current_position": "generate_knowledge_base_questions"
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
        
        # Format full state as markdown with hierarchical headings
        with open(debug_filepath, 'w', encoding='utf-8') as f:
            f.write(f"# Summarize Query Research Debug Output - {timestamp}\n\n")
            
            # Write the complete state with proper markdown hierarchy
            f.write("# Full State Dump\n\n")
            
            # Process all state variables as ## headings
            for key, value in result.items():
                f.write(f"## {key}\n\n")
                
                # Handle different data types with appropriate formatting
                if isinstance(value, dict):
                    if not value:
                        f.write("*Empty dictionary*\n\n")
                    else:
                        for sub_key, sub_value in value.items():
                            f.write(f"### {sub_key}\n\n")
                            
                            if isinstance(sub_value, list):
                                if not sub_value:
                                    f.write("*Empty list*\n\n")
                                else:
                                    for i, item in enumerate(sub_value):
                                        f.write(f"#### Item {i+1}\n\n")
                                        if hasattr(item, 'page_content'):
                                            # Document object
                                            f.write(f"**Content:**\n```\n{item.page_content}\n```\n\n")
                                            if hasattr(item, 'metadata'):
                                                f.write(f"**Metadata:**\n```json\n{json.dumps(item.metadata, indent=2, default=str)}\n```\n\n")
                                        else:
                                            # Regular item
                                            f.write(f"```\n{str(item)}\n```\n\n")
                            elif isinstance(sub_value, dict):
                                f.write(f"```json\n{json.dumps(sub_value, indent=2, default=str)}\n```\n\n")
                            else:
                                f.write(f"```\n{str(sub_value)}\n```\n\n")
                                
                elif isinstance(value, list):
                    if not value:
                        f.write("*Empty list*\n\n")
                    else:
                        for i, item in enumerate(value):
                            f.write(f"### Item {i+1}\n\n")
                            if hasattr(item, 'page_content'):
                                # Document object
                                f.write(f"**Content:**\n```\n{item.page_content}\n```\n\n")
                                if hasattr(item, 'metadata'):
                                    f.write(f"**Metadata:**\n```json\n{json.dumps(item.metadata, indent=2, default=str)}\n```\n\n")
                            else:
                                # Regular item
                                f.write(f"```\n{str(item)}\n```\n\n")
                                
                elif isinstance(value, str):
                    if value.strip():
                        # Check if it looks like JSON
                        try:
                            parsed_json = json.loads(value)
                            f.write(f"```json\n{json.dumps(parsed_json, indent=2, default=str)}\n```\n\n")
                        except (json.JSONDecodeError, TypeError):
                            # Regular string content
                            if len(value) > 200:
                                f.write(f"```\n{value}\n```\n\n")
                            else:
                                f.write(f"{value}\n\n")
                    else:
                        f.write("*Empty string*\n\n")
                        
                elif value is None:
                    f.write("*None*\n\n")
                    
                else:
                    # Handle other types (int, float, bool, etc.)
                    try:
                        # Try to serialize as JSON for complex objects
                        json_str = json.dumps(value, indent=2, default=str)
                        f.write(f"```json\n{json_str}\n```\n\n")
                    except (TypeError, OverflowError):
                        f.write(f"```\n{str(value)}\n```\n\n")
            
            # Add full ResearcherStateV2 workflow state dump at the end
            f.write("---\n\n")
            f.write("# FULL ResearcherStateV2 WORKFLOW STATE AT SUMMARIZE_QUERY_RESEARCH STEP\n\n")
            
            # Create a clean state dict for JSON serialization
            clean_state = {}
            for key, value in result.items():
                if key == 'retrieved_documents':
                    # Simplify document objects for JSON
                    clean_state[key] = {
                        query: [f"Document objects with metadata containing {', '.join(set([doc.metadata.get('name', ['Unknown'])[0] if isinstance(doc.metadata.get('name'), list) else doc.metadata.get('name', 'Unknown') for doc in docs]))} references"] 
                        for query, docs in value.items()
                    }
                elif key == 'search_summaries':
                    # Include full search summaries with complete content and metadata
                    clean_state[key] = {}
                    for query, docs in value.items():
                        if docs:
                            clean_state[key][query] = []
                            for doc in docs:
                                doc_info = {
                                    "content": doc.page_content,  # Full content, no truncation
                                    "metadata": {
                                        "position": doc.metadata.get('position', 'N/A'),
                                        "query": doc.metadata.get('query', 'N/A'),
                                        "name": doc.metadata.get('name', ['Unknown']),
                                        "path": doc.metadata.get('path', ['Unknown']),
                                        "importance_score": doc.metadata.get('importance_score', 'N/A')
                                    }
                                }
                                clean_state[key][query].append(doc_info)
                else:
                    clean_state[key] = value
            
            # Write the clean state as JSON
            f.write("```json\n")
            f.write(json.dumps(clean_state, indent=2, default=str, ensure_ascii=False))
            f.write("\n```\n\n")
            
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
        reranked_results = rerank_query_summaries(
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
        
        # Format full state as markdown with hierarchical headings
        with open(debug_filepath, 'w', encoding='utf-8') as f:
            f.write(f"# Reranked Summaries Debug Output - {timestamp}\n\n")
            
            # Write the complete state with proper markdown hierarchy
            f.write("# Full State Dump\n\n")
            
            # Create a comprehensive state object for the reranker
            full_state = {
                "user_query": user_query,
                "additional_context": additional_context,
                "report_llm": report_llm,
                "detected_language": detected_language,
                "search_summaries": reranked_summaries,
                "scoring_statistics": {}
            }
            
            # Calculate scoring statistics
            all_scores = []
            for summaries in reranked_summaries.values():
                for summary_doc in summaries:
                    score = summary_doc.metadata.get('rerank_score')
                    if score is not None and score != 'N/A':
                        all_scores.append(score)
            
            if all_scores:
                full_state["scoring_statistics"] = {
                    "total_summaries_scored": len(all_scores),
                    "average_score": sum(all_scores)/len(all_scores),
                    "highest_score": max(all_scores),
                    "lowest_score": min(all_scores),
                    "number_of_queries_processed": len(reranked_summaries)
                }
            else:
                full_state["scoring_statistics"] = {
                    "total_summaries_scored": 0,
                    "message": "No valid scores found",
                    "number_of_queries_processed": len(reranked_summaries)
                }
            
            # Process all state variables as ## headings
            for key, value in full_state.items():
                f.write(f"## {key}\n\n")
                
                # Handle different data types with appropriate formatting
                if isinstance(value, dict):
                    if not value:
                        f.write("*Empty dictionary*\n\n")
                    else:
                        for sub_key, sub_value in value.items():
                            f.write(f"### {sub_key}\n\n")
                            
                            if isinstance(sub_value, list):
                                if not sub_value:
                                    f.write("*Empty list*\n\n")
                                else:
                                    for i, item in enumerate(sub_value):
                                        f.write(f"#### Item {i+1} (Rank {i+1})\n\n")
                                        if hasattr(item, 'page_content'):
                                            # Document object with reranking info
                                            rerank_score = item.metadata.get('rerank_score', 'N/A')
                                            original_index = item.metadata.get('original_index', 'N/A')
                                            position = item.metadata.get('position', 'N/A')
                                            
                                            f.write(f"**Rerank Score:** {rerank_score}\n")
                                            f.write(f"**Original Index:** {original_index}\n")
                                            f.write(f"**Position:** {position}\n\n")
                                            
                                            f.write(f"**Content:**\n```\n{item.page_content}\n```\n\n")
                                            if hasattr(item, 'metadata'):
                                                f.write(f"**Metadata:**\n```json\n{json.dumps(item.metadata, indent=2, default=str)}\n```\n\n")
                                        else:
                                            # Regular item
                                            f.write(f"```\n{str(item)}\n```\n\n")
                            elif isinstance(sub_value, dict):
                                f.write(f"```json\n{json.dumps(sub_value, indent=2, default=str)}\n```\n\n")
                            else:
                                f.write(f"```\n{str(sub_value)}\n```\n\n")
                                
                elif isinstance(value, list):
                    if not value:
                        f.write("*Empty list*\n\n")
                    else:
                        for i, item in enumerate(value):
                            f.write(f"### Item {i+1}\n\n")
                            if hasattr(item, 'page_content'):
                                # Document object
                                f.write(f"**Content:**\n```\n{item.page_content}\n```\n\n")
                                if hasattr(item, 'metadata'):
                                    f.write(f"**Metadata:**\n```json\n{json.dumps(item.metadata, indent=2, default=str)}\n```\n\n")
                            else:
                                # Regular item
                                f.write(f"```\n{str(item)}\n```\n\n")
                                
                elif isinstance(value, str):
                    if value.strip():
                        # Check if it looks like JSON
                        try:
                            parsed_json = json.loads(value)
                            f.write(f"```json\n{json.dumps(parsed_json, indent=2, default=str)}\n```\n\n")
                        except (json.JSONDecodeError, TypeError):
                            # Regular string content
                            if len(value) > 200:
                                f.write(f"```\n{value}\n```\n\n")
                            else:
                                f.write(f"{value}\n\n")
                    else:
                        f.write("*Empty string*\n\n")
                        
                elif value is None:
                    f.write("*None*\n\n")
                    
                else:
                    # Handle other types (int, float, bool, etc.)
                    try:
                        # Try to serialize as JSON for complex objects
                        json_str = json.dumps(value, indent=2, default=str)
                        f.write(f"```json\n{json_str}\n```\n\n")
                    except (TypeError, OverflowError):
                        f.write(f"```\n{str(value)}\n```\n\n")
            
            # Add full ResearcherStateV2 workflow state dump at the end
            f.write("---\n\n")
            f.write("# FULL ResearcherStateV2 WORKFLOW STATE AT RERANK_SUMMARIES STEP\n\n")
            
            # Create a comprehensive state dict that includes the full workflow state
            complete_state = dict(state)  # Start with the original state
            complete_state["search_summaries"] = reranked_summaries  # Update with reranked summaries
            
            # Create a clean state dict for JSON serialization
            clean_state = {}
            for key, value in complete_state.items():
                if key == 'retrieved_documents':
                    # Simplify document objects for JSON
                    clean_state[key] = {
                        query: [f"Document objects with metadata containing {', '.join(set([doc.metadata.get('name', ['Unknown'])[0] if isinstance(doc.metadata.get('name'), list) else doc.metadata.get('name', 'Unknown') for doc in docs]))} references"] 
                        for query, docs in value.items()
                    }
                elif key == 'search_summaries':
                    # Include reranked summaries with scores and metadata
                    clean_state[key] = {}
                    for query, docs in value.items():
                        if docs:
                            clean_state[key][query] = []
                            for doc in docs:
                                doc_info = {
                                    "content": doc.page_content,  # Full content, no truncation
                                    "metadata": {
                                        "position": doc.metadata.get('position', 'N/A'),
                                        "query": doc.metadata.get('query', 'N/A'),
                                        "name": doc.metadata.get('name', ['Unknown']),
                                        "path": doc.metadata.get('path', ['Unknown']),
                                        "rerank_score": doc.metadata.get('rerank_score', 'N/A'),
                                        "original_index": doc.metadata.get('original_index', 'N/A')
                                    }
                                }
                                clean_state[key][query].append(doc_info)
                else:
                    clean_state[key] = value
            
            # Write the clean state as JSON
            f.write("```json\n")
            f.write(json.dumps(clean_state, indent=2, default=str, ensure_ascii=False))
            f.write("\n```\n\n")
            
        print(f"  [DEBUG] Successfully wrote reranked debug state to {debug_filepath}")
    except Exception as e:
        print(f"  [ERROR] Failed to write reranked debug state to file: {str(e)}")
    
    # Return updated state with reranked summaries
    return {
        "search_summaries": reranked_summaries
    }


def rerank_query_summaries(initial_query: str, query: str, summaries: list[dict], 
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
        score = score_summary(initial_query, query, content, additional_context, llm_model, language)
        results.append({
            "summary": s,
            "score": score,
            "original_index": idx
        })
    # sort highest score first
    return sorted(results, key=lambda x: x["score"], reverse=True)


def score_summary(initial_query: str, query: str, content: str, context: str, 
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
Return ONLY a number between 0 and 10 using the following ranges:
- 10 = perfectly relevant and accurate
- 9-8 = very relevant with strong detail
- 7-6 = relevant but somewhat incomplete
- 5-4 = partially relevant
- 3-0 = poorly relevant or inaccurate

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
            return 1.0
    except Exception as e:
        print(f"  [ERROR] Failed to score summary: {str(e)}")
        return 1.0

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
        
        # Get internet result from state
        internet_result = state.get("internet_result", "")
        
        # Generate the enhanced prompt using reranked summaries
        final_answer_prompt = _generate_final_answer_prompt(
            initial_query=user_query,
            reranked_summaries=all_reranked_summaries,
            additional_context=additional_context,
            language=detected_language,
            internet_result=internet_result
        )
        
        print(f"  [INFO] Generating final answer using {report_llm} with {len(all_reranked_summaries)} prioritized summaries")
        
        # Try structured output first, fallback to manual JSON parsing
        try:
            print(f"  [INFO] Attempting structured output with {report_llm}")
            
            # Try using structured output with Pydantic model
            structured_result = invoke_ollama(
                system_prompt=f"You are an expert assistant providing comprehensive answers. Respond in {detected_language}. Your response must be a JSON object with a single 'content' key containing the markdown-formatted report.",
                user_prompt=final_answer_prompt,
                model=report_llm,
                output_format=FinalReportOutput
            )
            
            # Extract content from structured output
            final_answer = structured_result.content
            print(f"  [INFO] Structured output successful (length: {len(final_answer)})")
            
        except Exception as e:
            print(f"  [WARNING] Structured output failed: {str(e)}. Falling back to manual JSON parsing.")
            
            # Fallback: Use strict JSON instructions in system prompt
            json_system_prompt = f"""You are an expert assistant providing comprehensive answers. Respond in {detected_language}.
            
IMPORTANT: Your response MUST be a valid JSON object with exactly this structure:
            {{
                "content": "your markdown-formatted report here"
            }}
            
Do not include any text outside the JSON object. The 'content' field should contain the complete report in markdown format."""
            
            raw_response = invoke_ollama(
                system_prompt=json_system_prompt,
                user_prompt=final_answer_prompt,
                model=report_llm
            )
            
            # Parse JSON manually
            try:
                import json
                # Clean the response - remove any markdown code blocks if present
                cleaned_response = raw_response.strip()
                if cleaned_response.startswith('```json'):
                    cleaned_response = cleaned_response[7:]
                if cleaned_response.endswith('```'):
                    cleaned_response = cleaned_response[:-3]
                cleaned_response = cleaned_response.strip()
                
                parsed_json = json.loads(cleaned_response)
                final_answer = parsed_json.get('content', raw_response)
                print(f"  [INFO] Manual JSON parsing successful (length: {len(final_answer)})")
                
            except json.JSONDecodeError as json_error:
                print(f"  [WARNING] JSON parsing failed: {str(json_error)}. Using raw response.")
                # Last resort: use raw response
                final_answer = raw_response
        
        print(f"  [INFO] Final answer generated successfully (length: {len(final_answer)})")
        
        return {"final_answer": final_answer}
        
    except Exception as e:
        print(f"  [ERROR] Exception in enhanced generate_final_answer: {str(e)}")
        return {"final_answer": f"Error occurred during final answer generation: {str(e)}. Check logs for details."}


def _generate_final_answer_prompt(initial_query: str, reranked_summaries: list[dict], 
                                 additional_context: str = "", language: str = "English", internet_result: str = "") -> str:
    """
    Create a prompt for generating the final answer using reranked summaries.
    
    Args:
        initial_query: The original user question
        reranked_summaries: List of summaries sorted by relevance score (highest first)
        additional_context: Conversation history or domain context
        language: Detected language for the response
        internet_result: Internet search results to include in the prompt
    
    Returns:
        Formatted prompt string for the LLM
    """
    
    prompt = f"""# ROLE
You are an expert assistant providing deep and extensive answer based on ranked document summaries.

# GOAL
Generate a detailled, complete, deep and extensive answer to the user's query using the provided summaries. Prioritize information from higher-ranked summaries.

# CONTEXT from research:
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

    # Add internet results section if available
    if internet_result and internet_result.strip():
        prompt += f"""

INTERNET SOURCES:
{internet_result}
"""

    prompt += f"""

# CONSTRAINTS:
• Base your answer PRIMARILY on the highest-ranked summary as it is most relevant to the query
• Focus on directly answering the original query 
• Include exact levels, figures, numbers, statistics, and quantitative data ONLY from the source material
• Maintain precision by using direct quotes for key definitions and important statements
• For citations, ALWAYS use the EXACT format ['Source'] after each fact.
• Use supporting sources to add context, details, or complementary information
• If internet sources are available, incorporate recent/current information to complement the supporting sources. Then, clearly indicate when information comes from recent web searches. Do this by adding a short separate section called "Internet Sources" and cite the sources URLs.
• If supporting sources or internet sources contradict the primary source, prioritize the primary source unless there's clear evidence of error
• If information is incomplete, acknowledge limitations
• Respond in {language} language"""

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

def source_linker(state: ResearcherStateV2, config: RunnableConfig = None) -> ResearcherStateV2:
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
        print(f"  [DEBUG] Final answer preview: {final_answer[:200]}...")
        
        if not final_answer:
            print("  [WARNING] No final answer found in state")
            return {**state, "linked_final_answer": ""}
        
        # Check for source references in the final answer
        import re
        source_pattern = re.compile(r'\[([^[\]]+?\.pdf)\]')
        source_matches = source_pattern.findall(final_answer)
        print(f"  [DEBUG] Found {len(source_matches)} source references: {source_matches}")
        
        if not source_matches:
            print("  [INFO] No source references found in final answer, returning as-is")
            return {**state, "linked_final_answer": final_answer}
        
        # Convert source references to clickable links
        print(f"  [DEBUG] Converting {len(source_matches)} source references to clickable links...")
        linked_answer = linkify_sources(final_answer, selected_database)
        
        print(f"  [DEBUG] Linked answer length: {len(linked_answer)} characters")
        print(f"  [DEBUG] Linked answer preview: {linked_answer[:300]}...")
        
        # Check if linking actually happened
        if linked_answer != final_answer:
            print("  [SUCCESS] Source linking completed - content was modified")
        else:
            print("  [WARNING] Source linking completed but content was not modified")
        
        # Return updated state with linked final answer
        return {**state, "linked_final_answer": linked_answer}
        
    except Exception as e:
        print(f"  [ERROR] Source linking failed: {str(e)}")
        # Fallback: return original final answer as linked answer
        return {**state, "linked_final_answer": state.get("final_answer", "")}

def update_position(state: ResearcherStateV2):
    """Wrapper for update_position_v1 to work with extended state."""
    return update_position_v1(state)

def quality_router_with_source_linker(state: ResearcherStateV2) -> str:
    """
    Router function that determines whether to improve the report or proceed to source linking.
    
    Args:
        state: ResearcherStateV2 containing quality check results
    
    Returns:
        str: Next node name ("generate_final_answer" for improvement or "source_linker" to finish)
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
        print(f"  [QUALITY ROUTER] Quality needs improvement (reflection {reflection_count + 1}/{MAX_REFLECTIONS}) -> generate_final_answer")
        return "generate_final_answer"
    
    # Fallback: proceed to source linker to prevent infinite loops
    print("  [QUALITY ROUTER] Fallback decision -> source_linker")
    return "source_linker"

def quality_router(state: ResearcherStateV2):
    """Wrapper for quality_router_v1 to work with extended state."""
    return quality_router_v1(state)

def conditional_quality_router(state: ResearcherStateV2):
    """
    Router that checks if quality checking is enabled before routing to quality checker.
    If quality checking is disabled, goes directly to end.
    """
    # Check if quality checking is enabled in the state
    enable_quality_checker = state.get("enable_quality_checker", False)
    
    if enable_quality_checker:
        print("  [INFO] Quality checker enabled, routing to quality_checker")
        return "quality_checker"
    else:
        print("  [INFO] Quality checker disabled, routing to end")
        return "end"

# Create simplified main researcher graph
def create_main_graph():
    """
    Create the simplified main researcher graph that starts with retrieve_rag_documents.
    This graph is designed to work after the HITL workflow has completed and generated research_queries.
    The main workflow skips the detect_language and generate_research_queries nodes since those
    steps are already handled in the HITL workflow.
    
    Note: Source linking functionality has been moved to the rerank-reporter graph used in Phase 3.
    
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
            "end": END  # Skip quality checker if disabled, go directly to END
        }
    )
    
    # Quality checker routes to END or back to generate_final_answer
    main_workflow.add_conditional_edges(
        "quality_checker",
        quality_router,
        {
            "generate_final_answer": "generate_final_answer",  # Loop back for improvement
            "end": END  # Quality passed, end the workflow
        }
    )
    
    return main_workflow.compile()

# Create the graphs
hitl_graph = create_hitl_graph()
main_graph = create_main_graph()


# Export the main function for use in the app
researcher_main = main_graph
