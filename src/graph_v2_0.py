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
    
    # Format the system prompt using the same pattern as generate_research_queries
    system_prompt = RESEARCH_QUERY_WRITER_SYSTEM_PROMPT.format(
        max_queries=max_queries,
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
    research_queries.insert(0, query)
    print(f"  [DEBUG] Generated knowledge base queries: {research_queries}")
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
    additional_context = state.get("additional_context", "")
    if additional_context:
        enhanced_state = dict(state)
        enhanced_state["human_feedback"] = additional_context
        return summarize_query_research_v1(enhanced_state, config)
    else:
        return summarize_query_research_v1(state, config)

def generate_final_answer(state: ResearcherStateV2, config: RunnableConfig):
    """Enhanced final answer generation that considers additional context from HITL."""
    additional_context = state.get("additional_context", "")
    if additional_context:
        enhanced_state = dict(state)
        enhanced_state["human_feedback"] = additional_context
        return generate_final_answer_v1(enhanced_state, config)
    else:
        return generate_final_answer_v1(state, config)

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
    4. generate_final_answer: Generates the final answer based on summaries
    5. quality_checker (optional): Checks the quality of the final answer
    
    Returns:
        Compiled StateGraph: The compiled main workflow graph
    """
    main_workflow = StateGraph(ResearcherStateV2)
    
    # Add nodes for main workflow starting with retrieve_rag_documents
    main_workflow.add_node("retrieve_rag_documents", retrieve_rag_documents)
    main_workflow.add_node("summarize_query_research", summarize_query_research)
    main_workflow.add_node("generate_final_answer", generate_final_answer)
    main_workflow.add_node("quality_checker", quality_checker)
    main_workflow.add_node("update_position", update_position)
    
    # Main workflow edges starting with retrieve_rag_documents
    main_workflow.add_edge(START, "retrieve_rag_documents")
    main_workflow.add_edge("retrieve_rag_documents", "update_position")
    main_workflow.add_edge("update_position", "summarize_query_research")
    main_workflow.add_edge("summarize_query_research", "generate_final_answer")
    
    # Quality checker routing
    main_workflow.add_conditional_edges(
        "generate_final_answer",
        quality_router,
        {
            "quality_checker": "quality_checker",
            "end": END
        }
    )
    
    main_workflow.add_conditional_edges(
        "quality_checker",
        query_router,
        {
            "continue": "update_position",
            "end": END
        }
    )
    
    return main_workflow.compile()

# Create the graphs
hitl_graph = create_hitl_graph()
main_graph = create_main_graph()


# Export the main function for use in the app
researcher_main = main_graph
