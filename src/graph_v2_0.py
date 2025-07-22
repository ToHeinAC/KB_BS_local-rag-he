import datetime
import os
import pathlib
import operator
from typing_extensions import Literal, Annotated
from langgraph.constants import Send
from langgraph.graph import START, END, StateGraph
from langchain_core.runnables.config import RunnableConfig
from langchain_core.documents import Document
from src.configuration_v1_1 import Configuration, get_config_instance
from src.vector_db_v1_1 import get_or_create_vector_db, search_documents, get_embedding_model_path
from src.state_v2_0 import ResearcherStateV2, InitState  # Extended state with HITL fields
from src.prompts_v1_1 import (
    # Language detection prompts
    LANGUAGE_DETECTOR_SYSTEM_PROMPT, LANGUAGE_DETECTOR_HUMAN_PROMPT,
    # Research query generation prompts
    RESEARCH_QUERY_WRITER_SYSTEM_PROMPT, RESEARCH_QUERY_WRITER_HUMAN_PROMPT,
    # Document summarization prompts
    SUMMARIZER_SYSTEM_PROMPT, SUMMARIZER_HUMAN_PROMPT,
    # Report writing prompts
    REPORT_WRITER_SYSTEM_PROMPT, REPORT_WRITER_HUMAN_PROMPT,
)
from src.utils_v1_1 import format_documents_with_metadata, invoke_ollama, parse_output, tavily_search, DetectedLanguage, Queries
from src.rag_helpers_v1_1 import source_summarizer_ollama, format_documents_as_plain_text, parse_document_to_formatted_content
import re
import time

# Get the directory path of the current file
this_path = os.path.dirname(os.path.abspath(__file__))

def extract_embedding_model_from_db_name(db_dir_name):
    """
    Extract the embedding model name from the database directory name.
    Fixed version that properly handles the database naming convention.
    
    Args:
        db_dir_name (str): The database directory name
        
    Returns:
        str: The extracted embedding model name
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

# Define state mappings for the transition from HITL to main workflow
def create_hitl_graph():
    """Create the modified HITL workflow graph without detect_language node."""
    # Create the HITL graph with InitState
    workflow = StateGraph(InitState)
    
    # Add HITL nodes (excluding detect_language)
    workflow.add_node("analyse_user_feedback", analyse_user_feedback)
    workflow.add_node("generate_follow_up_questions", generate_follow_up_questions)
    workflow.add_node("generate_knowledge_base_questions", generate_knowledge_base_questions)
    
    # Add edges for HITL workflow
    workflow.add_edge(START, "analyse_user_feedback")
    workflow.add_edge("analyse_user_feedback", "generate_follow_up_questions")
    workflow.add_edge("generate_follow_up_questions", "generate_knowledge_base_questions")
    workflow.add_edge("generate_knowledge_base_questions", END)
    
    # Compile the HITL graph
    return workflow.compile()

# Initialize the researcher graph with extended state
researcher_graph_v2 = StateGraph(ResearcherStateV2)

# Import existing nodes from graph_v1_1 and adapt them
from src.graph_v1_1 import (
    detect_language as detect_language_v1,
    display_embedding_model_info as display_embedding_model_info_v1,
    generate_research_queries as generate_research_queries_v1,
    retrieve_rag_documents as retrieve_rag_documents_v1,
    summarize_query_research as summarize_query_research_v1,
    generate_final_answer as generate_final_answer_v1,
    quality_checker as quality_checker_v1,
    query_router as query_router_v1,
    update_position as update_position_v1,
    quality_router as quality_router_v1
)

# HITL Node functions adapted from basic_HITL_app.py
def analyse_user_feedback(state: InitState, config: RunnableConfig):
    """
    Analyze user feedback in the context of the research workflow.
    
    Args:
        state (InitState): The current state.
        config (RunnableConfig): The configuration for the graph.
    
    Returns:
        dict: Updated state with analysis and additional context.
    """
    print("--- Analyzing user feedback ---")
    
    query = state["user_query"]
    detected_language = state.get("detected_language", "English")  # Default to English if not set
    human_feedback = state.get("human_feedback", "")
    additional_context = state.get("additional_context", "")
    
    # Use the configured LLM model from state or config
    model_to_use = state.get("report_llm", config["configurable"].get("report_llm", "deepseek-r1:latest"))
    
    # Skip analysis if no feedback provided
    if not human_feedback.strip():
        print("No human feedback provided, skipping analysis")
        return {
            "analysis": "No human feedback provided for analysis.",
            "additional_context": additional_context,
            "current_position": "analyse_user_feedback"
        }
    
    # Format system prompt for analysis
    system_prompt = f"""# ROLE
You are an expert domain analysis specialist for human-in-the-loop research conversations.

# GOAL
Analyze the human's feedback to identify the specific technical or specialized domain context that will help improve research query generation.

# AVAILABLE INFORMATION
- Initial user query: Available for context
- Conversation history: {additional_context if additional_context else "None yet"}
- Latest human feedback: Will be provided in the user prompt
- Detected language: {detected_language}

# ANALYSIS TASK
1. Identify the technical/specialized domain (e.g., "Nuclear safety and regulatory compliance", "Software engineering", "Medical diagnostics")
2. Determine the specific sub-area or context within that domain
3. Note any technical terminology or concepts that indicate expertise level

# OUTPUT FORMAT
Provide a concise domain analysis in 1-3 sentences.
Example: "This relates to nuclear waste management and regulatory compliance, specifically focusing on radioactive residue disposal protocols."

# CRITICAL CONSTRAINTS
- Respond EXCLUSIVELY in {detected_language} language
- Provide ONLY the domain analysis, no prefixes or explanations
- Do NOT return JSON, dictionaries, or structured data
- Maximum 3 sentences
- Focus on technical/domain context, not linguistic aspects
    """
    
    # Format the human prompt with context from previous interactions
    human_prompt = f"""# CONVERSATION CONTEXT
Initial Query: {query}

# CONVERSATION HISTORY
{additional_context if additional_context else "No previous conversation history."}

# LATEST HUMAN FEEDBACK TO ANALYZE
{human_feedback}

# DOMAIN ANALYSIS
Based on the above information, provide your domain analysis in {detected_language}:"""
    
    # Using local model with Ollama
    result = invoke_ollama(
        model=model_to_use,
        system_prompt=system_prompt,
        user_prompt=human_prompt,
    )
    
    # Parse the result
    parsed_result = parse_output(result)
    
    # Update additional_context with the new feedback and analysis
    if additional_context:
        additional_context += "\n\n"
    additional_context += f"Human Feedback: {human_feedback}\nAI Analysis: {parsed_result['response']}"
    
    return {
        "analysis": parsed_result["response"], 
        "additional_context": additional_context, 
        "current_position": "analyse_user_feedback"
    }

def generate_follow_up_questions(state: InitState, config: RunnableConfig):
    """
    Generate follow-up questions based on the current state and analysis.
    
    Args:
        state (InitState): The current state.
        config (RunnableConfig): The configuration for the graph.
    
    Returns:
        dict: Updated state with follow-up questions.
    """
    print("--- Generating follow-up questions ---")
    
    query = state["user_query"]
    detected_language = state.get("detected_language", "English")
    analysis = state.get("analysis", "")
    additional_context = state.get("additional_context", "")
    report_llm = state.get("report_llm", config["configurable"].get("report_llm", "deepseek-r1:latest"))
    summarization_llm = state.get("summarization_llm", config["configurable"].get("summarization_llm", "deepseek-r1:latest"))
    
    # Use the configured LLM model from state or config
    model_to_use = state.get("report_llm", config["configurable"].get("report_llm", "deepseek-r1:latest"))
    
    # Format system prompt for question generation
    system_prompt = f"""# ROLE
You are an expert information gathering specialist for technical consultations.

# GOAL
Generate 1-3 strategic clarifying questions to deepen understanding of the user's specific needs within their domain.

# AVAILABLE INFORMATION
- Initial user query: Available for context
- Conversation history: {additional_context if additional_context else "None yet"}
- Latest domain analysis: {analysis if analysis else "Not yet available"}
- Detected language: {detected_language}

# QUESTION GENERATION STRATEGY
1. Build upon the domain analysis to ask more specific technical questions
2. Focus on clarifying technical requirements, constraints, or specifications
3. Avoid repeating information already provided by the human
4. Progress from general domain understanding to specific implementation details
5. Ask questions that will help generate better knowledge base search queries

# OUTPUT FORMAT
Generate 1-3 questions in numbered markdown format:
1. [First specific question]
2. [Second specific question]
3. [Third specific question]

# CRITICAL CONSTRAINTS
- Write EXCLUSIVELY in {detected_language} language - NO EXCEPTIONS
- Every single word must be in {detected_language}
- Focus on technical/domain-specific aspects, not linguistic clarification
- Do NOT repeat previously asked questions
- Do NOT return JSON, dictionaries, or structured data
- Provide ONLY the numbered questions, no additional text
- Maximum 3 questions, minimum 1 question
    """
    
    # Format the human prompt with context from previous interactions
    human_prompt = f"""# CONVERSATION CONTEXT
Initial Query: {query}

# DOMAIN ANALYSIS
{analysis if analysis else "Domain analysis not yet available."}

# CONVERSATION HISTORY
{additional_context if additional_context else "No previous conversation history."}

# TASK
Based on the above context, generate 1-3 NEW clarifying questions in {detected_language} that will help you better understand the user's specific technical needs:"""
    
    # Using local model with Ollama
    result = invoke_ollama(
        model=model_to_use,
        system_prompt=system_prompt,
        user_prompt=human_prompt,
    )
    
    # Parse the result
    parsed_result = parse_output(result)
    
    # Update additional_context with the generated questions
    if additional_context:
        additional_context += "\n"
    additional_context += f"AI Follow-up Questions: {parsed_result['response']}"
    
    return {
        "follow_up_questions": parsed_result["response"], 
        "additional_context": additional_context, 
        "current_position": "generate_follow_up_questions"
    }

def generate_knowledge_base_questions(state: InitState, config: RunnableConfig):
    """
    Generate knowledge base questions based on the completed interaction.
    The final node in the HITL graph that will transition to the main workflow.
    FIXED: Now outputs research_queries as list[str] for proper handover.
    
    Args:
        state (InitState): The current state with all human feedback.
        config (RunnableConfig): The configuration for the graph.
    
    Returns:
        dict: Updated state with research_queries list for main workflow transition.
    """
    print("--- Generating knowledge base questions and transitioning to main workflow ---")
    
    query = state["user_query"]
    detected_language = state.get("detected_language", "English")
    additional_context = state.get("additional_context", "")
    report_llm = state.get("report_llm", config["configurable"].get("report_llm", "deepseek-r1:latest"))
    summarization_llm = state.get("summarization_llm", config["configurable"].get("summarization_llm", "deepseek-r1:latest"))
    
    # Use the configured LLM model from state or config
    model_to_use = state.get("report_llm", config["configurable"].get("report_llm", "deepseek-r1:latest"))
    
    # Format system prompt for KB question generation
    system_prompt = f"""# ROLE
You are an expert knowledge base search query specialist.

# GOAL
Generate 5 highly targeted, searchable questions optimized for knowledge base retrieval based on the completed human-in-the-loop conversation.

# AVAILABLE INFORMATION
- Initial user query: Available for context
- Complete conversation history: {additional_context if additional_context else "Limited conversation history"}
- Detected language: {detected_language}
- Domain context: Extracted from conversation analysis

# SEARCH QUERY OPTIMIZATION STRATEGY
1. Use specific technical terminology likely to match knowledge base content
2. Focus on different aspects of the user's information need
3. Frame as search queries, not conversational questions
4. Cover both broad concepts and specific implementation details
5. Avoid redundancy between questions
6. Include relevant keywords and domain-specific terms
7. Consider different search angles (what, how, why, when, where)

# OUTPUT FORMAT
Generate exactly 5 questions in numbered markdown format:
1. [First targeted search question]
2. [Second targeted search question]
3. [Third targeted search question]
4. [Fourth targeted search question]
5. [Fifth targeted search question]

# CRITICAL CONSTRAINTS
- You MUST Write EXCLUSIVELY in {detected_language} language, both your prefix and your questions - NO EXCEPTIONS
- Focus on technical/domain-specific search terms
- Phrase as search queries optimized for knowledge retrieval
- Do NOT return JSON, dictionaries, or structured data
- Provide ONLY the numbered questions, no additional text
- Exactly 5 questions required, formulated as full questions
    """
    
    # Format the human prompt with context from all interactions
    human_prompt = f"""# CONVERSATION SUMMARY
Initial Query: {query}

# COMPLETE CONVERSATION HISTORY
{additional_context if additional_context else "No detailed conversation history available."}

# TASK
Based on the complete conversation above, generate 5 targeted knowledge base search questions in {detected_language} that will help retrieve the most relevant information:"""
    
    # Using local model with Ollama
    result = invoke_ollama(
        model=model_to_use,
        system_prompt=system_prompt,
        user_prompt=human_prompt,
    )
    
    # Parse the result
    parsed_result = parse_output(result)
    kb_questions_text = parsed_result["response"]
    
    # Parse the numbered list into a list of strings
    research_queries = []
    lines = kb_questions_text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line and (line[0].isdigit() or line.startswith('-') or line.startswith('*')):
            # Remove numbering and clean up
            clean_query = re.sub(r'^\d+\.\s*', '', line)  # Remove "1. "
            clean_query = re.sub(r'^[-*]\s*', '', clean_query)  # Remove "- " or "* "
            clean_query = clean_query.strip()
            if clean_query:
                research_queries.append(clean_query)
    
    # Ensure we have at least some queries
    if not research_queries:
        # Fallback: create basic queries from the original query
        research_queries = [
            query,
            f"How to {query}",
            f"What is {query}",
            f"Examples of {query}",
            f"Best practices for {query}"
        ]
    
    # Limit to 5 queries maximum
    research_queries = research_queries[:5]
    
    # Update additional_context with the generated KB questions
    if additional_context:
        additional_context += "\n\n"
    additional_context += f"AI Knowledge Base Questions:\n{kb_questions_text}"
    
    # Return updated state for HITL completion with research_queries for main workflow
    return {
        "user_query": query,
        "detected_language": detected_language,
        "research_queries": research_queries,  # This is the key fix - output as list[str]
        "additional_context": additional_context,
        "report_llm": report_llm,
        "summarization_llm": summarization_llm,
        "current_position": "hitl_complete"
    }

# Wrapper functions to adapt v1 nodes to v2 state
def detect_language(state: ResearcherStateV2, config: RunnableConfig):
    """Wrapper for detect_language_v1 to work with extended state."""
    return detect_language_v1(state, config)

def display_embedding_model_info(state: ResearcherStateV2):
    """
    Wrapper for display_embedding_model_info_v1 to work with extended state.
    FIXED: Now properly extracts embedding model from database name.
    """
    print("--- Displaying embedding model information ---")
    
    # Get configuration
    config = get_config_instance()
    
    # Check if external database is being used
    use_ext_database = config.use_ext_database if hasattr(config, 'use_ext_database') else False
    selected_database = config.selected_database if hasattr(config, 'selected_database') else None
    
    if use_ext_database and selected_database:
        # Extract embedding model from database name using fixed function
        embedding_model = extract_embedding_model_from_db_name(selected_database)
        print(f"Using external database: {selected_database}")
        print(f"Extracted embedding model: {embedding_model}")
        
        # Return state with embedding model info
        return {
            "current_position": "display_embedding_model_info",
            "additional_context": state.get("additional_context", "") + f"\nUsing embedding model: {embedding_model} from database: {selected_database}"
        }
    else:
        # Use default embedding model info
        return display_embedding_model_info_v1(state)

# Wrapper functions for other nodes to handle additional_context

# Wrapper functions for other nodes to handle additional_context
def generate_research_queries(state: ResearcherStateV2, config: RunnableConfig):
    """
    Enhanced research query generation that considers additional context from HITL.
    FIXED: Now uses research_queries from HITL if available.
    """
    print("--- Generating research queries ---")
    
    # Check if research_queries already exist from HITL
    existing_queries = state.get("research_queries", [])
    if existing_queries:
        print(f"Using research queries from HITL: {len(existing_queries)} queries")
        return {
            "research_queries": existing_queries,
            "current_position": "generate_research_queries"
        }
    
    # Otherwise, use the original function with additional context
    additional_context = state.get("additional_context", "")
    if additional_context:
        # Modify the state to include additional context in a way the original function can use
        enhanced_state = dict(state)
        enhanced_state["human_feedback"] = additional_context
        return generate_research_queries_v1(enhanced_state, config)
    else:
        return generate_research_queries_v1(state, config)

def summarize_query_research(state: ResearcherStateV2, config: RunnableConfig):
    """
    Enhanced summarization that considers additional context from HITL.
    """
    print("--- Summarizing query research (with HITL context) ---")
    
    # Get additional context for enhanced summarization
    additional_context = state.get("additional_context", "")
    
    if additional_context:
        # Enhance the summarization process with HITL context
        # We'll modify the config to include additional context instructions
        enhanced_config = dict(config)
        if "configurable" not in enhanced_config:
            enhanced_config["configurable"] = {}
        enhanced_config["configurable"]["additional_context"] = additional_context
        return summarize_query_research_v1(state, enhanced_config)
    else:
        return summarize_query_research_v1(state, config)

def generate_final_answer(state: ResearcherStateV2, config: RunnableConfig):
    """
    Enhanced final answer generation that considers additional context from HITL.
    """
    print("--- Generating final answer (with HITL context) ---")
    
    # Get additional context for enhanced report generation
    additional_context = state.get("additional_context", "")
    
    if additional_context:
        # Enhance the final answer generation with HITL context
        enhanced_config = dict(config)
        if "configurable" not in enhanced_config:
            enhanced_config["configurable"] = {}
        enhanced_config["configurable"]["additional_context"] = additional_context
        return generate_final_answer_v1(state, enhanced_config)
    else:
        return generate_final_answer_v1(state, config)

# Wrapper functions for remaining nodes
def retrieve_rag_documents(state: ResearcherStateV2, config: RunnableConfig):
    """Wrapper for retrieve_rag_documents_v1 to work with extended state."""
    return retrieve_rag_documents_v1(state, config)

def quality_checker(state: ResearcherStateV2, config: RunnableConfig):
    """
    Enhanced quality checker that considers additional context from HITL.
    """
    print("--- Running quality check (with HITL context) ---")
    
    # Get additional context for enhanced quality checking
    additional_context = state.get("additional_context", "")
    
    # If we have additional context, enhance the quality check
    original_query = state["user_query"]
    final_answer = state.get("final_answer", "")
    
    if additional_context and final_answer:
        # Create a temporary state with enhanced input for quality checking
        temp_state = dict(state)
        temp_state["quality_context"] = f"Additional Context from Human Feedback:\n{additional_context}\n\nOriginal Query: {original_query}\n\nFinal Answer: {final_answer}"
        return quality_checker_v1(temp_state, config)
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

# Define main researcher nodes (NO HITL nodes - they are in separate graph)
researcher_graph_v2.add_node(display_embedding_model_info)
researcher_graph_v2.add_node(detect_language)
researcher_graph_v2.add_node(generate_research_queries)
researcher_graph_v2.add_node(retrieve_rag_documents)
researcher_graph_v2.add_node(summarize_query_research)
researcher_graph_v2.add_node(generate_final_answer)
researcher_graph_v2.add_node(quality_checker)
researcher_graph_v2.add_node(query_router)
researcher_graph_v2.add_node(update_position)

# Define transitions for the main workflow (starts after HITL completes)
researcher_graph_v2.add_edge(START, "display_embedding_model_info")
researcher_graph_v2.add_edge("display_embedding_model_info", "detect_language")
researcher_graph_v2.add_edge("detect_language", "generate_research_queries")  # Direct to research after HITL
researcher_graph_v2.add_edge("generate_research_queries", "query_router")

# Conditional routing for multiple queries
researcher_graph_v2.add_conditional_edges(
    "query_router",
    lambda state: "retrieve_rag_documents" if state["research_queries"] else END,
    {
        "retrieve_rag_documents": "retrieve_rag_documents",
        END: END
    }
)

researcher_graph_v2.add_edge("retrieve_rag_documents", "update_position")
researcher_graph_v2.add_edge("update_position", "summarize_query_research")
researcher_graph_v2.add_edge("summarize_query_research", "generate_final_answer")

# Quality checker routing
researcher_graph_v2.add_conditional_edges(
    "generate_final_answer",
    quality_router,
    {
        "quality_checker": "quality_checker",
        END: END
    }
)

researcher_graph_v2.add_conditional_edges(
    "quality_checker",
    quality_router,
    {
        "generate_final_answer": "generate_final_answer",
        END: END
    }
)

# Compile the main researcher graph (this will be invoked as child via Command.PARENT)
researcher_main = researcher_graph_v2.compile()

# Create the integrated workflow with proper HITL to main research transition
def create_integrated_researcher_v2():
    """
    Create the integrated researcher v2.0 with HITL as initial step and main research as follow-up.
    Uses proper LangGraph parent-child pattern with Command.PARENT for state transition.
    """
    # Create the main integrated workflow graph
    integrated_workflow = StateGraph(ResearcherStateV2)
    
    # Add HITL nodes that work with ResearcherStateV2 (they'll handle InitState internally)
    integrated_workflow.add_node("analyse_user_feedback", analyse_user_feedback_v2)
    integrated_workflow.add_node("generate_follow_up_questions", generate_follow_up_questions_v2)
    integrated_workflow.add_node("generate_knowledge_base_questions", generate_knowledge_base_questions_v2)
    
    # Add main research nodes
    integrated_workflow.add_node("display_embedding_model_info", display_embedding_model_info)
    integrated_workflow.add_node("detect_language", detect_language)
    integrated_workflow.add_node("generate_research_queries", generate_research_queries)
    integrated_workflow.add_node("retrieve_rag_documents", retrieve_rag_documents)
    integrated_workflow.add_node("summarize_query_research", summarize_query_research)
    integrated_workflow.add_node("generate_final_answer", generate_final_answer)
    integrated_workflow.add_node("quality_checker", quality_checker)
    integrated_workflow.add_node("query_router", query_router)
    integrated_workflow.add_node("update_position", update_position)
    
    # Define workflow: HITL first, then main research
    integrated_workflow.add_edge(START, "analyse_user_feedback")
    integrated_workflow.add_edge("analyse_user_feedback", "generate_follow_up_questions")
    integrated_workflow.add_edge("generate_follow_up_questions", "generate_knowledge_base_questions")
    integrated_workflow.add_edge("generate_knowledge_base_questions", "display_embedding_model_info")
    integrated_workflow.add_edge("display_embedding_model_info", "detect_language")
    integrated_workflow.add_edge("detect_language", "generate_research_queries")
    integrated_workflow.add_edge("generate_research_queries", "retrieve_rag_documents")
    
    # Continue with main workflow - direct flow without conditional routing for now
    # integrated_workflow.add_conditional_edges(
    #     "query_router",
    #     lambda state: "retrieve_rag_documents" if state.get("research_queries", []) else END,
    #     {
    #         "retrieve_rag_documents": "retrieve_rag_documents",
    #         END: END
    #     }
    # )
    
    integrated_workflow.add_edge("retrieve_rag_documents", "update_position")
    integrated_workflow.add_edge("update_position", "summarize_query_research")
    integrated_workflow.add_edge("summarize_query_research", "generate_final_answer")
    
    # Quality checker routing
    integrated_workflow.add_conditional_edges(
        "generate_final_answer",
        quality_router,
        {
            "quality_checker": "quality_checker",
            END: END
        }
    )
    
    integrated_workflow.add_conditional_edges(
        "quality_checker",
        quality_router,
        {
            "generate_final_answer": "generate_final_answer",
            END: END
        }
    )
    
    return integrated_workflow.compile()

# Create wrapper functions for HITL nodes to work with ResearcherStateV2
def analyse_user_feedback_v2(state: ResearcherStateV2, config: RunnableConfig):
    """Wrapper to adapt HITL analyse_user_feedback to ResearcherStateV2."""
    # Convert ResearcherStateV2 to InitState for HITL processing
    init_state = {
        "user_query": state["user_query"],
        "detected_language": state.get("detected_language", "English"),
        "additional_context": state.get("additional_context", ""),
        "report_llm": state.get("report_llm", "deepseek-r1:latest"),
        "summarization_llm": state.get("summarization_llm", "deepseek-r1:latest"),
        "current_position": "analyse_user_feedback"
    }
    result = analyse_user_feedback(init_state, config)
    # Update ResearcherStateV2 with results
    return {
        "additional_context": result.get("additional_context", state.get("additional_context", "")),
        "current_position": "analyse_user_feedback"
    }

def generate_follow_up_questions_v2(state: ResearcherStateV2, config: RunnableConfig):
    """Wrapper to adapt HITL generate_follow_up_questions to ResearcherStateV2."""
    init_state = {
        "user_query": state["user_query"],
        "detected_language": state.get("detected_language", "English"),
        "additional_context": state.get("additional_context", ""),
        "report_llm": state.get("report_llm", "deepseek-r1:latest"),
        "summarization_llm": state.get("summarization_llm", "deepseek-r1:latest"),
        "current_position": "generate_follow_up_questions"
    }
    result = generate_follow_up_questions(init_state, config)
    return {
        "additional_context": result.get("additional_context", state.get("additional_context", "")),
        "current_position": "generate_follow_up_questions"
    }

def generate_knowledge_base_questions_v2(state: ResearcherStateV2, config: RunnableConfig):
    """Wrapper to adapt HITL generate_knowledge_base_questions to ResearcherStateV2."""
    init_state = {
        "user_query": state["user_query"],
        "detected_language": state.get("detected_language", "English"),
        "additional_context": state.get("additional_context", ""),
        "report_llm": state.get("report_llm", "deepseek-r1:latest"),
        "summarization_llm": state.get("summarization_llm", "deepseek-r1:latest"),
        "current_position": "generate_knowledge_base_questions"
    }
    result = generate_knowledge_base_questions(init_state, config)
    # Extract the state updates from the result dictionary
    return {
        "additional_context": result.get("additional_context", state.get("additional_context", "")),
        "detected_language": result.get("detected_language", state.get("detected_language", "English")),
        "current_position": "hitl_complete"
    }

# Create the integrated researcher v2 workflow
researcher_v2 = create_integrated_researcher_v2()
