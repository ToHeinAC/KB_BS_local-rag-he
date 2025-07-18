import datetime
import os
import uuid
from typing import TypedDict, Dict, List, Any, Annotated, Optional
from typing_extensions import Literal

from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt
from langgraph.constants import Send
from langchain_core.runnables.config import RunnableConfig
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from src.configuration_v1_1 import Configuration, get_config_instance
from src.vector_db_v1_1 import get_or_create_vector_db, search_documents, get_embedding_model_path
from src.state_v1_1 import ResearcherState, InitState
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

# Get the directory path of the current file
this_path = os.path.dirname(os.path.abspath(__file__))

# Initialize the human feedback briefing graph
briefing_graph = StateGraph(InitState)

# Initialize the researcher graph
researcher_graph = StateGraph(ResearcherState)

#############################
# Human Feedback Briefing Graph
#############################

def display_embedding_model_info_init(state: InitState):
    """Display information about which embedding model is being used."""
    config = get_config_instance()
    embedding_model = config.embedding_model
    print(f"\n=== Using embedding model: {embedding_model} ===\n")
    # Return a dictionary with a key for embedding_model - LangGraph nodes must return dictionaries
    return {"embedding_model": embedding_model, "current_step": "display_embedding_model_info"}

def detect_language_init(state: InitState, config: RunnableConfig):
    """
    Detect language of user query for the initialization phase.
    
    Args:
        state (InitState): The current state.
        config (RunnableConfig): The configuration for the graph.
    
    Returns:
        dict: A state update containing the detected language.
    """
    print("--- Detecting language of user query ---")
    query = state["user_query"]  # Get the query from user_query
    # Use the report writer LLM for language detection
    llm_model = config["configurable"].get("report_llm", "qwq")
    
    # First check if a language is already set in the config (from GUI)
    user_selected_language = config["configurable"].get("selected_language", None)
    
    if user_selected_language:
        print(f"Using user-selected language: {user_selected_language}")
        return {"detected_language": user_selected_language}
    
    # If no language is set in config, detect it from the query
    print("No language selected by user, detecting from query...")
    
    # Format the system prompt
    system_prompt = LANGUAGE_DETECTOR_SYSTEM_PROMPT
    
    # Format the human prompt
    human_prompt = LANGUAGE_DETECTOR_HUMAN_PROMPT.format(
        query=query
    )
    
    # Check if report_llm is in state and use it directly if available
    # This ensures we use the model selected in the UI
    if "report_llm" in state:
        model_to_use = state["report_llm"]
        print(f"  [DEBUG] Using model from state: {model_to_use}")
    else:
        model_to_use = llm_model
        print(f"  [DEBUG] Using model from config: {model_to_use}")
        
    # Using local model with Ollama
    result = invoke_ollama(
        model=model_to_use,
        system_prompt=system_prompt,
        user_prompt=human_prompt,
        output_format=DetectedLanguage
    )
    
    detected_language = result.language
    print(f"Detected language: {detected_language}")
    
    return {"detected_language": detected_language}

def generate_ai_questions(state: InitState, config: RunnableConfig):
    """
    Generate questions for human feedback based on the initial query and language.
    
    Args:
        state (InitState): The current state.
        config (RunnableConfig): Configuration for the graph.
    
    Returns:
        dict: A state update containing the AI message with questions.
    """
    print("--- Generating AI questions for human feedback ---")
    query = state["user_query"]
    detected_language = state["detected_language"]
    
    # Use the report LLM for generating questions
    if "report_llm" in state:
        model_to_use = state["report_llm"]
    else:
        model_to_use = config["configurable"].get("report_llm", "qwq")
    
    print(f"  [DEBUG] Using model: {model_to_use}")
    
    # Use prompts from prompts_v1_1.py
    from src.prompts_v1_1 import AI_QUESTION_GENERATOR_SYSTEM_PROMPT, AI_QUESTION_GENERATOR_HUMAN_PROMPT
    
    # Format system prompt for question generation
    system_prompt = AI_QUESTION_GENERATOR_SYSTEM_PROMPT.format(language=detected_language)
    
    # Format human prompt
    human_prompt = AI_QUESTION_GENERATOR_HUMAN_PROMPT.format(query=query)
    
    # Use the configured model
    ai_response = invoke_ollama(
        model=model_to_use,
        system_prompt=system_prompt,
        user_prompt=human_prompt
    )
    
    # Get the raw message
    raw_message = ai_response if isinstance(ai_response, str) else ai_response.content if hasattr(ai_response, 'content') else str(ai_response)
    print(f"AI questions: {raw_message}")
    
    # Extract only the questions part, removing any thinking process
    # First, check if the response follows the <think>...</think> format
    import re
    
    # Parse the AI response - strip thinking process if present
    structured_questions = raw_message
    
    # Try to remove <think>...</think> blocks if they exist
    think_pattern = re.compile(r'<think>.*?</think>', re.DOTALL)
    structured_questions = re.sub(think_pattern, '', structured_questions).strip()
    
    # If no questions found after removing thinking, fall back to the original response
    if not structured_questions or len(structured_questions) < 10:  # Arbitrary check to ensure we have enough content
        structured_questions = raw_message
    
    # Store both raw message (for debugging) and structured questions
    return {
        "ai_message": structured_questions,
        "ai_raw_message": raw_message  # Keep the raw message for reference
    }

def human_feedback_node(state: InitState):
    """
    Human feedback node that interrupts execution to get user feedback.
    
    Args:
        state (InitState): The current state.
    
    Returns:
        Command: A command to update state and determine next step.
    """
    print("\n[human_feedback_node] Awaiting human feedback...")
    
    # Display AI message and get user feedback
    ai_message = state.get("ai_message", "Please provide additional context for your query.")
    
    # Interrupt to get user feedback
    user_feedback = interrupt(
        {
            "ai_message": ai_message,
            "user_query": state["user_query"],
            "message": "Please provide your response to the questions, or type 'done' to finish the feedback loop."
        }
    )
    
    print(f"[human_feedback_node] Received human feedback: {user_feedback}")
    
    # Check if user wants to end the feedback loop
    if user_feedback.lower() == "done":
        # Create a final summary before ending
        return Command(update={"human_feedback": state.get("human_feedback", []) + [user_feedback]}, goto="summarize_feedback")
    
    # Otherwise update feedback and generate more questions or go to summary
    human_feedback_list = state.get("human_feedback", [])
    updated_feedback = human_feedback_list + [user_feedback]
    
    # If we have at least 2 rounds of feedback, go to summarization
    if len(updated_feedback) >= 2:
        return Command(update={"human_feedback": updated_feedback}, goto="summarize_feedback")
    
    # Otherwise continue the conversation
    return Command(update={"human_feedback": updated_feedback}, goto="generate_ai_questions")

def summarize_feedback(state: InitState, config: RunnableConfig):
    """
    Summarize the human-AI interaction to create a comprehensive briefing.
    
    Args:
        state (InitState): The current state.
        config (RunnableConfig): Configuration for the graph.
    
    Returns:
        dict: A state update containing the additional context from human feedback.
    """
    print("--- Summarizing human feedback ---")
    query = state["user_query"]
    detected_language = state["detected_language"]
    human_feedback = state.get("human_feedback", [])
    ai_message = state.get("ai_message", "")
    
    # Use the report LLM for summarization
    if "report_llm" in state:
        model_to_use = state["report_llm"]
    else:
        model_to_use = config["configurable"].get("report_llm", "qwq")
    
    print(f"  [DEBUG] Using model: {model_to_use}")
    
    # Use prompts from prompts_v1_1.py
    from src.prompts_v1_1 import HUMAN_FEEDBACK_SUMMARIZER_SYSTEM_PROMPT, HUMAN_FEEDBACK_SUMMARIZER_HUMAN_PROMPT
    
    # Format system prompt for summarization
    system_prompt = HUMAN_FEEDBACK_SUMMARIZER_SYSTEM_PROMPT.format(language=detected_language)
    
    # Format the conversation for summarization
    conversation = f"Original Query: {query}\n\nAI Questions:\n{ai_message}\n\nHuman Feedback:\n"
    for i, feedback in enumerate(human_feedback):
        conversation += f"- Response {i+1}: {feedback}\n"
    
    # Format human prompt
    human_prompt = HUMAN_FEEDBACK_SUMMARIZER_HUMAN_PROMPT.format(conversation=conversation)
    
    # Use the configured model
    summary_response = invoke_ollama(
        model=model_to_use,
        system_prompt=system_prompt,
        user_prompt=human_prompt
    )
    
    summary = summary_response if isinstance(summary_response, str) else summary_response.content if hasattr(summary_response, 'content') else str(summary_response)
    print(f"Feedback summary: {summary}")
    
    # Create Document objects for the additional context
    additional_context = [
        Document(
            page_content=summary,
            metadata={
                "source": "human_feedback_summary",
                "date": datetime.datetime.now().strftime("%Y-%m-%d"),
                "language": detected_language
            }
        )
    ]
    
    # Return the summary as additional context
    return {"additional_context": additional_context, "feedback_summary": summary}

def prepare_researcher_state(state: InitState):
    """
    Prepare the state to be passed to the researcher graph.
    
    Args:
        state (InitState): The current state.
    
    Returns:
        dict: The final state to be passed to the researcher graph.
    """
    print("--- Preparing researcher state ---")
    
    # Return the final state for the next graph
    return {
        "user_query": state["user_query"],
        "current_position": state.get("current_position", 0),
        "detected_language": state["detected_language"],
        "additional_context": state.get("additional_context", []),
        "report_llm": state.get("report_llm", "qwq"),
        "summarization_llm": state.get("summarization_llm", "llama3.2")
    }

#############################
# Researcher Graph Functions 
#############################

def display_embedding_model_info(state: ResearcherState):
    """Display information about which embedding model is being used."""
    config = get_config_instance()
    embedding_model = config.embedding_model
    print(f"\n=== Using embedding model: {embedding_model} ===\n")
    # Return a dictionary with a key for embedding_model - LangGraph nodes must return dictionaries
    return {"embedding_model": embedding_model, "current_step": "display_embedding_model_info"}

def detect_language(state: ResearcherState, config: RunnableConfig):
    """
    Detect language of user query. In this modified flow, we already have the language from the briefing phase.
    
    Args:
        state (ResearcherState): The current state of the researcher.
        config (RunnableConfig): The configuration for the graph.
    
    Returns:
        dict: A state update containing the detected language.
    """
    print("--- Detect language node (researcher graph) ---")
    
    # The language should already be detected in the briefing phase
    if "detected_language" in state and state["detected_language"]:
        print(f"Using detected language from briefing phase: {state['detected_language']}")
        return {"detected_language": state["detected_language"]}
    
    # Fallback to detecting language if somehow missing
    query = state["user_query"]  # Get the query from user_query
    # Use the report writer LLM for language detection
    llm_model = config["configurable"].get("report_llm", "qwq")
    
    # First check if a language is already set in the config (from GUI)
    user_selected_language = config["configurable"].get("selected_language", None)
    
    if user_selected_language:
        print(f"Using user-selected language: {user_selected_language}")
        return {"detected_language": user_selected_language}
    
    # If no language is set in config, detect it from the query
    print("No language selected by user, detecting from query...")
    
    # Format the system prompt
    system_prompt = LANGUAGE_DETECTOR_SYSTEM_PROMPT
    
    # Format the human prompt
    human_prompt = LANGUAGE_DETECTOR_HUMAN_PROMPT.format(
        query=query
    )
    
    # Check if report_llm is in state and use it directly if available
    # This ensures we use the model selected in the UI
    if "report_llm" in state:
        model_to_use = state["report_llm"]
        print(f"  [DEBUG] Using model from state: {model_to_use}")
    else:
        model_to_use = llm_model
        print(f"  [DEBUG] Using model from config: {model_to_use}")
        
    # Using local model with Ollama
    result = invoke_ollama(
        model=model_to_use,
        system_prompt=system_prompt,
        user_prompt=human_prompt,
        output_format=DetectedLanguage
    )
    
    detected_language = result.language
    print(f"Detected language: {detected_language}")
    
    return {"detected_language": detected_language}

def generate_research_queries(state: ResearcherState, config: RunnableConfig):
    """
    Generate research queries based on the user's instructions and additional context from human feedback.

    Args:
        state (ResearcherState): The current state of the researcher.
        config (RunnableConfig): The configuration for the graph.

    Returns:
        dict: A state update containing the generated research queries as a list.
    """
    print("--- Generating research queries ---")
    query = state["user_query"]  # Get the query from user_query
    detected_language = state["detected_language"]
    max_queries = config["configurable"].get("max_search_queries", 3)
    # Use the report writer LLM for generating research queries
    llm_model = config["configurable"].get("report_llm", "qwq")
    print(f"  [DEBUG] Research Query LLM (report_llm): {llm_model}")
    
    # Get additional context from human feedback if available
    additional_context_docs = state.get("additional_context", [])
    additional_context_text = ""
    
    if additional_context_docs:
        additional_context_text = "\n\n".join([doc.page_content for doc in additional_context_docs])
        print(f"  [DEBUG] Using additional context from human feedback: {len(additional_context_text)} characters")
    else:
        print("  [DEBUG] No additional context available from human feedback")
    
    # Format the system prompt
    system_prompt = RESEARCH_QUERY_WRITER_SYSTEM_PROMPT.format(
        max_queries=max_queries,
        date=datetime.datetime.now().strftime("%Y/%m/%d %H:%M"),
        language=detected_language
    )
    
    # Format the human prompt with the additional context
    human_prompt = RESEARCH_QUERY_WRITER_HUMAN_PROMPT.format(
        query=query,
        language=detected_language,
        additional_context=f"Consider this additional context when generating queries: {additional_context_text}" if additional_context_text else ""
    )
    
    # Check if report_llm is in state and use it directly if available
    # This ensures we use the model selected in the UI
    if "report_llm" in state:
        model_to_use = state["report_llm"]
        print(f"  [DEBUG] Using model from state: {model_to_use}")
    else:
        model_to_use = llm_model
        print(f"  [DEBUG] Using model from config: {model_to_use}")
        
    # Using local llm model with Ollama
    result = invoke_ollama(
        model=model_to_use,
        system_prompt=system_prompt,
        user_prompt=human_prompt,
        output_format=Queries
    )
    
    all_queries = result.queries
    all_queries.insert(0, query)
    print(f"  [DEBUG] Generated research queries: {all_queries}")
    assert isinstance(all_queries, list), "all_queries must be a list"
    
    return {"research_queries": all_queries}

# Import the other functions from graph_v1_1.py here
# For this implementation, we'll reuse the existing functions for retrieve_rag_documents, summarize_query_research, etc.
from src.graph_v1_1 import (
    retrieve_rag_documents, 
    summarize_query_research,
    generate_final_answer,
    quality_checker,
    query_router,
    update_position,
    quality_router
)

#############################
# Graph Definitions
#############################

# Human Feedback Briefing Graph
briefing_graph.add_node("display_embedding_model_info", display_embedding_model_info_init)
briefing_graph.add_node("detect_language", detect_language_init)
briefing_graph.add_node("generate_ai_questions", generate_ai_questions)
briefing_graph.add_node("human_feedback_node", human_feedback_node)
briefing_graph.add_node("summarize_feedback", summarize_feedback)
briefing_graph.add_node("prepare_researcher_state", prepare_researcher_state)

# Define transitions for the human feedback briefing graph
briefing_graph.add_edge(START, "display_embedding_model_info")
briefing_graph.add_edge("display_embedding_model_info", "detect_language")
briefing_graph.add_edge("detect_language", "generate_ai_questions")
briefing_graph.add_edge("generate_ai_questions", "human_feedback_node")
briefing_graph.add_edge("human_feedback_node", "generate_ai_questions")
briefing_graph.add_edge("human_feedback_node", "summarize_feedback")
briefing_graph.add_edge("summarize_feedback", "prepare_researcher_state")
briefing_graph.add_edge("prepare_researcher_state", END)

# Set up the briefing graph with checkpointing for interrupts
briefing_checkpointer = MemorySaver()
briefing_app = briefing_graph.compile(checkpointer=briefing_checkpointer)

# Researcher Graph
researcher_graph.add_node("detect_language", detect_language)
researcher_graph.add_node("generate_research_queries", generate_research_queries)
researcher_graph.add_node("retrieve_rag_documents", retrieve_rag_documents)
researcher_graph.add_node("summarize_query_research", summarize_query_research)
researcher_graph.add_node("generate_final_answer", generate_final_answer)
researcher_graph.add_node("quality_checker", quality_checker)

# Define transitions for the researcher graph
researcher_graph.add_edge(START, "detect_language")
researcher_graph.add_edge("detect_language", "generate_research_queries")
researcher_graph.add_edge("generate_research_queries", "retrieve_rag_documents")
researcher_graph.add_edge("retrieve_rag_documents", "summarize_query_research")
researcher_graph.add_edge("summarize_query_research", "generate_final_answer")

# Add conditional transitions based on quality checking
researcher_graph.add_conditional_edges(
    "generate_final_answer",
    quality_router,
    {
        "quality_checker": "quality_checker",
        END: END
    }
)

researcher_graph.add_edge("quality_checker", "generate_final_answer")

# Set up the researcher graph
researcher_checkpointer = MemorySaver()
researcher_app = researcher_graph.compile(checkpointer=researcher_checkpointer)

# Combined workflow function
def run_combined_workflow(initial_state: InitState, config: Dict[str, Any]):
    """
    Run the combined workflow with human feedback followed by research.
    
    Args:
        initial_state (InitState): The initial state for the workflow.
        config (Dict[str, Any]): Configuration for the workflow.
    
    Returns:
        Final output from the researcher graph.
    """
    # Set up thread config with a unique thread ID
    thread_config = {"configurable": config}
    thread_config["configurable"]["thread_id"] = str(uuid.uuid4())
    
    # Initialize the workflow with the provided state
    print("Starting human feedback briefing workflow...")
    
    # Results to collect briefing outputs
    briefing_results = None
    
    # Run the briefing graph with interrupt handling
    for chunk in briefing_app.stream(initial_state, config=thread_config):
        for node_id, value in chunk.items():
            # Handle interrupts to get user input
            if node_id == "__interrupt__":
                while True:
                    # Display the AI's questions to the user
                    ai_message = value.get("ai_message", "Please provide additional context.")
                    user_query = value.get("user_query", "")
                    print(f"\nOriginal Query: {user_query}")
                    print(f"\nAI Questions: {ai_message}")
                    
                    user_feedback = input("\nYour response (type 'done' to finish): ")
                    
                    # Resume execution with the user's feedback
                    briefing_results = briefing_app.invoke(Command(resume=user_feedback), config=thread_config)
                    
                    # Exit loop if user says done
                    if user_feedback.lower() == "done":
                        break
            elif node_id == END:
                # Capture final results
                briefing_results = value
    
    # Initialize researcher state from briefing output
    if briefing_results and "prepare_researcher_state" in briefing_results:
        researcher_initial_state = briefing_results["prepare_researcher_state"]
    else:
        print("Warning: Briefing results incomplete. Using initial state for research.")
        researcher_initial_state = {
            "user_query": initial_state["user_query"],
            "detected_language": initial_state.get("detected_language", "English"),
            "current_position": 0,
            "research_queries": [],
            "retrieved_documents": {},
            "search_summaries": {},
            "final_answer": "",
            "report_llm": initial_state.get("report_llm", "qwq"),
            "summarization_llm": initial_state.get("summarization_llm", "llama3.2"),
        }
    
    # Run researcher workflow
    print("Starting research workflow...")
    researcher_results = researcher_app.invoke(researcher_initial_state, config=thread_config)
    
    return researcher_results
