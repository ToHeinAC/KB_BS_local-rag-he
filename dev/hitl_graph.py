import datetime
import os
import sys
from typing import TypedDict, Dict, List, Any, Annotated, Optional
from typing_extensions import Literal

# Import LangGraph components
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt

# Import LangChain components
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Add parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project components
from src.state_v1_1 import InitState
from src.configuration_v1_1 import Configuration, get_config_instance
from src.utils_v1_1 import invoke_ollama, parse_output, DetectedLanguage


def display_embedding_model_info(state: InitState):
    """Display information about which embedding model is being used."""
    config = get_config_instance()
    embedding_model = config.embedding_model
    print(f"\n=== Using embedding model: {embedding_model} ===\n")
    return {"embedding_model": embedding_model}


def detect_language(state: InitState):
    """
    Detect language of user query.
    
    Args:
        state (InitState): The current state.
    
    Returns:
        dict: A state update containing the detected language.
    """
    print("--- Detecting language of user query ---")
    query = state["user_query"]
    
    # Use the report writer LLM for language detection
    model_to_use = state.get("report_llm", "deepseek-r1:latest")
    
    # Format the system prompt for language detection
    system_prompt = """You are a language detection assistant. Your task is to identify the language of the given text.
    Respond with the language name in English (e.g., "English", "German", "Spanish", "French", "Chinese", etc.).
    Only provide the language name, nothing else."""
    
    # Format the human prompt
    human_prompt = f"Identify the language of the following text:\n\n{query}"
    
    # Using local model with Ollama
    result = invoke_ollama(
        model=model_to_use,
        system_prompt=system_prompt,
        user_prompt=human_prompt,
    )
    
    # Parse the result to get just the language name
    if isinstance(result, dict) and "language" in result:
        detected_language = result["language"]
    else:
        # If we received a string directly, use it as the language
        detected_language = str(result).strip()
    
    print(f"Detected language: {detected_language}")
    
    return {"detected_language": detected_language}


def generate_ai_questions(state: InitState):
    """
    Generate AI questions for human feedback.
    
    Args:
        state (InitState): The current state.
    
    Returns:
        dict: A state update containing the AI message with questions in structured format.
    """
    print("--- Generating AI questions for human feedback ---")
    query = state["user_query"]
    detected_language = state["detected_language"]
    human_feedback = state.get("human_feedback", [])
    
    # Use the report LLM for generating questions
    model_to_use = state.get("report_llm", "deepseek-r1:latest")
    
    print(f"  [DEBUG] Using model: {model_to_use}")
    
    # Format system prompt for question generation with structured output
    system_prompt = f"""You are an AI assistant helping to gather information from a human to answer their query.
    Your task is to ask clarifying questions that will help you better understand what they're looking for.
    
    If there have been previous exchanges, build upon that information rather than asking the same questions again.

    Ask 1-3 specific questions about their query to gather more context. Be concise and focused. Explore the phase space of possible contexts.
    You MUST phrase your questions in {detected_language}.
    
    IMPORTANT: Format your response in JSON with exactly 1-3 questions in this format:
    {{
        "Q1": "[Your first question]",
        "Q2": "[Your second question]",
        "Q3": "[Your third question]"
    }}
    
    Only include questions (no explanations or other text). If you have fewer than 3 questions, only include those (e.g., just Q1 and Q2).
    """
    
    # Build context from previous interactions if they exist
    context = ""
    if human_feedback:
        context = "Previous feedback:\n"
        for i, feedback in enumerate(human_feedback):
            context += f"- Feedback {i+1}: {feedback}\n"
    
    # Format human prompt
    human_prompt = f"""Initial query: {query}
    
    {context}
    
    Based on this information, what clarifying questions should I ask? Remember to format as Q1, Q2, Q3."""
    
    # Use the configured model
    ai_response = invoke_ollama(
        model=model_to_use,
        system_prompt=system_prompt,
        user_prompt=human_prompt
    )
    
    # Get the message content
    raw_questions = ai_response if isinstance(ai_response, str) else str(ai_response)
    print(f"Raw AI questions: {raw_questions}")
    
    # Parse the questions into a structured format
    structured_questions = []
    lines = raw_questions.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if line.startswith('Q') and ':' in line:
            # Extract the question text after the colon
            question_text = line.split(':', 1)[1].strip()
            structured_questions.append(question_text)
    
    # If parsing failed, use the raw text
    if not structured_questions:
        structured_questions = [raw_questions]
    
    # Create a formatted display string for the UI
    formatted_display = ""
    for i, question in enumerate(structured_questions):
        formatted_display += f"Q{i+1}: {question}\n"
    
    print(f"Structured questions: {structured_questions}")
    print(f"Formatted display: {formatted_display}")
    
    # Return both the structured questions and the formatted display
    return {
        "ai_message": formatted_display,
        "structured_questions": structured_questions
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
    print(f"[human_feedback_node] Current state: {state}")
    
    # Check if we need to generate questions first
    if "ai_message" not in state or not state["ai_message"]:
        print("[human_feedback_node] Warning: No AI message found in state. Make sure generate_ai_questions ran before this node.")
    
    # Display AI message and get user feedback - use the actual AI-generated questions
    ai_message = state.get("ai_message", "")
    
    # Get structured questions if available
    structured_questions = state.get("structured_questions", [])
    
    # If we still don't have an AI message, create a default one
    if not ai_message and structured_questions:
        # Build a message from structured questions
        ai_message = "\n".join([f"Q{i+1}: {q}" for i, q in enumerate(structured_questions)])
    elif not ai_message:
        ai_message = "Please provide additional context for your query."
    
    print(f"[human_feedback_node] AI message: {ai_message}")
    print(f"[human_feedback_node] Structured questions: {structured_questions}")
    
    # Only pass the necessary data to the interrupt, not the full state
    # This follows LangGraph best practices
    user_feedback = interrupt({
        "task": "Please provide your response to the questions, or type '/end' to finish the feedback loop.",
        "ai_message": ai_message,
        "structured_questions": structured_questions,
        "user_query": state["user_query"]
    })
    
    print(f"[human_feedback_node] Received human feedback: {user_feedback}")
    
    # Check if user wants to end the feedback loop
    # user_feedback could be a string or a dict depending on how the app handles the interrupt
    user_response = user_feedback
    if isinstance(user_feedback, dict) and "user_response" in user_feedback:
        user_response = user_feedback["user_response"]
    
    if isinstance(user_response, str) and user_response.lower() == "/end":
        # Create a final summary before ending
        # Only update the human_feedback field, not the entire state
        human_feedback_list = state.get("human_feedback", [])
        updated_feedback = human_feedback_list + [user_response]
        return Command(update={"human_feedback": updated_feedback}, goto="summarize_feedback")
    
    # Otherwise update feedback and generate more questions
    # Store the actual user response in the human_feedback list
    human_feedback_list = state.get("human_feedback", [])
    
    # Handle different response formats
    if isinstance(user_feedback, dict) and "user_response" in user_feedback:
        response_to_add = user_feedback["user_response"]
    else:
        response_to_add = str(user_feedback)
    
    updated_feedback = human_feedback_list + [response_to_add]
    
    # If we have collected enough feedback (3+ rounds), go to summarization
    if len(updated_feedback) >= 3:
        return Command(update={"human_feedback": updated_feedback}, goto="summarize_feedback")
    
    # Otherwise continue the conversation
    return Command(update={"human_feedback": updated_feedback}, goto="generate_ai_questions")


def summarize_feedback(state: InitState):
    """
    Summarize the human-AI interaction to create a comprehensive briefing.
    
    Args:
        state (InitState): The current state.
    
    Returns:
        dict: A state update containing the additional context from human feedback.
    """
    print("--- Summarizing human feedback ---")
    query = state["user_query"]
    detected_language = state["detected_language"]
    human_feedback = state.get("human_feedback", [])
    ai_message = state.get("ai_message", "")
    
    # Use the report LLM for summarization
    model_to_use = state.get("report_llm", "deepseek-r1:latest")
    
    # Format system prompt for summarization
    system_prompt = f"""You are an AI assistant tasked with summarizing a conversation between a human and an AI.
    
    The conversation began with an initial query, followed by clarifying questions from the AI and responses from the human.
    
    Your task is to create a concise summary that captures all relevant information from this exchange.
    IMPORTANT: The summary MUST be in {detected_language} and focus on the key points that were discussed.
    """
    
    # Format the conversation for summarization
    conversation = f"Original Query: {query}\n\nAI Questions:\n{ai_message}\n\nHuman Feedback:\n"
    for i, feedback in enumerate(human_feedback):
        conversation += f"- Response {i+1}: {feedback}\n"
    
    # Format human prompt
    human_prompt = f"""Here is the conversation to summarize:
    
    {conversation}
    
    Create a comprehensive summary that includes all important details and context.
    """
    
    # Use the configured model
    summary_response = invoke_ollama(
        model=model_to_use,
        system_prompt=system_prompt,
        user_prompt=human_prompt
    )
    
    summary = summary_response if isinstance(summary_response, str) else str(summary_response)
    print(f"Feedback summary: {summary}")
    
    # Create list for the additional context
    additional_context = [summary]
    
    # Return the summary as additional context
    return {"additional_context": additional_context, "feedback_summary": summary}


def prepare_final_response(state: InitState):
    """
    Generate the final response based on the initial query and all human feedback.
    
    Args:
        state (InitState): The current state.
    
    Returns:
        dict: A state update containing the final answer.
    """
    print("--- Generating final response ---")
    query = state["user_query"]
    detected_language = state["detected_language"]
    additional_context = state.get("additional_context", [])
    
    # Use the report LLM for final response
    model_to_use = state.get("report_llm", "deepseek-r1:latest")
    
    # Format system prompt for final response
    system_prompt = f"""You are an AI assistant helping to answer user queries.
    Based on the initial query and additional context from the human-AI interaction,
    generate a comprehensive and helpful response.
    
    IMPORTANT: Your response MUST be in {detected_language} and address all aspects of the user's query.
    """
    
    # Combine additional context
    context_text = "\n\n".join(additional_context) if additional_context else ""
    
    # Format human prompt
    human_prompt = f"""Initial query: {query}
    
    Additional context from conversation:
    {context_text}
    
    Please generate a comprehensive response to the user's query.
    """
    
    # Use the configured model
    final_response = invoke_ollama(
        model=model_to_use,
        system_prompt=system_prompt,
        user_prompt=human_prompt
    )
    
    response_text = final_response if isinstance(final_response, str) else str(final_response)
    print(f"Final response generated.")
    
    # Return the final response
    return {"final_answer": response_text}


# Create the graph with StateGraph
def create_hitl_graph():
    """Create and configure the HITL workflow graph."""
    # Initialize the graph with InitState
    workflow = StateGraph(InitState)
    
    # Add nodes to the graph
    workflow.add_node("display_embedding_model", display_embedding_model_info)
    workflow.add_node("detect_language", detect_language)
    workflow.add_node("generate_ai_questions", generate_ai_questions)
    workflow.add_node("human_feedback", human_feedback_node)
    workflow.add_node("summarize_feedback", summarize_feedback)
    workflow.add_node("prepare_final_response", prepare_final_response)
    
    # Define edges
    workflow.add_edge(START, "display_embedding_model")
    workflow.add_edge("display_embedding_model", "detect_language")
    workflow.add_edge("detect_language", "generate_ai_questions")
    workflow.add_edge("generate_ai_questions", "human_feedback")
    workflow.add_edge("human_feedback", "generate_ai_questions")
    workflow.add_edge("human_feedback", "summarize_feedback")
    workflow.add_edge("summarize_feedback", "prepare_final_response")
    workflow.add_edge("prepare_final_response", END)
    
    # Compile the graph
    hitl_app = workflow.compile(checkpointer=MemorySaver())
    
    return hitl_app


# Create the graph instance
hitl_app = create_hitl_graph()
