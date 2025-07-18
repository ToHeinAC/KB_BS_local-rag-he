import streamlit as st
import os
import sys
import time
import json
from typing import TypedDict, Dict, List, Any, Annotated, Optional
from typing_extensions import Literal

# Add parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import LangGraph components
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Import project components
from src.state_v1_1 import InitState
from src.utils_v1_1 import invoke_ollama, parse_output, get_configured_llm_model, DetectedLanguage

# Configure Streamlit page
st.set_page_config(
    page_title="Human-in-the-Loop Assistant",
    page_icon="🔄",
    layout="wide"
)

# Custom CSS for better UI visibility
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .conversation-container {
        border: 1px solid #888888;
        border-radius: 10px;
        padding: 15px;
        background-color: #f0f2f6;
        margin-bottom: 20px;
        max-height: 400px;
        overflow-y: auto;
    }
    .user-message {
        background-color: #aedff7;
        color: #000000;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        box-shadow: 1px 1px 3px rgba(0,0,0,0.2);
    }
    .ai-message {
        background-color: #d4d4d6;
        color: #000000;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        box-shadow: 1px 1px 3px rgba(0,0,0,0.2);
    }
    .workflow-step {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 5px;
        font-weight: bold;
    }
    .active-step {
        background-color: #28a745;
        color: white;
    }
    .completed-step {
        background-color: #4a5568;
        color: white;
    }
    .pending-step {
        background-color: #e2e8f0;
        color: #4a5568;
        border: 1px solid #cbd5e0;
    }
    .stButton>button {
        background-color: #4e8cff;
        color: white;
        font-weight: bold;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        border: 1px solid #f5c6cb;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        border: 1px solid #c3e6cb;
    }
</style>
""", unsafe_allow_html=True)

# Function to display the conversation history
def display_conversation(history):
    st.markdown('<div class="conversation-container">', unsafe_allow_html=True)
    for message in history:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message"><strong>User:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        else:
            # Format AI messages with markdown
            st.markdown(f'<div class="ai-message"><strong>AI:</strong> {message["content"]}</div>', unsafe_allow_html=True)

def detect_language(query: str, model: str) -> str:
    """
    Detect language of user query.
    
    Args:
        query (str): The user query.
        model (str): The LLM model to use.
    
    Returns:
        str: The detected language.
    """
    # Format the system prompt for language detection
    system_prompt = """You are a language detection assistant. Your task is to identify the language of the given text.
    Respond with the language name in English (e.g., "English", "German", "Spanish", "French", "Chinese", etc.).
    Only provide the language name, nothing else. Be precise and accurate."""
    
    # Format the human prompt
    human_prompt = f"Identify the language of the following text:\n\n{query}"
    
    # Using local model with Ollama
    result = invoke_ollama(
        model=model,
        system_prompt=system_prompt,
        user_prompt=human_prompt,
    )
    
    # Parse the result to get just the language name
    if isinstance(result, dict) and "language" in result:
        detected_language = result["language"]
    else:
        # If we received a string directly, use it as the language
        detected_language = str(result).strip()
    
    # Log the detected language
    print(f"Detected language: {detected_language}")
    
    return detected_language

def generate_follow_up_questions(state: InitState) -> str:
    """
    Generate follow-up questions based on the current state.
    
    Args:
        state (InitState): The current state.
    
    Returns:
        str: Generated follow-up questions.
    """
    query = state["user_query"]
    detected_language = state["detected_language"]
    human_feedback = state.get("human_feedback", [])
    additional_context = state.get("additional_context", [])
    
    # Use the configured LLM model
    model_to_use = state.get("report_llm", "deepseek-r1:latest")
    
    # Format system prompt for question generation
    system_prompt = f"""You are an AI assistant helping to gather information from a human to answer their query.
    Your task is to ask 1-3 clarifying questions that will help you better understand what they're looking for.
    
    EXTREMELY IMPORTANT: You MUST respond ONLY in {detected_language} language. Do not switch to any other language.
    
    Focus on the DOMAIN CONTEXT of the conversation - these should be technical/domain-specific questions
    related to the user's query, not questions about language or clarification of terms.
    The DOMAIN CONTEXT is: {additional_context}
    Make use of the DOMAIN CONTEXT to ask more specific and relevant questions.
    
    Format your response in markdown as follows:
    1. First question
    2. Second question
    3. Third question
    
    Only include questions that are truly necessary for understanding the user's needs better.
    DO NOT repeat questions that have already been asked and answered.
    Build upon the information you've already received to ask more specific and relevant questions.

    ONLY provide the questions, nothing else (no prefix, no explanation, no additional text).
    """
    
    # Format the human prompt with context from previous interactions
    human_prompt = f"Initial Query: {query}\n\n"
    
    # Include the full conversation history from additional_context
    if additional_context:
        human_prompt += "Conversation history:\n"
        for i, context in enumerate(additional_context):
            human_prompt += f"{context}\n\n"
    
    human_prompt += f"\nBased on this information, what 1-3 NEW clarifying questions should you ask? DO NOT repeat previous questions. REMEMBER: Respond ONLY in {detected_language} language and focus on understanding the DOMAIN CONTEXT, not linguistic clarification."
    
    # Using local model with Ollama
    result = invoke_ollama(
        model=model_to_use,
        system_prompt=system_prompt,
        user_prompt=human_prompt,
    )
    
    # Parse the result
    parsed_result = parse_output(result)
    return parsed_result["response"]

def generate_knowledge_base_questions(state: InitState) -> str:
    """
    Generate knowledge base questions based on the completed interaction.
    
    Args:
        state (InitState): The current state with all human feedback.
    
    Returns:
        str: Generated knowledge base questions.
    """
    query = state["user_query"]
    detected_language = state["detected_language"]
    human_feedback = state.get("human_feedback", [])
    additional_context = state.get("additional_context", [])
    
    # Use the configured LLM model
    model_to_use = state.get("report_llm", "deepseek-r1:latest")
    
    # Format system prompt for KB question generation
    system_prompt = f"""You are an AI assistant tasked with generating targeted questions for a knowledge base search.
    Based on the initial user query and subsequent conversation, generate 5 specific questions that would help retrieve the most relevant information from a knowledge base.
    
    These questions should:
    1. Be specific and focused
    2. Cover different aspects of the user's information need
    3. Use terminology likely to match knowledge base content
    4. Avoid redundancy
    5. Be phrased as search queries, not conversational questions
    
    EXTREMELY IMPORTANT: You MUST respond ONLY in {detected_language} language. Do not switch to any other language.
    
    Focus on the DOMAIN CONTEXT of the conversation - these should be technical/domain-specific questions
    related to the user's query, not questions about language or clarification of terms.
    The DOMAIN CONTEXT is: {additional_context}
    
    Format your response in markdown as follows:
    1. First knowledge base question
    2. Second knowledge base question
    3. Third knowledge base question
    4. Fourth knowledge base question
    5. Fifth knowledge base question

    ONLY provide the questions, nothing else (no prefix, no explanation, no additional text).
    """
    
    # Format the human prompt with context from all interactions
    human_prompt = f"Initial Query: {query}\n\n"
    
    if human_feedback:
        human_prompt += "Conversation history:\n"
        for i, feedback in enumerate(human_feedback):
            human_prompt += f"Exchange {i+1}: {feedback}\n"
    
    human_prompt += "\nBased on this information, generate 5 targeted knowledge base search questions:"
    
    # Using local model with Ollama
    result = invoke_ollama(
        model=model_to_use,
        system_prompt=system_prompt,
        user_prompt=human_prompt,
    )
    
    # Parse the result
    parsed_result = parse_output(result)
    return parsed_result["response"]

def create_hitl_graph():
    """
    Create the HITL workflow graph.
    
    Returns:
        StateGraph: The configured workflow graph.
    """
    # Create the graph
    workflow = StateGraph(InitState)
    
    # Add nodes
    workflow.add_node("detect_language", detect_language)
    workflow.add_node("generate_follow_up_questions", generate_follow_up_questions)
    workflow.add_node("generate_knowledge_base_questions", generate_knowledge_base_questions)
    
    # Add edges
    workflow.add_edge(START, "detect_language")
    workflow.add_edge("detect_language", "generate_follow_up_questions")
    workflow.add_edge("generate_follow_up_questions", END)
    
    # Compile the graph
    return workflow.compile()

def main():
    st.title("Human-in-the-Loop AI Assistant")
    st.markdown("""
    This application demonstrates a human-in-the-loop approach where the AI asks clarifying questions 
    to better understand your needs before providing final knowledge base search questions.
    
    Type `/end` at any point to finish the conversation and generate knowledge base search questions.
    """)
    
    # Initialize session state variables if they don't exist
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    
    if "state" not in st.session_state:
        st.session_state.state = None
    
    if "waiting_for_human_input" not in st.session_state:
        st.session_state.waiting_for_human_input = False
    
    if "conversation_ended" not in st.session_state:
        st.session_state.conversation_ended = False
    
    if "kb_questions_generated" not in st.session_state:
        st.session_state.kb_questions_generated = False
    
    # Model selection
    available_models = ["qwq", "deepseek-r1:latest", "deepseek-r1:70b", "gemma3:27b", "mistral-small:latest", 
                 "deepseek-r1:1.5b", "llama3.1:8b-instruct-q4_0", "llama3.2", "llama3.3", "llama3.3:70b-instruct-q4_K_M", "gemma3:4b", "phi4-mini", 
                 "mistral:instruct", "mistrallite", "qwen3:30b-a3b"]
    selected_model = st.selectbox("Select LLM Model", available_models, index=1)
    
    # Initial query input
    if not st.session_state.state:
        user_query = st.text_area("Enter your initial query:", height=100)
        submit_button = st.button("Submit")
        
        if submit_button and user_query:
            # Initialize the state
            st.session_state.state = {
                "user_query": user_query,
                "current_position": 0,
                "detected_language": "",
                "additional_context": [],  # Will store annotated conversation history
                "human_feedback": [],
                "report_llm": selected_model,
                "summarization_llm": selected_model
            }
            
            # Add user message to conversation history
            st.session_state.conversation_history.append({
                "role": "user",
                "content": user_query
            })
            
            # Create the workflow graph
            hitl_graph = create_hitl_graph()
            
            # Detect language
            with st.spinner("Detecting language..."):
                detected_language = detect_language(user_query, selected_model)
                st.session_state.state["detected_language"] = detected_language
            
            # Generate initial follow-up questions
            with st.spinner("Generating follow-up questions..."):
                follow_up_questions = generate_follow_up_questions(st.session_state.state)
            
            # Add AI message to conversation history
            st.session_state.conversation_history.append({
                "role": "assistant",
                "content": follow_up_questions
            })
            
            # Store initial AI questions in additional_context
            st.session_state.state["additional_context"].append(f"Initial AI Question(s):\n{follow_up_questions}")
            
            # Set waiting for human input
            st.session_state.waiting_for_human_input = True
            
            # Force a rerun to update the UI
            st.rerun()
    
    # Display conversation history
    if st.session_state.conversation_history:
        display_conversation(st.session_state.conversation_history)
        
    # Display debug information about the current state
    if st.session_state.state:
        with st.expander("Debug: Current State"):
            # Create a deep copy of the state to display
            display_state = {}
            for key, value in st.session_state.state.items():
                # Handle special cases for display
                if isinstance(value, list) and key == "human_feedback":
                    display_state[key] = [str(item) for item in value]
                else:
                    display_state[key] = value
            
            st.json(display_state)
    
    # Handle human feedback
    if st.session_state.waiting_for_human_input and not st.session_state.conversation_ended:
        # Use a fixed key for the text area
        if "human_feedback_input" not in st.session_state:
            st.session_state.human_feedback_input = ""
            
        human_feedback = st.text_area("Your response (type /end to finish):", 
                                    value=st.session_state.human_feedback_input, 
                                    height=100, 
                                    key="human_feedback_area")
        submit_feedback_button = st.button("Submit Response")
        
        if submit_feedback_button and human_feedback:
            # Check if the user wants to end the conversation
            if human_feedback.strip().lower() == "/end":
                st.session_state.conversation_ended = True
                
                # Add user message to conversation history
                st.session_state.conversation_history.append({
                    "role": "user",
                    "content": "/end - Conversation ended"
                })
                
                # Generate knowledge base questions
                with st.spinner("Generating knowledge base questions..."):
                    kb_questions = generate_knowledge_base_questions(st.session_state.state)
                
                # Add AI message to conversation history
                st.session_state.conversation_history.append({
                    "role": "assistant",
                    "content": f"Based on our conversation, here are targeted knowledge base search questions:\n\n{kb_questions}"
                })
                
                # Add final KB questions to additional_context
                st.session_state.state["additional_context"].append(
                    f"Final Knowledge Base Questions:\n{kb_questions}"
                )
                
                st.session_state.kb_questions_generated = True
                
                # Clear the input after processing
                st.session_state.human_feedback_input = ""
                st.rerun()
            else:
                # Add user message to conversation history
                st.session_state.conversation_history.append({
                    "role": "user",
                    "content": human_feedback
                })
                
                # Update state with human feedback
                st.session_state.state["human_feedback"].append(human_feedback)
                
                # Generate follow-up questions
                with st.spinner("Generating follow-up questions..."):
                    follow_up_questions = generate_follow_up_questions(st.session_state.state)
                
                # Add AI message to conversation history
                st.session_state.conversation_history.append({
                    "role": "assistant",
                    "content": follow_up_questions
                })
                
                # Store the conversation turn in additional_context
                conversation_turn = f"AI Question(s):\n{follow_up_questions}\n\nHuman Answer:\n{human_feedback}"
                st.session_state.state["additional_context"].append(conversation_turn)
                
                # Clear the input after processing
                st.session_state.human_feedback_input = ""
                st.rerun()
    
    # Show a reset button
    if st.session_state.state:
        if st.button("Start New Conversation"):
            # Reset all session state
            st.session_state.conversation_history = []
            st.session_state.state = None
            st.session_state.waiting_for_human_input = False
            st.session_state.conversation_ended = False
            st.session_state.kb_questions_generated = False
            st.rerun()

if __name__ == "__main__":
    main()
