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
from src.rag_helpers_v1_1 import get_all_available_models, get_license_content

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
    st.markdown('</div>', unsafe_allow_html=True)

def detect_language(state: InitState):
    """
    Detect language of user query.
    
    Args:
        state (InitState): The current state.
    
    Returns:
        state (InitState): The updated state with the detected language.
    """
    query = state["user_query"]
    model = state.get("report_llm", "deepseek-r1:latest")
    # Format the system prompt for language detection
    system_prompt = """# ROLE
You are an expert language detection specialist.

# GOAL
Identify the primary language of the provided text with high accuracy.

# CONTEXT
- You will receive a user query or text snippet
- The text may contain technical terms, proper nouns, or mixed content
- Focus on the grammatical structure and core vocabulary to determine the language

# OUTPUT FORMAT
Respond with ONLY the language name in English.
Examples of valid responses: "English", "German", "Spanish", "French", "Chinese", "Japanese", "Arabic"

# CONSTRAINTS
- Provide only the language name, no additional text
- Be precise and confident in your detection
- If uncertain between similar languages, choose the most likely one"""
    
    # Format the human prompt
    human_prompt = f"""TEXT TO ANALYZE:
{query}

LANGUAGE:"""
    
    # Using local model with Ollama
    result = invoke_ollama(
        model=model,
        system_prompt=system_prompt,
        user_prompt=human_prompt,
    )
    
    # Parse the result to get just the language name
    parsed_result = parse_output(result)
    detected_language = parsed_result["response"].strip()
    
    # Log the detected language
    print(f"Detected language: {detected_language}")
    
    return {"detected_language": detected_language, "current_position": "detect_language"}

def analyse_user_feedback(state: InitState):
    """
    Analyze user feedback.
    
    Args:
        state (InitState): The current state.
    
    Returns:
        str: Analyzed user feedback.
    """
    query = state["user_query"]
    detected_language = state["detected_language"]
    human_feedback = state.get("human_feedback", "")
    additional_context = state.get("additional_context", [])
    
    # Use the configured LLM model
    model_to_use = state.get("report_llm", "deepseek-r1:latest")
    
    # Format system prompt for analysis
    system_prompt = f"""# ROLE
You are an expert domain analysis specialist for human-in-the-loop conversations.

# GOAL
Analyze the human's feedback to identify the specific technical or specialized domain context.

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
    additional_context += "\n\n" + f"Human Feedback: {human_feedback}\nAI Analysis: {parsed_result['response']}"
    return {"analysis": parsed_result["response"], "additional_context": additional_context, "current_position": "analyse_user_feedback"}

def generate_follow_up_questions(state: InitState):
    """
    Generate follow-up questions based on the current state.
    
    Args:
        state (InitState): The current state.
    
    Returns:
        state (InitState): The updated state with generated follow-up questions.
    """
    query = state["user_query"]
    detected_language = state["detected_language"]
    human_feedback = state.get("human_feedback", "")
    analysis = state.get("analysis", "")
    additional_context = state.get("additional_context", "")
    
    # Use the configured LLM model
    model_to_use = state.get("report_llm", "deepseek-r1:latest")
    
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
- Human feedback received: Available in user prompt

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
    additional_context += "\n" + f"AI Follow-up Questions: {parsed_result['response']}"
    return {"follow_up_questions": parsed_result["response"], "additional_context": additional_context, "current_position": "generate_follow_up_questions"}

def generate_knowledge_base_questions(state: InitState):
    """
    Generate knowledge base questions based on the completed interaction.
    Uses two separate LLM invocations:
    1. First LLM call: Deep analysis of user query and HITL feedback
    2. Second LLM call: Generate knowledge base questions using initial query + additional_context
    
    Args:
        state (InitState): The current state with all human feedback.
    
    Returns:
        state (InitState): The updated state with generated knowledge base questions
        and deep analysis as additional_context.
    """
    query = state["user_query"]
    detected_language = state["detected_language"]
    human_feedback = state.get("human_feedback", "")
    additional_context = state.get("additional_context", "")
    
    # Use the configured LLM model
    model_to_use = state.get("report_llm", "deepseek-r1:latest")
    
    print(f"[DEBUG] Starting generate_knowledge_base_questions with two LLM calls")
    print(f"[DEBUG] Model to use: {model_to_use}")
    print(f"[DEBUG] Detected language: {detected_language}")
    
    # ========================================
    # FIRST LLM CALL: Deep Analysis
    # ========================================
    print(f"[DEBUG] Step 1/2: Performing deep analysis of user query and HITL feedback")
    
    analysis_system_prompt = f"""# ROLE
You are an expert information synthesis and analysis specialist with deep subject matter expertise.

# GOAL
Create a profound and insightful representation of the user's information needs based on the initial query and human-in-the-loop conversation.

# AVAILABLE INFORMATION
- Initial user query: The user's original question or request
- Complete conversation history: All interactions between user and AI
- Human feedback exchanges: Clarifications and additional context from the user
- Detected language: {detected_language}

# ANALYSIS TASK
1. Deeply analyze the original user query to identify the core information need
2. Examine the HITL feedback exchanges to identify clarifications and refinements
3. Synthesize these insights into a clear, profound representation of what the user truly needs
4. Identify underlying assumptions, constraints, and priorities revealed in the conversation
5. Recognize technical terminology and domain-specific concepts that indicate expertise level

# OUTPUT FORMAT
Provide 3-4 clear, profound, and query-oriented summaries that represent the essence of the information need.
Each summary should be 1-2 sentences and capture a different aspect of the user's requirements.

Format as:
1. [First insight about the user's core information need]
2. [Second insight focusing on specific technical requirements]
3. [Third insight capturing context, constraints or priorities]
4. [Optional fourth insight if needed for completeness]

# CRITICAL CONSTRAINTS
- Write EXCLUSIVELY in {detected_language} language
- Focus on deep understanding rather than surface-level query reformulation
- Capture nuance, technical specificity, and context from the entire conversation
- Each insight must be standalone and valuable for guiding information retrieval
- Prioritize clarity and precision over length
"""
    
    analysis_human_prompt = f"""# ORIGINAL QUERY
{query}

# COMPLETE CONVERSATION HISTORY
{additional_context if additional_context else "No detailed conversation history available."}

# HUMAN FEEDBACK EXCHANGES
"""
    
    if human_feedback:
        for i, feedback in enumerate(human_feedback):
            analysis_human_prompt += f"Exchange {i+1}: {feedback}\n"
    else:
        analysis_human_prompt += "No additional human feedback exchanges.\n"
    
    analysis_human_prompt += f"\n# TASK\nBased on the complete conversation above, provide 3-4 profound insights that capture the essence of the user's information needs in {detected_language}:"
    
    # First LLM invocation: Generate deep analysis
    print(f"[DEBUG] Invoking first LLM call for deep analysis...")
    analysis_result = invoke_ollama(
        model=model_to_use,
        system_prompt=analysis_system_prompt,
        user_prompt=analysis_human_prompt,
    )
    
    # Parse the result to get the deep analysis
    analysis_parsed_result = parse_output(analysis_result)
    deep_analysis = analysis_parsed_result["response"]
    print(f"[DEBUG] First LLM call completed. Deep analysis generated.")
    
    # Update additional_context with the deep analysis
    updated_additional_context = additional_context
    if updated_additional_context:
        updated_additional_context += "\n\n"
    updated_additional_context += f"## Deep Analysis of Information Needs\n{deep_analysis}"
    
    print(f"[DEBUG] Additional context updated with deep analysis")
    
    # ========================================
    # SECOND LLM CALL: Knowledge Base Questions
    # ========================================
    print(f"[DEBUG] Step 2/2: Generating knowledge base questions using initial query + additional_context")
    
    kb_system_prompt = f"""# ROLE
You are an expert knowledge base search query specialist.

# GOAL
Generate 5 highly targeted, searchable questions optimized for knowledge base retrieval based on the initial user query and the deep analysis of their information needs.

# AVAILABLE INFORMATION
- Initial user query: The user's original question
- Deep analysis of information needs: Comprehensive analysis from previous step
- Detected language: {detected_language}

# SEARCH QUERY OPTIMIZATION STRATEGY
1. Use specific technical terminology likely to match knowledge base content
2. Focus on different aspects of the user's information need identified in the analysis
3. Frame as search queries, not conversational questions
4. Cover both broad concepts and specific implementation details
5. Avoid redundancy between questions
6. Include relevant keywords and domain-specific terms
7. Consider different search angles (what, how, why, when, where)
8. Leverage the deep analysis insights to create more targeted queries

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
    
    kb_human_prompt = f"""# INITIAL USER QUERY
{query}

# DEEP ANALYSIS OF INFORMATION NEEDS
{deep_analysis}

# TASK
Based on the initial query and the deep analysis above, generate 5 targeted knowledge base search questions in {detected_language} that will help retrieve the most relevant information:"""
    
    # Second LLM invocation: Generate knowledge base questions
    print(f"[DEBUG] Invoking second LLM call for knowledge base questions...")
    kb_result = invoke_ollama(
        model=model_to_use,
        system_prompt=kb_system_prompt,
        user_prompt=kb_human_prompt,
    )
    
    # Parse the result
    kb_parsed_result = parse_output(kb_result)
    knowledge_base_questions = kb_parsed_result["response"]
    print(f"[DEBUG] Second LLM call completed. Knowledge base questions generated.")
    
    print(f"[DEBUG] generate_knowledge_base_questions completed successfully with two separate LLM calls")
    
    return {
        "knowledge_base_questions": knowledge_base_questions, 
        #"additional_context": updated_additional_context,
        "additional_context": deep_analysis,
        "current_position": "generate_knowledge_base_questions"
    }

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
    workflow.add_node("analyse_user_feedback", analyse_user_feedback)
    workflow.add_node("generate_follow_up_questions", generate_follow_up_questions)
    workflow.add_node("generate_knowledge_base_questions", generate_knowledge_base_questions)
    
    # Add edges
    workflow.add_edge(START, "detect_language")
    workflow.add_edge("detect_language", "analyse_user_feedback")
    workflow.add_edge("analyse_user_feedback", "generate_follow_up_questions")
    workflow.add_edge("generate_follow_up_questions", END)
    
    # Compile the graph
    return workflow.compile()

def main():
    st.title("Human-in-the-Loop AI Assistant")
    st.markdown('<p style="font-size:12px; font-weight:bold; color:darkorange; margin-top:-10px;">LICENCE</p>', 
               unsafe_allow_html=True, help=get_license_content())
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
    
    # Model selection - load from global configuration
    available_models = get_all_available_models()
    # Prefer qwen3:1.7b as default, fallback to qwen3:30b-a3b, then index 0
    default_index = 0
    if "qwen3:1.7b" in available_models:
        default_index = available_models.index("qwen3:1.7b")
    elif "qwen3:30b-a3b" in available_models:
        default_index = available_models.index("qwen3:30b-a3b")
    
    selected_model = st.selectbox(
        "Select LLM Model", 
        available_models, 
        index=default_index,
        help="Select LLM model; loaded from global report_llms.md and summarization_llms.md configuration"
    )
    
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
                "additional_context": "",  # Will store annotated conversation history
                "human_feedback": "",
                "analysis": "",
                "follow_up_questions": "",
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
                detected_language = detect_language(st.session_state.state)
                st.session_state.state["detected_language"] = detected_language
            
            # Generate initial follow-up questions
            with st.spinner("Generating follow-up questions..."):
                follow_up_result = generate_follow_up_questions(st.session_state.state)
                st.session_state.state["follow_up_questions"] = follow_up_result["follow_up_questions"]
            
            # For initial query, we don't have analysis yet, so use empty string
            st.session_state.state["analysis"] = ""
            
            # Format the combined response for initial query
            combined_response = f"FOLLOW-UP: {st.session_state.state['follow_up_questions']}"
            
            # Add AI message to conversation history with formatted content
            st.session_state.conversation_history.append({
                "role": "assistant",
                "content": combined_response
            })
            
            # Store initial AI questions in additional_context
            st.session_state.state["additional_context"] += f"Initial AI Question(s):\n{st.session_state.state['follow_up_questions']}"
            
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
        # Initialize input counter for dynamic keys
        if "input_counter" not in st.session_state:
            st.session_state.input_counter = 0
            
        # Use a dynamic key that changes after each submission to force widget reset
        human_feedback = st.text_area("Your response (type /end to finish):", 
                                    value="", 
                                    height=100, 
                                    key=f"human_feedback_area_{st.session_state.input_counter}")
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
                    kb_questions_result = generate_knowledge_base_questions(st.session_state.state)
                    kb_questions_content = kb_questions_result["knowledge_base_questions"]
                    deep_analysis_content = kb_questions_result["additional_context"]
                
                # Create formatted response with both deep analysis and questions
                formatted_response = f"""## Deep Analysis of Your Information Needs

{deep_analysis_content}

## Targeted Knowledge Base Search Questions

Based on our conversation and the analysis above, here are targeted knowledge base search questions:

{kb_questions_content}"""
                
                # Add AI message to conversation history
                st.session_state.conversation_history.append({
                    "role": "assistant",
                    "content": formatted_response
                })
                
                # Add final KB questions to additional_context
                st.session_state.state["additional_context"] += (
                    f"Final Knowledge Base Questions:\n{kb_questions_content}"
                )
                
                st.session_state.kb_questions_generated = True
                
                # Increment counter to force new text area widget and clear input
                st.session_state.input_counter += 1
                st.rerun()
            else:
                # Add user message to conversation history
                st.session_state.conversation_history.append({
                    "role": "user",
                    "content": human_feedback
                })
                
                # Update state with human feedback
                st.session_state.state["human_feedback"] += human_feedback
                
                # Analyze user feedback first
                with st.spinner("Analyzing feedback..."):
                    analysis_result = analyse_user_feedback(st.session_state.state)
                    st.session_state.state["analysis"] = analysis_result["analysis"]
                
                # Generate follow-up questions
                with st.spinner("Generating follow-up questions..."):
                    follow_up_result = generate_follow_up_questions(st.session_state.state)
                    st.session_state.state["follow_up_questions"] = follow_up_result["follow_up_questions"]
                
                # Format the combined response using state values as requested
                combined_response = f"ANALYSIS: {st.session_state.state['analysis']}\n\nFOLLOW-UP: {st.session_state.state['follow_up_questions']}"
                
                # Add AI message to conversation history
                st.session_state.conversation_history.append({
                    "role": "assistant",
                    "content": combined_response
                })
                
                # Store the conversation turn in additional_context
                conversation_turn = f"AI Question(s):\n{st.session_state.state['follow_up_questions']}\n\nHuman Answer:\n{human_feedback}"
                st.session_state.state["additional_context"] += conversation_turn
                
                # Increment counter to force new text area widget and clear input
                st.session_state.input_counter += 1
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
