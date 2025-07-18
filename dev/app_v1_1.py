import streamlit as st
import os
import sys
import warnings
import logging
from typing import Dict, List, Any, Optional

# Add parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project components
from hitl_graph import hitl_app, create_hitl_graph
from src.state_v1_1 import InitState
from src.configuration_v1_1 import get_config_instance
from src.utils_v1_1 import clear_cuda_memory
from langchain_community.embeddings import HuggingFaceEmbeddings

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.getLogger("streamlit").setLevel(logging.ERROR)

# Configure Streamlit page
st.set_page_config(
    page_title="Human-in-the-Loop RAG Assistant",
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

# Function to get embedding model
def get_embedding_model(model_name):
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'}
    )

# Function to display the conversation history
def display_conversation(history):
    st.markdown('<div class="conversation-container">', unsafe_allow_html=True)
    
    for i, entry in enumerate(history):
        if entry["role"] == "user":
            st.markdown(f'<div class="user-message"><b>You:</b> {entry["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="ai-message"><b>AI:</b> {entry["content"]}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Function to display workflow progress
def display_workflow_progress(current_step):
    # This function is now a no-op as requested by the user
    pass

# Function to display the graph visualization using the project's existing function
def display_graph_visualization():
    # Create a new graph instance for visualization
    graph = create_hitl_graph()
    
    try:
        # Generate a PNG image using the draw_mermaid_png method
        png_data = graph.get_graph().draw_mermaid_png()
        
        # Display the image
        st.image(png_data, use_container_width=True, caption="LangGraph Workflow")
    except Exception as e:
        st.error(f"Error generating graph visualization: {str(e)}")
        
        # Fall back to the text representation if image fails
        try:
            mermaid_representation = graph.get_graph().draw_mermaid()
            st.markdown("""
            ### Mermaid Text Representation (Fallback)
            The image generation failed, displaying text representation instead.
            """)
            st.markdown(mermaid_representation, unsafe_allow_html=True)
        except Exception as inner_e:
            st.error(f"Error generating mermaid text: {str(inner_e)}")

# Main function
def main():
    st.title("Human-in-the-Loop RAG Assistant")
    st.markdown("""
    This app demonstrates a human-in-the-loop approach to RAG (Retrieval-Augmented Generation).
    The AI will ask clarifying questions based on your initial query and learn from your responses.
    
    To end the conversation at any time, type `/end` or click the "End Conversation" button.
    """)
    
    # Create tabs for different sections
    tab1, tab2 = st.tabs(["Chat", "Workflow Visualization"])
    
    # Tab 1: Chat Interface
    with tab1:
        # Model selection
        st.subheader("Model Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            report_llm = st.selectbox(
                "Report LLM",
                options=["deepseek-r1:latest", "deepseek-r1:7b", "llama3:8b", "llama3:70b", "mistral:7b", "mixtral:8x7b"],
                index=0
            )
        
        with col2:
            summarization_llm = st.selectbox(
                "Summarization LLM",
                options=["deepseek-r1:latest", "deepseek-r1:7b", "llama3:8b", "llama3:70b", "mistral:7b", "mixtral:8x7b"],
                index=0
            )
        
        model_options = {
            "report_llm": report_llm,
            "summarization_llm": summarization_llm
        }
        
        # User input for initial query
        st.subheader("Start Conversation")
        user_query = st.text_area("Enter your initial query:", height=100)
        
        # Initialize session state variables for managing the HITL workflow
        if "conversation_history" not in st.session_state:
            st.session_state.conversation_history = []
            
        if "workflow_state" not in st.session_state:
            st.session_state.workflow_state = None
            
        if "current_step" not in st.session_state:
            st.session_state.current_step = "START"
            
        if "waiting_for_input" not in st.session_state:
            st.session_state.waiting_for_input = False
            
        if "is_first_run" not in st.session_state:
            st.session_state.is_first_run = True
            
        if "thread_config" not in st.session_state:
            st.session_state.thread_config = {"configurable": {"report_llm": report_llm}}
            
        if "thread_id" not in st.session_state:
            # Generate a unique thread ID for this conversation
            import uuid
            st.session_state.thread_id = str(uuid.uuid4())
            
        # Reset button to clear conversation
        if st.button("Reset Conversation"):
            st.session_state.conversation_history = []
            st.session_state.workflow_state = None
            st.session_state.current_step = "START"
            st.session_state.waiting_for_input = False
            st.session_state.is_first_run = True
            
        # Start button to begin the conversation
        start_button = st.button("Start Conversation", disabled=st.session_state.waiting_for_input)
        
        # Display conversation history if it exists
        if st.session_state.conversation_history:
            st.subheader("Conversation")
            display_conversation(st.session_state.conversation_history)
            
            # Display current state if available
            if st.session_state.workflow_state and st.session_state.workflow_state != "":
                with st.expander("Current LangGraph State", expanded=False):
                    # Function to display state in a more readable format
                    def display_state_value(value):
                        if isinstance(value, list):
                            return "[" + ", ".join([str(item) for item in value]) + "]"
                        return str(value)
                    
                    # Get the current state excluding special keys
                    state_dict = {}
                    for key, value in st.session_state.workflow_state.items():
                        if not key.startswith("__") and key != "final_answer":
                            state_dict[key] = display_state_value(value)
                    
                    # Display the state
                    st.write("### Current State Values")
                    for key, value in state_dict.items():
                        st.write(f"**{key}**: {value}")
                    
                    # Display current position in workflow
                    if "__current_node__" in st.session_state.workflow_state:
                        st.write(f"**Current Node**: {st.session_state.workflow_state['__current_node__']}")
                    
                    # Display if we're in an interrupt
                    if "__interrupt__" in st.session_state.workflow_state:
                        st.write("**Status**: Waiting for user input (interrupt)")
            
        # Create containers for workflow progress and user input
        progress_container = st.container()
        user_input_container = st.container()
        
        # Start the HITL workflow when the button is clicked
        if start_button and user_query and st.session_state.is_first_run:
            # Clear existing memory
            clear_cuda_memory()
            
            # Update thread config with selected models
            st.session_state.thread_config = {"configurable": {"report_llm": report_llm}}
            
            # Initialize the state
            initial_state = InitState(
                user_query=user_query,
                current_position=0,
                detected_language="",
                additional_context=[],
                human_feedback=[],
                report_llm=model_options["report_llm"],
                summarization_llm=model_options["summarization_llm"]
            )
            
            # Add user query to conversation history
            st.session_state.conversation_history.append({"role": "user", "content": user_query})
            
            # Start the workflow and get the first result
            try:
                # Use thread_id to maintain state across invocations
                result = hitl_app.invoke(
                    input=initial_state, 
                    config={"thread_id": st.session_state.thread_id, "configurable": {"report_llm": report_llm}}
                )
                
                # Store the workflow state
                st.session_state.workflow_state = result
                
                # Mark that we've started the workflow
                st.session_state.is_first_run = False
                
                # Update current step if available
                if "__current_node__" in result:
                    st.session_state.current_step = result["__current_node__"]
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                
        # Handle ongoing HITL workflow
        if not st.session_state.is_first_run and st.session_state.workflow_state:
            # Display workflow progress
            with progress_container:
                st.subheader("Workflow Progress")
                # Workflow progress display removed as requested
            
            # Check for interrupts
            if "__interrupt__" in st.session_state.workflow_state:
                # Extract interrupt data - handle different interrupt structures
                try:
                    # Get the interrupt object
                    interrupt_obj = st.session_state.workflow_state["__interrupt__"]
                    
                    # Try to extract the AI message in various ways
                    ai_message = None
                    
                    # Handle list of Interrupt objects
                    if isinstance(interrupt_obj, list) and len(interrupt_obj) > 0:
                        interrupt_item = interrupt_obj[0]
                        
                        # Check if it has a value attribute that's a dict with ai_message
                        if hasattr(interrupt_item, "value") and isinstance(interrupt_item.value, dict) and "ai_message" in interrupt_item.value:
                            ai_message = interrupt_item.value["ai_message"]
                        # Try other methods if the above didn't work
                        elif hasattr(interrupt_item, "ai_message"):
                            ai_message = interrupt_item.ai_message
                    
                    # Method 1: Direct attribute access
                    elif hasattr(interrupt_obj, "ai_message"):
                        ai_message = interrupt_obj.ai_message
                    
                    # Method 2: Through value attribute
                    elif hasattr(interrupt_obj, "value"):
                        value = interrupt_obj.value
                        if hasattr(value, "ai_message"):
                            ai_message = value.ai_message
                        elif isinstance(value, dict) and "ai_message" in value:
                            ai_message = value["ai_message"]
                    
                    # Method 3: As a dictionary
                    elif isinstance(interrupt_obj, dict):
                        if "ai_message" in interrupt_obj:
                            ai_message = interrupt_obj["ai_message"]
                        elif "value" in interrupt_obj and isinstance(interrupt_obj["value"], dict) and "ai_message" in interrupt_obj["value"]:
                            ai_message = interrupt_obj["value"]["ai_message"]
                    
                    # Fallback if we couldn't extract the message
                    if ai_message is None:
                        ai_message = "Please provide your response or type '/end' to finish."
                        st.warning(f"Could not extract AI message from interrupt: {str(interrupt_obj)}")
                
                except Exception as e:
                    # Ultimate fallback - create generic message
                    st.warning(f"Could not extract AI message from interrupt: {str(e)}")
                    ai_message = "Please provide your response or type '/end' to finish."
                
                # Check if we're waiting for user input
                if not st.session_state.waiting_for_input:
                    # Add AI questions to conversation history
                    st.session_state.conversation_history.append({"role": "assistant", "content": ai_message})
                    
                    # Mark that we're waiting for user input
                    st.session_state.waiting_for_input = True
                    st.session_state.current_step = "human_feedback"
                    
                # Display user input form
                with user_input_container:
                    st.markdown('<div class="user-input-container">', unsafe_allow_html=True)
                    
                    # Display AI questions prominently
                    st.markdown('<div class="ai-question-box"><strong>AI Questions:</strong></div>', unsafe_allow_html=True)
                    
                    # Check if we have structured questions in the state
                    structured_questions = None
                    if "__interrupt__" in st.session_state.workflow_state:
                        interrupt_data = st.session_state.workflow_state["__interrupt__"]
                        
                        # Try to extract structured questions
                        if isinstance(interrupt_data, list) and len(interrupt_data) > 0:
                            interrupt_obj = interrupt_data[0]
                            # Method 1: As an object with value attribute
                            if hasattr(interrupt_obj, "value") and isinstance(interrupt_obj.value, dict):
                                structured_questions = interrupt_obj.value.get("structured_questions")
                            # Method 2: As a direct attribute
                            elif hasattr(interrupt_obj, "structured_questions"):
                                structured_questions = interrupt_obj.structured_questions
                        # Method 3: As a dictionary
                        elif isinstance(interrupt_data, dict):
                            structured_questions = interrupt_data.get("structured_questions")
                        # Method 4: Direct access to interrupt object
                        elif hasattr(interrupt_data, "value") and isinstance(interrupt_data.value, dict):
                            structured_questions = interrupt_data.value.get("structured_questions")
                        
                        # Debug output
                        print(f"Extracted structured questions: {structured_questions}")
                        print(f"Interrupt data type: {type(interrupt_data)}")
                        if isinstance(interrupt_data, list) and len(interrupt_data) > 0:
                            print(f"First interrupt object type: {type(interrupt_data[0])}")
                            if hasattr(interrupt_data[0], "value"):
                                print(f"Value type: {type(interrupt_data[0].value)}")
                                print(f"Value contents: {interrupt_data[0].value}")
                        elif isinstance(interrupt_data, dict):
                            print(f"Interrupt dict keys: {interrupt_data.keys()}")
                            if "value" in interrupt_data and isinstance(interrupt_data["value"], dict):
                                print(f"Value dict keys: {interrupt_data['value'].keys()}")
                    
                    # Display questions in a structured format if available
                    if structured_questions and isinstance(structured_questions, list):
                        for i, question in enumerate(structured_questions):
                            st.markdown(f'<div class="ai-question-item">Q{i+1}: {question}</div>', unsafe_allow_html=True)
                    else:
                        # Fallback to displaying the raw message
                        st.markdown(f'<div class="ai-question-raw">{ai_message}</div>', unsafe_allow_html=True)
                    
                    user_feedback = st.text_area(
                        "Your response:",
                        key=f"user_feedback_{len(st.session_state.conversation_history)}",
                        height=100
                    )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        submit_button = st.button(
                            "Submit Response", 
                            key=f"submit_{len(st.session_state.conversation_history)}"
                        )
                    
                    with col2:
                        end_button = st.button(
                            "End Conversation (/end)", 
                            key=f"end_{len(st.session_state.conversation_history)}"
                        )
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                # Handle user input submission
                if submit_button or end_button:
                    # Process the user's response
                    user_response = user_feedback
                    if end_button:
                        user_response = "/end"
                    
                    # Add user response to conversation history
                    st.session_state.conversation_history.append({"role": "user", "content": user_response})
                    
                    # Continue workflow with user input
                    try:
                        # Use invoke with input, from_state and resume parameters
                        result = hitl_app.invoke(
                            input=user_response,
                            from_state=st.session_state.workflow_state,
                            config={"thread_id": st.session_state.thread_id, "configurable": {"report_llm": report_llm}}
                        )
                        
                        # Store new workflow state
                        st.session_state.workflow_state = result
                        
                        # Update current step if available
                        if "__current_node__" in result:
                            st.session_state.current_step = result["__current_node__"]
                            
                        # Reset waiting flag
                        st.session_state.waiting_for_input = False
                        
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
            
            # Check for final answer
            elif "final_answer" in st.session_state.workflow_state:
                final_answer = st.session_state.workflow_state["final_answer"]
                
                # Add final answer to conversation if it's not already there
                if all(entry["content"] != final_answer for entry in st.session_state.conversation_history if entry["role"] == "assistant"):
                    st.session_state.conversation_history.append({"role": "assistant", "content": final_answer})
                    
                # Update workflow progress
                st.session_state.current_step = "END"
                
                # Display success message
                st.success("Conversation completed successfully!")
                
    # Tab 2: Workflow Visualization
    with tab2:
        st.subheader("LangGraph Workflow")
        
        # Display the workflow diagram using the existing function
        display_graph_visualization()
        
        # Display model information
        config = get_config_instance()
        embedding_model = config.embedding_model
        
        st.markdown(f"""
        ### Model Information
        
        - **Embedding Model:** {embedding_model}
        - **Report LLM:** {model_options["report_llm"]}
        - **Summarization LLM:** {model_options["summarization_llm"]}
        """)
        
        # Explanation of the workflow
        st.markdown("""
        ### Workflow Explanation
        
        1. **Start**: The workflow begins when you submit your initial query.
        2. **Display Embedding Model**: The system shows which embedding model is being used.
        3. **Detect Language**: The system detects the language of your query.
        4. **Generate AI Questions**: The AI generates clarifying questions based on your query.
        5. **Human Feedback**: You provide responses to the AI's questions.
        6. **Summarize Feedback**: After you end the conversation, the AI summarizes all the information gathered.
        7. **Final Response**: The AI generates a comprehensive response based on your initial query and all the additional context.
        8. **End**: The workflow completes, and the final answer is displayed.
        """)

if __name__ == "__main__":
    main()
