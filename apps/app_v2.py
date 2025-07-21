import streamlit as st
import streamlit_nested_layout
import warnings
import logging
import os
import re
import sys
import time
import uuid
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from IPython.display import Image, display

# Add a workaround for the Streamlit/torch module path extraction issue
# This needs to be done before importing torch
class PathHack:
    def __init__(self, path):
        self.path = path
    def _path(self):
        return self.path
    def __getattr__(self, name):
        return getattr(self.path, name)

sys.modules['torch._classes.__path__'] = PathHack(os.path.dirname(os.path.abspath(__file__)))

# Now import torch after the workaround
import torch
import numpy as np
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.schema.document import Document
from langchain.schema.messages import HumanMessage, AIMessage
from langchain.schema.runnable import RunnableConfig
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.base import BaseCallbackHandler

# Try to import pyperclip for clipboard functionality
try:
    import pyperclip
    PYPERCLIP_AVAILABLE = True
except ImportError:
    PYPERCLIP_AVAILABLE = False

# Custom logging handler for Streamlit
class StreamlitLogHandler(logging.Handler):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback
        self.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    def emit(self, record):
        log_entry = self.format(record)
        self.callback(log_entry)

# Load environment variables
load_dotenv()

# Define paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
KB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kb")
DB_DIR = os.path.join(KB_DIR, "database")

# Import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.graph_v2 import briefing_app, researcher_app
from src.state_v1_1 import InitState, ResearcherState
from src.configuration_v1_1 import get_config_instance
from src.rag_helpers_v1_1 import get_report_llm_models, get_summarization_llm_models

# Function to clear CUDA memory
def clear_cuda_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        st.toast("CUDA memory cache cleared", icon="🧹")
        print("\nCUDA memory cache cleared\n")

# Function to clean model name for display
def clean_model_name(model_name):
    return model_name.replace(":", "-")

# Function to extract embedding model name from database directory
def extract_embedding_model(db_dir_name):
    """Extract the embedding model name from a database directory name."""
    # Convert from format like 'sentence-transformers--all-mpnet-base-v2--2000--400'
    # or 'Qwen--Qwen3-Embedding-0.6B--3000--600'
    # to 'sentence-transformers/all-mpnet-base-v2' or 'Qwen/Qwen3-Embedding-0.6B'
    
    # Default model if we can't extract
    default_model = "jinaai/jina-embeddings-v2-base-de"
    
    # Try to extract from directory name
    if not db_dir_name or not isinstance(db_dir_name, str):
        return default_model
    
    parts = db_dir_name.split('--')
    if len(parts) >= 2:
        # The first two parts are the embedding model name
        model_name = parts[0].replace('--', '/') + '/' + parts[1]
        return model_name
        
    return default_model

# Function to get embedding model
def get_embedding_model(model_name):
    """Get the embedding model instance based on the model name."""
    # Create and return the embedding model
    return HuggingFaceEmbeddings(model_name=model_name)

def create_workflow_visualization(return_mermaid=False):
    """
    Generate a visualization of the updated workflow with human feedback loop.
    
    Args:
        return_mermaid (bool): If True, return mermaid diagram code
        
    Returns:
        str: Path to visualization image or mermaid diagram code
    """
    # Get the current LLM models in use
    from src.configuration_v1_1 import get_config_instance
    config = get_config_instance()
    embedding_model = config.embedding_model
    summarization_llm = st.session_state.get('summarization_llm', 'llama3.2')
    report_llm = st.session_state.get('report_llm', 'qwq')
    
    # Create the mermaid diagram
    mermaid_code = """```mermaid
graph TD
    classDef start fill:#4CAF50,stroke:#388E3C,color:white
    classDef process fill:#2196F3,stroke:#1976D2,color:white
    classDef decision fill:#FF9800,stroke:#F57C00,color:white
    classDef end fill:#F44336,stroke:#D32F2F,color:white
    classDef human fill:#9C27B0,stroke:#7B1FA2,color:white
    classDef data fill:#607D8B,stroke:#455A64,color:white

    A[Initial Query] --> B[Detect Language]
    B --> C{Human Feedback Loop}
    C -->|AI Questions| D[Human Input]
    D -->|User Feedback| E[Summarize Feedback]
    E --> F[Generate Research Queries]
    F --> G[Web Search]
    G --> H[Retrieve Documents]
    H --> I[Summarize Documents]
    I --> J[Generate Final Report]
    
    A:::start
    B:::process
    C:::decision
    D:::human
    E:::process
    F:::process
    G:::process
    H:::process
    I:::process
    J:::end
```

**Models in use:**

- **Embedding Model:** ${embedding_model}
- **Summarization LLM:** ${summarization_llm}
- **Report LLM:** ${report_llm}
"""
    
    if return_mermaid:
        return mermaid_code
    
    # Save the mermaid diagram to a file
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mermaid_file = os.path.join(project_root, "src", "mermaid_diagram.md")
    
    with open(mermaid_file, "w") as f:
        f.write(mermaid_code)
    
    # Generate the image using mermaid-cli if available
    try:
        # Try to use mmdc command if available
        import subprocess
        output_file = os.path.join(project_root, "src", "mermaid_researcher_graph.png")
        
        # Run mmdc command
        result = subprocess.run(
            ["mmdc", "-i", mermaid_file, "-o", output_file, "-t", "dark"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            return output_file
        else:
            print(f"Error generating mermaid diagram: {result.stderr}")
            return mermaid_code
    except Exception as e:
        print(f"Failed to generate mermaid diagram: {str(e)}")
        return mermaid_code


def generate_response(user_input, enable_web_search, report_structure, max_search_queries, report_llm, summarization_llm, enable_quality_checker, quality_check_loops=1, use_ext_database=False, selected_database=None, k_results=3):
    """
    Generate response using the researcher agent with human feedback loop and stream steps
    """
    # Clear CUDA memory before processing a new query
    clear_cuda_memory()
    
    # Create a placeholder for the workflow visualization
    workflow_vis_placeholder = st.empty()
    
    # Display the workflow visualization
    st.subheader("LangGraph Workflow Visualization")
    
    # Create tabs for different visualizations
    viz_tab1, viz_tab2 = st.tabs(["Mermaid Diagram", "Standard Visualization"])
    
    # Tab 1: Mermaid diagram
    with viz_tab1:
        # Display the mermaid diagram from a static file
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        graph_img_path = os.path.join(project_root, "src", "mermaid_researcher_graph.png")
        
        # Check if the image exists
        if os.path.exists(graph_img_path):
            st.image(graph_img_path, caption="LangGraph Workflow", use_container_width=False)
        else:
            # If image doesn't exist, display a message
            st.warning("Mermaid graph image not found. Please run the workflow once to generate it.")
            
            # Display model information
            # Use the globally imported get_config_instance
            config = get_config_instance()
            embedding_model = config.embedding_model
            summarization_llm = st.session_state.get('summarization_llm', 'llama3.2')
            report_llm = st.session_state.get('report_llm', 'qwq')
            
            st.markdown(f"""**Models in use:**

- **Embedding Model:** {embedding_model}
- **Summarization LLM:** {summarization_llm}
- **Report LLM:** {report_llm}""")

    
    # Tab 2: Standard visualization
    with viz_tab2:
        # Use the briefing graph directly
        try:
            # Try to get a visualization of the graph
            # Different versions of LangGraph have different APIs
            try:
                # Newer LangGraph versions
                graph_viz = briefing_app.get_graph().to_dot()
                st.graphviz_chart(graph_viz)
            except AttributeError:
                # Fallback for older versions
                import networkx as nx
                import matplotlib.pyplot as plt
                from io import BytesIO
                
                # Create a simple NetworkX visualization
                G = nx.DiGraph()
                
                # Add nodes and edges based on our known workflow
                nodes = ["START", "detect_language", "generate_ai_questions", "human_feedback_node", 
                         "summarize_feedback", "prepare_researcher_state", "END"]
                
                for node in nodes:
                    G.add_node(node)
                
                # Add edges representing the workflow
                edges = [
                    ("START", "detect_language"),
                    ("detect_language", "generate_ai_questions"),
                    ("generate_ai_questions", "human_feedback_node"),
                    ("human_feedback_node", "generate_ai_questions"),
                    ("human_feedback_node", "summarize_feedback"),
                    ("summarize_feedback", "prepare_researcher_state"),
                    ("prepare_researcher_state", "END")
                ]
                
                G.add_edges_from(edges)
                
                # Create a plot
                plt.figure(figsize=(10, 6))
                pos = nx.spring_layout(G)
                
                # Color nodes by type
                node_colors = {
                    "START": "lightgreen",
                    "END": "lightcoral",
                    "human_feedback_node": "lightskyblue",
                    "detect_language": "lightgray",
                    "generate_ai_questions": "lightgray",
                    "summarize_feedback": "lightgray",
                    "prepare_researcher_state": "lightgray"
                }
                
                # Draw nodes with colors
                for node, color in node_colors.items():
                    nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color=color)
                
                # Draw edges and labels
                nx.draw_networkx_edges(G, pos)
                nx.draw_networkx_labels(G, pos)
                
                plt.title("Human Feedback Workflow")
                plt.axis("off")
                
                # Convert plot to image
                buf = BytesIO()
                plt.savefig(buf, format="png")
                buf.seek(0)
                plt.close()
                
                # Display the image
                st.image(buf, caption="Workflow Visualization (Fallback)")
        except Exception as e:
            st.error(f"Error generating graph visualization: {str(e)}")
            st.info("Showing text description of workflow instead.")
            
            # Show a text description of the workflow
            st.markdown("""
            **Workflow Steps:**
            1. Detect Language
            2. Generate AI Questions
            3. Human Feedback Loop
            4. Summarize Feedback
            5. Research Query Generation
            6. Web Search & Document Retrieval
            7. Document Summarization
            8. Final Report Generation
            """)

    
    # Create containers for different sections
    status_container = st.container()
    conversation_container = st.container()
    research_container = st.container()
    final_answer_container = st.container()
    
    # Create status indicators
    with status_container:
        st.subheader("Workflow Status")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            human_feedback_status = st.empty()
            human_feedback_status.info("🔄 Human Feedback Loop: Waiting")
        
        with col2:
            research_status = st.empty()
            research_status.info("⏳ Research Phase: Pending")
        
        with col3:
            report_status = st.empty()
            report_status.info("⏳ Report Generation: Pending")
    
    # Initialize workflow state
    if "workflow_state" not in st.session_state:
        st.session_state.workflow_state = None
    
    if "workflow_results" not in st.session_state:
        st.session_state.workflow_results = None
    
    # Get configuration
    config = get_config_instance()
    
    # Set up the database
    if use_ext_database and selected_database:
        # Use the selected external database
        db_path = os.path.join(DB_DIR, selected_database)
        embedding_model_name = extract_embedding_model(selected_database)
    else:
        # Create a temporary database for this session
        temp_dir = tempfile.mkdtemp(prefix="temp_db_")
        db_path = temp_dir
        embedding_model_name = config.embedding_model
    
    # Display the embedding model being used
    print(f"\n=== Using embedding model: {embedding_model_name} ===\n")
    
    # Create the embedding function
    embedding_function = get_embedding_model(embedding_model_name)
    
    # Use the researcher app directly
    # The researcher_app is already compiled and ready to use
    
    # Create the initial state
    initial_state = InitState(
        user_query=user_input,
        additional_context=[],
        language=None
    )
            
    # Display the conversation log
    with conversation_container:
        # First show the user query
        st.write(f"👤 **Initial Query**: {user_input}")
        
        # Initialize progress tracking
        progress_placeholder = st.empty()
        progress_bar = progress_placeholder.progress(0)
        
        # Set up console output capturing
        console_output = st.expander("Console Output", expanded=False)
        console_log = console_output.empty()
        captured_output = []
        
        # Define a callback to capture console output
        def capture_output(text):
            captured_output.append(text)
            console_log.code("\n".join(captured_output), language="bash")
        
        # Set up logging to capture output
        logger = logging.getLogger('langgraph')
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        for handler in logger.handlers[:]: 
            logger.removeHandler(handler)
            
        # Add our custom handler
        stream_handler = StreamlitLogHandler(capture_output)
        logger.addHandler(stream_handler)
        
        # Configure LangGraph thread
        # Convert the Configuration object to a dict to avoid .copy() method requirement
        config_dict = {key: value for key, value in config.items()}
        
        # Generate a unique thread_id for this session
        thread_id = f"thread_{uuid.uuid4().hex[:8]}"
        
        thread_config = {
            "configurable": config_dict,
            "callbacks": [],
            "thread_id": thread_id
        }
        
        # Setup for human-in-the-loop interaction
        interaction_container = st.container()
        feedback_placeholder = st.empty()
        result_placeholder = st.empty()
        
        # Run the combined workflow with interrupt handling
        try:
            # Run workflow until first interrupt
            initial_run = briefing_app.invoke(initial_state, config=thread_config)
            st.session_state.workflow_state = initial_run

            ai_message = None

            if "__interrupt__" in initial_run:
                interrupt_data = initial_run["__interrupt__"]

                # Always extract the first in a list if applicable
                if isinstance(interrupt_data, list):
                    interrupt = interrupt_data[0] if interrupt_data else None
                else:
                    interrupt = interrupt_data

                interrupt_value = None
                # Check for .value attribute (object-style interrupts)
                if hasattr(interrupt, "value"):
                    interrupt_value = interrupt.value
                # Check for 'value' key (dict-style interrupt)
                elif isinstance(interrupt, dict) and "value" in interrupt:
                    interrupt_value = interrupt["value"]
                else:
                    interrupt_value = interrupt  # fallback

                if isinstance(interrupt_value, dict) and "ai_message" in interrupt_value:
                    ai_message = interrupt_value["ai_message"]
                else:
                    # fallback to string or empty
                    ai_message = str(interrupt_value) if interrupt_value is not None else ""

            # Now you can use ai_message
        except Exception as e:
            ai_message = ""
            st.error(f"Interrupt parse error: {e}")

        # Display the AI questions
        if ai_message:
            with interaction_container:
                st.markdown("### 🤖 AI Clarifying Questions")
                
                # Simply display the raw value of ai_message with no additional processing
                st.write(ai_message)
            
            # Create input form for user feedback
            with feedback_placeholder.form(key="feedback_form"):
                user_feedback = st.text_area(
                    "Your response:", 
                    key="user_feedback", 
                    help="Respond to the AI's questions or type 'done' to finish the feedback loop.",
                    height=150
                )
                submit_button = st.form_submit_button("Submit")
                done_button = st.form_submit_button("Done (Complete Feedback)")
                    
                if submit_button and user_feedback:
                    # Resume the workflow with user feedback
                    progress_bar.progress(33)
                    feedback_placeholder.empty()
                    
                    # Display the user's response
                    with interaction_container:
                        st.markdown(f"""👤 **Your Response**:  
{user_feedback}""")
                            
                    # Resume with the user's feedback
                    next_run = briefing_app.invoke(
                        from_state=st.session_state.workflow_state,
                        config=thread_config,
                        resume=user_feedback
                    )
                        
                    st.session_state.workflow_state = next_run
                        
                    # Check if we have another interrupt or if we're done
                    if "__interrupt__" in next_run:
                        interrupt_data = next_run["__interrupt__"]
                        ai_message = interrupt_data.get("ai_message", "")
                        
                        # Display the next AI questions
                        with interaction_container:
                            st.markdown(f"""🤖 **Follow-up Questions**:  
{ai_message}""")
                            
                        # Re-create input form for user feedback
                        with feedback_placeholder.form(key="feedback_form_2"):
                            user_feedback = st.text_area(
                                "Your response:", 
                                key="user_feedback_2", 
                                help="Respond to the AI's follow-up questions or type 'done' to finish the feedback loop.",
                                height=150
                                )
                            submit_button = st.form_submit_button("Submit")
                            done_button = st.form_submit_button("Done (Complete Feedback)")
                                
                            if submit_button and user_feedback:
                                # Resume the workflow with user feedback
                                progress_bar.progress(66)
                                feedback_placeholder.empty()
                                
                                # Display the user's response
                                with interaction_container:
                                    st.markdown(f"""👤 **Your Response**:  
{user_feedback}""")
                                        
                                # Resume with the user's feedback
                                next_run = briefing_app.invoke(
                                    from_state=st.session_state.workflow_state,
                                    config=thread_config,
                                    resume=user_feedback
                                )
                                    
                                st.session_state.workflow_state = next_run
                            elif done_button or user_feedback.lower() == "done":
                                # Force completion of feedback loop
                                progress_bar.progress(66)
                                feedback_placeholder.empty()
                                
                                # Display the user's final response
                                with interaction_container:
                                    st.markdown(f"""👤 **Final Response**:  
{user_feedback if user_feedback else 'done'}""")
                                        
                                # Resume with "done" signal
                                next_run = briefing_app.invoke(
                                    from_state=st.session_state.workflow_state,
                                    config=thread_config,
                                    resume="done"
                                )
                                    
                                st.session_state.workflow_state = next_run
                    else:
                        # We're done with briefing
                        progress_bar.progress(100)
                elif done_button or user_feedback.lower() == "done":
                    # Force completion of feedback loop
                    progress_bar.progress(66)
                    feedback_placeholder.empty()
                    
                    # Display the user's final response
                    with interaction_container:
                        st.markdown(f"""👤 **Final Response**:  
{user_feedback if user_feedback else 'done'}""")
                            
                    # Resume with "done" signal
                    next_run = briefing_app.invoke(
                        from_state=st.session_state.workflow_state,
                        config=thread_config,
                        resume="done"
                    )
                        
                    st.session_state.workflow_state = next_run
            
            # Complete the human feedback phase
            human_feedback_status.success("✅ Human Feedback Loop: Completed")
            
            # Check if we have the final results from briefing
            if "prepare_researcher_state" in st.session_state.workflow_state:
                # Display the summary
                with interaction_container:
                    feedback_summary = st.session_state.workflow_state.get("summarize_feedback", {}).get("feedback_summary", "")
                    if feedback_summary:
                        st.markdown("### Summary of Feedback")
                        st.markdown(feedback_summary)
                
                # Get the prepared state for researcher
                researcher_state = st.session_state.workflow_state["prepare_researcher_state"]
                
                # Update status
                research_status.info("🔄 Research Phase: In progress...")
                
                # Run the researcher workflow
                with research_container:
                    st.subheader("Research Phase")
                    progress_placeholder = st.empty()
                    progress_bar = progress_placeholder.progress(0)
                    
                    # Display research progress
                    research_log = st.empty()
                    research_log.info("Starting research phase...")
                    
                    # Run the researcher workflow
                    researcher_results = researcher_app.invoke(researcher_state, config=thread_config)
                    
                    # Store results
                    st.session_state.workflow_results = researcher_results
                    
                    # Update progress
                    progress_bar.progress(100)
                    research_log.success("Research phase completed!")
                    
                    # Display research queries
                    if "research_queries" in researcher_results:
                        st.markdown("### Research Queries")
                        for i, query in enumerate(researcher_results["research_queries"]):
                            st.markdown(f"{i+1}. {query}")
                    
                    # Display document summaries if available
                    if "search_summaries" in researcher_results and researcher_results["search_summaries"]:
                        st.markdown("### Document Summaries")
                        summaries_expander = st.expander("Click to view document summaries")
                        with summaries_expander:
                            for query, summaries in researcher_results["search_summaries"].items():
                                st.markdown(f"**Query**: {query.split(':', 1)[1] if ':' in query else query}")
                                for i, summary in enumerate(summaries):
                                    st.markdown(f"*Document {i+1}*: {summary.page_content[:200]}...")
                                    st.markdown("---")
                
                # Update status
                research_status.success("✅ Research Phase: Completed")
                report_status.info("🔄 Report Generation: In progress...")
                
                # Final answer generation
                with final_answer_container:
                    st.subheader("Final Report")
                    
                    # Get the final answer
                    final_answer = researcher_results.get("final_answer", "")
                    
                    if final_answer:
                        # Display the final report
                        st.markdown(final_answer)
                        
                        # Add copy button if pyperclip is available
                        if PYPERCLIP_AVAILABLE:
                            if st.button("Copy Report to Clipboard"):
                                pyperclip.copy(final_answer)
                                st.success("Report copied to clipboard!")
                        
                        # Update status
                        report_status.success("✅ Report Generation: Completed")
                    else:
                        report_status.error("❌ Report Generation: Failed")
                        st.error("Failed to generate final report. Please check the console output for errors.")
            else:
                # Something went wrong
                human_feedback_status.warning("⚠️ Human Feedback Loop: Incomplete")
                st.warning("The human feedback loop did not complete successfully. Check the console for errors.")
        
        # Return final results
        if st.session_state.workflow_results and "final_answer" in st.session_state.workflow_results:
            return {
                "final_answer": st.session_state.workflow_results["final_answer"],
                "additional_context": st.session_state.workflow_results.get("additional_context", [])
            }
        else:
            return {
                "final_answer": "Failed to generate a final answer. Please check the console output for errors.",
                "additional_context": []
            }

def main():
    """Main application function"""
    # Remove the default Streamlit menu and footer
    hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """
    st.markdown(hide_st_style, unsafe_allow_html=True)
    
    # Set page config
    st.set_page_config(
        page_title="Local RAG Researcher with Human Feedback",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # App title
    st.title("🔍 Local RAG Researcher with Human Feedback")
    st.markdown("A research assistant that uses local LLMs and RAG with a human-in-the-loop feedback mechanism.")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # LLM Model Selection
        st.subheader("LLM Model Selection")
        
        # Get available models from global configuration
        report_models = get_report_llm_models()
        summarization_models = get_summarization_llm_models()
        
        # Report LLM selection
        report_llm = st.selectbox(
            "Report Generation LLM",
            report_models,
            index=0,
            help="Select the LLM to use for generating the final report; loaded from global report_llms.md configuration"
        )
        
        # Summarization LLM selection
        summarization_llm = st.selectbox(
            "Summarization LLM",
            summarization_models,
            index=0,
            help="Select the LLM to use for summarization tasks; loaded from global summarization_llms.md configuration"
        )
        
        # Store the selected models in session state
        st.session_state.report_llm = report_llm
        st.session_state.summarization_llm = summarization_llm
        
        # Web search toggle
        enable_web_search = st.toggle("Enable Web Search", value=True, help="Enable or disable web search for research")
        
        # Report structure selection
        report_structure = st.selectbox(
            "Report Structure",
            ["Detailed", "Concise", "Academic", "Bullet Points"],
            index=0,
            help="Select the structure for the final report"
        )
        
        # Advanced options
        with st.expander("Advanced Options"):
            # Max search queries
            max_search_queries = st.slider(
                "Max Search Queries",
                min_value=1,
                max_value=10,
                value=3,
                help="Maximum number of search queries to generate"
            )
            
            # Quality checker
            enable_quality_checker = st.toggle("Enable Quality Checker", value=False, help="Enable quality checking for the final report")
            
            # Quality check loops
            quality_check_loops = st.slider(
                "Quality Check Loops",
                min_value=1,
                max_value=5,
                value=1,
                help="Number of quality check loops to perform",
                disabled=not enable_quality_checker
            )
            
            # Number of results to retrieve
            k_results = st.slider(
                "Results per Query",
                min_value=1,
                max_value=10,
                value=3,
                help="Number of results to retrieve per query"
            )
        
        # Database selection
        st.subheader("Vector Database")
        
        # Option to use an existing database
        use_ext_database = st.toggle("Use Existing Database", value=False, help="Use an existing vector database instead of creating a temporary one")
        
        # Database selection
        selected_database = None
        if use_ext_database:
            # Get available databases
            available_databases = []
            if os.path.exists(DB_DIR):
                available_databases = [d for d in os.listdir(DB_DIR) if os.path.isdir(os.path.join(DB_DIR, d))]
            
            if available_databases:
                selected_database = st.selectbox(
                    "Select Database",
                    available_databases,
                    help="Select an existing vector database to use"
                )
                
                # Display the embedding model for the selected database
                if selected_database:
                    embedding_model = extract_embedding_model(selected_database)
                    st.info(f"Embedding Model: {embedding_model}")
            else:
                st.warning("No existing databases found.")
                use_ext_database = False
    
    # Main content
    query = st.text_area("Enter your research query:", height=100)
    
    if st.button("Start Research with Human Feedback"):
        if query:
            with st.spinner("Processing your query..."):
                # Execute the research workflow with human feedback
                result = generate_response(
                    user_input=query,
                    enable_web_search=enable_web_search,
                    report_structure=report_structure,
                    max_search_queries=max_search_queries,
                    report_llm=report_llm,
                    summarization_llm=summarization_llm,
                    enable_quality_checker=enable_quality_checker,
                    quality_check_loops=quality_check_loops,
                    use_ext_database=use_ext_database,
                    selected_database=selected_database,
                    k_results=k_results
                )
        else:
            st.error("Please enter a research query.")

if __name__ == "__main__":
    main()
