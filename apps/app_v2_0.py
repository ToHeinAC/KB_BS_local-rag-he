import streamlit as st
import streamlit_nested_layout
import warnings
import logging
import os
import re
import sys
import time
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
        return [self.path]
    def __getattr__(self, name):
        if name == '_path':
            return self._path
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

sys.modules['torch._classes.__path__'] = PathHack(os.path.dirname(os.path.abspath(__file__)))

# Now import torch after the workaround
import torch

# Import visualization libraries (with fallback if not available)
try:
    import networkx as nx
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

# Add project root to Python path to fix import issues
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Suppress specific PyTorch warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.getLogger("streamlit").setLevel(logging.ERROR)

# Import ResearcherStateV2 and InitState for the enhanced workflow with HITL
from src.state_v2_0 import ResearcherStateV2, InitState
from src.graph_v2_0 import hitl_graph, main_graph, create_hitl_graph, researcher_main
from src.utils_v1_1 import get_report_structures, process_uploaded_files, clear_cuda_memory
from src.rag_helpers_v1_1 import (
    get_report_llm_models, 
    get_summarization_llm_models, 
    get_all_available_models,
    get_license_content
)

# Set page configuration
st.set_page_config(
    page_title="RAG Deep Researcher v2.0 - Human-in-the-Loop (HITL)",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to clean model names for display
def clean_model_name(model_name):
    """Clean model name by removing common prefixes and suffixes for better display"""
    return model_name.replace(":latest", "").replace("_", " ").title()

# Function to extract embedding model name from database directory
def extract_embedding_model(db_dir_name):
    """
    Extract the embedding model name from the database directory name.
    
    This function properly handles various database naming conventions, including:
    - Standard format: "organization/model_name"
    - Directory format with separators: "Qwen/Qwen--Qwen3-Embedding-0.6B--3000--600"
    - Legacy format: "model_name/chunk_size/overlap"
    
    Args:
        db_dir_name (str): The database directory name (e.g., "Qwen/Qwen--Qwen3-Embedding-0.6B--3000--600")
        
    Returns:
        str: The extracted embedding model name (e.g., "Qwen/Qwen3-Embedding-0.6B")
    """
    # Handle the specific case of database names with '--' separators
    # Example: "Qwen--Qwen3-Embedding-0.6B--3000--600" -> "Qwen/Qwen3-Embedding-0.6B"
    if '--' in db_dir_name:
        parts = db_dir_name.split('--')
        
        if len(parts) >= 2:
            # The first part should contain the model organization and name
            first_part = parts[0]  # "Qwen"
            second_part = parts[1]  # "Qwen3-Embedding-0.6B"
            
            if '/' in first_part:
                # Extract organization from first part
                org = first_part.split('/')[0]  # "Qwen"
                result = f"{org}/{second_part}"  # "Qwen/Qwen3-Embedding-0.6B"
                return result
            else:
                # Fallback: use first part as org
                result = f"{first_part}/{second_part}"
                return result
    
    # Fallback to original logic if the new parsing fails
    model_name = db_dir_name.replace("vectordb_", "")
    model_name = model_name.replace("--", "/")

    return model_name

# Function to get embedding model
def get_embedding_model(model_name):
    """Get the embedding model path from the model name"""
    return model_name.replace("/", "_")

def create_mermaid_png_representation(researcher):
    """
    Create a Mermaid PNG diagram representation of the workflow.
    
    Returns:
        PNG bytes of the Mermaid diagram
    """
    return researcher.get_graph().draw_mermaid_png()

def generate_workflow_visualization_legacy(researcher, workflow_type="main", return_mermaid_only=False):
    """
    Generate a visualization of the LangGraph workflow using NetworkX
    
    Args:
        researcher: The researcher graph to visualize
        workflow_type (str): Type of workflow - "hitl" or "main"
        return_mermaid_only (bool): If True, only return the Mermaid representation
        
    Returns:
        str: Path to the visualization image or Mermaid representation
    """
    if return_mermaid_only:
        return create_mermaid_representation(researcher)
    
    if not NETWORKX_AVAILABLE:
        st.warning("NetworkX and matplotlib are not available. Cannot generate workflow visualization.")
        return create_mermaid_representation(researcher)
    
    try:
        # Create a directed graph
        G = nx.DiGraph()
        
        # Define nodes and edges based on workflow type
        if workflow_type == "hitl":
            # HITL workflow nodes
            nodes = [
                ("START", "Start"),
                ("analyse_user_feedback", "Analyze User\nFeedback"),
                ("generate_follow_up_questions", "Generate Follow-up\nQuestions"),
                ("generate_knowledge_base_questions", "Generate KB\nQuestions"),
                ("END", "To Main Workflow")
            ]
            
            # HITL workflow edges
            edges = [
                ("START", "analyse_user_feedback"),
                ("analyse_user_feedback", "generate_follow_up_questions"),
                ("generate_follow_up_questions", "generate_knowledge_base_questions"),
                ("generate_knowledge_base_questions", "END")
            ]
        elif workflow_type == "integrated":
            # Integrated workflow nodes (HITL + Main)
            nodes = [
                ("START", "Start"),
                ("analyse_user_feedback", "Analyze User\nFeedback (HITL)"),
                ("generate_follow_up_questions", "Generate Follow-up\nQuestions (HITL)"),
                ("generate_knowledge_base_questions", "Generate KB\nQuestions (HITL)"),
                ("display_embedding_model_info", "Display Embedding\nModel Info"),
                ("detect_language", "Detect Language"),
                ("generate_research_queries", "Generate Research\nQueries"),
                ("retrieve_rag_documents", "Retrieve RAG\nDocuments"),
                ("summarize_query_research", "Summarize Query\nResearch"),
                ("generate_final_answer", "Generate Final\nAnswer"),
                ("quality_checker", "Quality Checker"),
                ("END", "End")
            ]
            
            # Integrated workflow edges
            edges = [
                ("START", "analyse_user_feedback"),
                ("analyse_user_feedback", "generate_follow_up_questions"),
                ("generate_follow_up_questions", "generate_knowledge_base_questions"),
                ("generate_knowledge_base_questions", "display_embedding_model_info"),
                ("display_embedding_model_info", "detect_language"),
                ("detect_language", "generate_research_queries"),
                ("generate_research_queries", "retrieve_rag_documents"),
                ("retrieve_rag_documents", "summarize_query_research"),
                ("summarize_query_research", "generate_final_answer"),
                ("generate_final_answer", "quality_checker"),
                ("quality_checker", "generate_final_answer"),  # Quality loop
                ("generate_final_answer", "END"),
                ("quality_checker", "END")
            ]
        else:  # main workflow
            # Main workflow nodes
            nodes = [
                ("START", "From HITL"),
                ("display_embedding_model_info", "Display Embedding\nModel Info"),
                ("detect_language", "Detect Language"),
                ("generate_research_queries", "Generate Research\nQueries"),
                ("retrieve_rag_documents", "Retrieve RAG\nDocuments"),
                ("summarize_query_research", "Summarize Query\nResearch"),
                ("generate_final_answer", "Generate Final\nAnswer"),
                ("quality_checker", "Quality Checker"),
                ("END", "End")
            ]
            
            # Main workflow edges
            edges = [
                ("START", "display_embedding_model_info"),
                ("display_embedding_model_info", "detect_language"),
                ("detect_language", "generate_research_queries"),
                ("generate_research_queries", "retrieve_rag_documents"),
                ("retrieve_rag_documents", "summarize_query_research"),
                ("summarize_query_research", "generate_final_answer"),
                ("generate_final_answer", "quality_checker"),
                ("quality_checker", "generate_final_answer"),  # Quality loop
                ("generate_final_answer", "END"),
                ("quality_checker", "END")
            ]
        
        # Add nodes to graph
        for node_id, label in nodes:
            G.add_node(node_id, label=label)
        
        # Add edges to graph
        G.add_edges_from(edges)
        
        # Create layout
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Create figure
        plt.figure(figsize=(16, 12))
        plt.clf()
        
        # Define colors for different node types based on workflow type
        node_colors = []
        for node_id, _ in nodes:
            if node_id in ["START", "END"]:
                node_colors.append('#FF6B6B')  # Red for start/end
            elif workflow_type == "hitl":
                # HITL workflow coloring
                if node_id == "analyse_user_feedback":
                    node_colors.append('#4ECDC4')  # Teal for user feedback analysis
                elif node_id == "generate_follow_up_questions":
                    node_colors.append('#6EC4DB')  # Light blue for follow-up questions
                elif node_id == "generate_knowledge_base_questions":
                    node_colors.append('#56B870')  # Green for KB questions
                else:
                    node_colors.append('#95E1D3')  # Default for HITL nodes
            elif workflow_type == "integrated":
                # Integrated workflow coloring (HITL + Main)
                if node_id in ["analyse_user_feedback", "generate_follow_up_questions", "generate_knowledge_base_questions"]:
                    node_colors.append('#4ECDC4')  # Teal for HITL nodes
                elif node_id == "quality_checker":
                    node_colors.append('#FFE66D')  # Yellow for quality checker
                elif node_id in ["generate_research_queries", "generate_final_answer"]:
                    node_colors.append('#78C3FB')  # Blue for generation nodes
                elif node_id == "retrieve_rag_documents":
                    node_colors.append('#F8B195')  # Orange for retrieval
                elif node_id == "summarize_query_research":
                    node_colors.append('#C06C84')  # Purple for summarization
                else:
                    node_colors.append('#95E1D3')  # Light green for regular nodes
            else:
                # Main workflow coloring
                if node_id == "quality_checker":
                    node_colors.append('#FFE66D')  # Yellow for quality checker
                elif node_id in ["generate_research_queries", "generate_final_answer"]:
                    node_colors.append('#78C3FB')  # Blue for generation nodes
                elif node_id == "retrieve_rag_documents":
                    node_colors.append('#F8B195')  # Orange for retrieval
                elif node_id == "summarize_query_research":
                    node_colors.append('#C06C84')  # Purple for summarization
                else:
                    node_colors.append('#95E1D3')  # Light green for regular nodes
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=3000, alpha=0.9)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color='gray', 
                              arrows=True, arrowsize=20, 
                              arrowstyle='->', alpha=0.6)
        
        # Draw labels
        labels = {node_id: data['label'] for node_id, data in G.nodes(data=True)}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')
        
        # Set title based on workflow type
        if workflow_type == "hitl":
            plt.title("RAG Deep Researcher v2.0 - Human-in-the-Loop Initial Workflow", 
                     fontsize=16, fontweight='bold', pad=20)
        elif workflow_type == "integrated":
            plt.title("RAG Deep Researcher v2.0 - Complete Integrated Workflow", 
                     fontsize=16, fontweight='bold', pad=20)
        else:
            plt.title("RAG Deep Researcher v2.0 - Main Research Workflow", 
                     fontsize=16, fontweight='bold', pad=20)
        plt.axis('off')
        plt.tight_layout()
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        plt.savefig(temp_file.name, dpi=300, bbox_inches='tight')
        plt.close()
        
        return temp_file.name
        
    except Exception as e:
        st.error(f"Error generating workflow visualization: {str(e)}")
        return create_mermaid_representation(researcher)

def generate_langgraph_visualization():
    """
    Generate PNG visualizations of the HITL and main LangGraph workflows using draw_mermaid_png().
    """
    try:
        from IPython.display import Image, display
        from src.graph_v2_0 import create_hitl_graph, create_main_graph
        
        # Get graph instances
        hitl_graph = create_hitl_graph()
        main_graph = create_main_graph()
        
        # Display in Streamlit
        st.subheader("🔄 Workflow Visualization")
        
        # Create tabs for different workflow visualizations
        tab1, tab2 = st.tabs(["HITL Workflow", "Main Research Workflow"])
        
        # HITL workflow tab
        with tab1:
            st.markdown("### Human-in-the-Loop (HITL) Workflow")
            st.markdown("This is the initial workflow that gathers human feedback before starting the main research process.")
            try:
                hitl_png = hitl_graph.get_graph().draw_mermaid_png()
                st.image(hitl_png)
            except Exception as e:
                st.error(f"Could not generate HITL workflow PNG: {str(e)}")
        
        # Main workflow tab
        with tab2:
            st.markdown("### Main Research Workflow")
            st.markdown("This is the main workflow that executes after the HITL process completes.")
            try:
                main_png = main_graph.get_graph().draw_mermaid_png()
                st.image(main_png)
            except Exception as e:
                st.error(f"Could not generate main workflow PNG: {str(e)}")
                
    except Exception as e:
        st.error(f"Error generating workflow visualization: {str(e)}")

def generate_workflow_visualization(researcher, return_mermaid_only=False):
    """
    Generate a visualization of the LangGraph workflow.
    If return_mermaid_only is True, it will only return the Mermaid representation.
    Otherwise, it returns the Mermaid representation.
    """
    return create_mermaid_representation(researcher)

# Import the HITL functions from correct locations
# detect_language is in basic_HITL_app.py, others are in src.graph_v2_0
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'dev'))
from basic_HITL_app import detect_language
from src.graph_v2_0 import analyse_user_feedback as _analyse_user_feedback
from src.graph_v2_0 import generate_follow_up_questions as _generate_follow_up_questions
from src.graph_v2_0 import generate_knowledge_base_questions as _generate_knowledge_base_questions
from langchain_core.runnables.config import RunnableConfig

# Create wrapper functions that handle the config parameter
def analyse_user_feedback(state):
    """Wrapper for analyse_user_feedback that handles config parameter"""
    config = RunnableConfig(configurable={
        "report_llm": state.get("report_llm", "deepseek-r1:latest"),
        "summarization_llm": state.get("summarization_llm", "deepseek-r1:latest")
    })
    return _analyse_user_feedback(state, config)

def generate_follow_up_questions(state):
    """Wrapper for generate_follow_up_questions that handles config parameter"""
    config = RunnableConfig(configurable={
        "report_llm": state.get("report_llm", "deepseek-r1:latest"),
        "summarization_llm": state.get("summarization_llm", "deepseek-r1:latest")
    })
    return _generate_follow_up_questions(state, config)

def generate_knowledge_base_questions(state):
    """Wrapper for generate_knowledge_base_questions that handles config parameter"""
    config = RunnableConfig(configurable={
        "report_llm": state.get("report_llm", "deepseek-r1:latest"),
        "summarization_llm": state.get("summarization_llm", "deepseek-r1:latest"),
        "max_search_queries": state.get("max_search_queries", 3)  # Add max_search_queries config
    })
    return _generate_knowledge_base_questions(state, config)

def initialize_hitl_state(user_query, report_llm, summarization_llm, max_search_queries=3):
    """
    Initialize HITL state following the basic_HITL_app.py pattern.
    """
    return {
        "user_query": user_query,
        "current_position": 0,
        "detected_language": "",
        "additional_context": "",  # Will store annotated conversation history
        "human_feedback": "",
        "analysis": "",
        "follow_up_questions": "",
        "report_llm": report_llm,
        "summarization_llm": summarization_llm,
        "max_search_queries": max_search_queries  # Add max_search_queries to state
    }

def process_initial_query(state):
    """
    Process the initial user query following basic_HITL_app.py pattern.
    """
    # Detect language
    with st.spinner("Detecting language..."):
        detected_language = detect_language(state)
        state["detected_language"] = detected_language
    
    # Generate initial follow-up questions
    with st.spinner("Generating follow-up questions..."):
        follow_up_result = generate_follow_up_questions(state)
        state["follow_up_questions"] = follow_up_result["follow_up_questions"]
    
    # For initial query, we don't have analysis yet
    state["analysis"] = ""
    
    # Format the combined response for initial query
    combined_response = f"FOLLOW-UP:\n {state['follow_up_questions']}"
    
    # Store initial AI questions in additional_context
    state["additional_context"] += f"Initial AI Question(s):\n{state['follow_up_questions']}"
    
    return combined_response

def process_human_feedback(state, human_feedback):
    """
    Process human feedback following basic_HITL_app.py pattern.
    """
    # Update state with human feedback
    state["human_feedback"] += human_feedback
    
    # Analyze user feedback first
    with st.spinner("Analyzing feedback..."):
        analysis_result = analyse_user_feedback(state)
        state["analysis"] = analysis_result["analysis"]
    
    # Generate follow-up questions
    with st.spinner("Generating follow-up questions..."):
        follow_up_result = generate_follow_up_questions(state)
        state["follow_up_questions"] = follow_up_result["follow_up_questions"]
    
    # Format the combined response
    combined_response = f"ANALYSIS: {state['analysis']}\n\nFOLLOW-UP:\n {state['follow_up_questions']}"
    
    # Store the conversation turn in additional_context
    conversation_turn = f"AI Question(s):\n{state['follow_up_questions']}\n\nHuman Answer:\n{human_feedback}"
    state["additional_context"] += conversation_turn
    
    return combined_response

def finalize_hitl_conversation(state):
    """
    Finalize HITL conversation and generate knowledge base questions.
    """
    # Generate knowledge base questions
    with st.spinner("Generating knowledge base questions..."):
        kb_questions_result = generate_knowledge_base_questions(state)
        kb_questions_content = kb_questions_result["research_queries"]  # Use research_queries for main workflow
    
    # IMPORTANT: Use the deep_analysis returned from generate_knowledge_base_questions
    # instead of the accumulated conversation history
    state["additional_context"] = kb_questions_result["additional_context"]  # This is the deep_analysis
    
    # Store research queries for main workflow handover
    state["research_queries"] = kb_questions_content if isinstance(kb_questions_content, list) else [kb_questions_content]
    
    return f"Based on our conversation, here are targeted knowledge base search questions:\n\n{kb_questions_content}"


def execute_main_workflow(enable_web_search, report_structure, max_search_queries, 
                         enable_quality_checker, quality_check_loops=1, 
                         use_ext_database=False, selected_database=None, k_results=3):
    """
    Execute the main research workflow using HITL results from session state.
    Returns the final research results.
    """
    if not st.session_state.hitl_result:
        st.error("No HITL results found. Please complete the HITL workflow first.")
        return None
    
    # Clear CUDA memory before starting
    clear_cuda_memory()
    
    # Configuration for the main graph
    config = {
        "configurable": {
            "report_structure": report_structure,
            "max_search_queries": max_search_queries,
            "report_llm": st.session_state.hitl_result["report_llm"],
            "summarization_llm": st.session_state.hitl_result["summarization_llm"],
            "enable_web_search": enable_web_search,
            "enable_quality_checker": enable_quality_checker,
            "quality_check_loops": quality_check_loops,
            "use_ext_database": use_ext_database,
            "selected_database": selected_database,
            "k_results": k_results
        }
    }
    
    try:
        # Initialize main workflow state using HITL results
        main_state = ResearcherStateV2(
            user_query=st.session_state.hitl_result["user_query"],
            current_position=st.session_state.hitl_result["current_position"],
            detected_language=st.session_state.hitl_result["detected_language"],
            research_queries=st.session_state.hitl_result["research_queries"],
            retrieved_documents={},
            search_summaries={},
            final_answer="",
            quality_check=None,
            additional_context=st.session_state.hitl_result["additional_context"],
            report_llm=st.session_state.hitl_result["report_llm"],
            summarization_llm=st.session_state.hitl_result["summarization_llm"],
            query_mapping=None,
            enable_quality_checker=enable_quality_checker,
            # Transfer HITL fields
            human_feedback=st.session_state.hitl_result["human_feedback"],
            analysis=st.session_state.hitl_result["analysis"],
            follow_up_questions=st.session_state.hitl_result["follow_up_questions"]
        )
        
        # Create progress tracking
        progress_container = st.container()
        with progress_container:
            st.subheader("🔬 Main Research Workflow")
            main_progress_bar = st.progress(0)
            main_status_text = st.empty()
            step_details = st.empty()
        
        # Define main workflow steps
        main_steps = ["retrieve_rag_documents", "update_position", "summarize_query_research", "generate_final_answer"]
        if enable_quality_checker:
            main_steps.append("quality_checker")
        
        # Execute main graph
        main_current_step = 0
        main_final_state = main_state  # Initialize with main_state to avoid None
        
        main_status_text.text("🚀 Starting main research workflow...")
        
        for step_output in main_graph.stream(main_state, config):
            # Update progress
            main_current_step += 1
            main_progress = min(main_current_step / len(main_steps), 1.0)
            main_progress_bar.progress(main_progress)
            
            # Update status based on current step
            if main_current_step == 1:
                main_status_text.text("🔍 Retrieving relevant documents...")
            elif main_current_step == 2:
                main_status_text.text("📍 Updating research position...")
            elif main_current_step == 3:
                main_status_text.text("📋 Summarizing research findings...")
                # Debug info for summarization
                summarization_info = st.empty()
            elif main_current_step == 4:
                main_status_text.text("✍️ Generating final answer...")
            elif main_current_step == 5 and enable_quality_checker:
                main_status_text.text("✅ Checking answer quality...")
            
            # Get the latest state from the step output
            print(f"  [DEBUG] Processing step output: {list(step_output.keys())}")
            for node_name, node_state in step_output.items():
                print(f"  [DEBUG] Processing node: {node_name}, state type: {type(node_state)}")
                if node_state is not None:  # Only update if node_state is not None
                    print(f"  [DEBUG] Updating main_final_state from {node_name}")
                    main_final_state = node_state
                    # Debug print the state keys if available
                    if hasattr(main_final_state, 'keys'):
                        try:
                            print(f"  [DEBUG] State keys: {list(main_final_state.keys())}")
                            if 'final_answer' in main_final_state:
                                print(f"  [DEBUG] Final answer length: {len(main_final_state['final_answer']) if main_final_state['final_answer'] else 0} chars")
                        except Exception as e:
                            print(f"  [DEBUG] Could not inspect state keys: {str(e)}")
                
                # Show summarization debugging info when summarize_query_research node runs
                if node_name == "summarize_query_research" and node_state is not None:
                    # Display summarization debugging info in the UI
                    try:
                        # Get the summarization LLM from session state
                        summarization_llm = st.session_state.get('summarization_llm', 'Unknown')
                        
                        # Get document counts if available
                        doc_count = 0
                        query_count = 0
                        if hasattr(node_state, 'get') and node_state.get('search_summaries'):
                            search_summaries = node_state.get('search_summaries')
                            if isinstance(search_summaries, dict):
                                query_count = len(search_summaries)
                                doc_count = sum(len(docs) for docs in search_summaries.values() if isinstance(docs, list))
                        
                        # Create debug message
                        debug_message = f"""**🔍 Summarization Debug Info:**
- Using LLM: `{summarization_llm}`
- Processing {query_count} research queries
- Found {query_count} x {st.session_state.k_results} total chunks
- Analysed {doc_count} summarized documents"""
                        
                        # Add research queries if available
                        if hasattr(node_state, 'get') and node_state.get('research_queries'):
                            research_queries = node_state.get('research_queries')
                            if isinstance(research_queries, list) and research_queries:
                                debug_message += f"\n\n**Current Research Queries ({len(research_queries)}):**\n"
                                for i, query in enumerate(research_queries):
                                    debug_message += f"\n{i+1}. {query}"
                        
                        # Display in the UI
                        summarization_info.markdown(debug_message)
                    except Exception as e:
                        print(f"[ERROR] Error displaying summarization debug info: {str(e)}")
                        summarization_info.error(f"Error displaying summarization debug info: {str(e)}")
                
                # Show retrieved documents after retrieve_rag_documents step
                if node_name == "retrieve_rag_documents" and node_state is not None and "retrieved_documents" in node_state:
                    with st.expander("📄 Retrieved Documents by Query", expanded=False):
                        retrieved_docs = node_state["retrieved_documents"]
                        research_queries = node_state.get("research_queries", [])
                        
                        if isinstance(retrieved_docs, dict):
                            for query_key, documents in retrieved_docs.items():
                                # Extract the actual query text (remove numbering prefix if present)
                                if ':' in query_key:
                                    query_display = query_key.split(':', 1)[1].strip()
                                else:
                                    query_display = query_key
                                
                                st.markdown(f"**Query:** {query_display}")
                                st.markdown(f"**Documents found:** {len(documents)}")
                                
                                # Show each document
                                for i, doc in enumerate(documents, 1):
                                    with st.expander(f"Document {i}", expanded=False):
                                        if hasattr(doc, 'page_content'):
                                            st.text_area(f"Content", doc.page_content, height=150, key=f"doc_{query_key}_{i}")
                                        if hasattr(doc, 'metadata'):
                                            st.json(doc.metadata)
                                        elif isinstance(doc, dict):
                                            st.json(doc)
                                st.divider()
                        else:
                            st.write("No retrieved documents found or unexpected format.")
            
            # Display step details
            step_details.info(f"Main Step {main_current_step}/{len(main_steps)} completed")
            time.sleep(0.1)
        
        # Complete main phase
        main_progress_bar.progress(1.0)
        main_status_text.text("✅ Research completed")
        
        # Use the final state from main workflow with defensive programming
        try:
            # Add detailed debugging for final state
            print(f"\n[DEBUG] === FINAL STATE PROCESSING START ===")
            print(f"[DEBUG] main_final_state type: {type(main_final_state)}")
            
            # Debug print the entire main_final_state if possible
            try:
                import json
                print(f"[DEBUG] main_final_state content: {json.dumps({k: str(v)[:200] + '...' if isinstance(v, (str, bytes)) else v 
                                 for k, v in main_final_state.items()}, indent=2, default=str)}")
            except Exception as e:
                print(f"[DEBUG] Could not serialize main_final_state: {str(e)}")
                
            if main_final_state is not None:
                print(f"[DEBUG] main_final_state keys: {list(main_final_state.keys()) if hasattr(main_final_state, 'keys') else 'No keys attribute'}")
                if hasattr(main_final_state, 'get') and 'final_answer' in main_final_state:
                    print(f"[DEBUG] Final answer exists, length: {len(main_final_state['final_answer']) if main_final_state['final_answer'] else 0} chars")
            else:
                print(f"[WARNING] main_final_state is None, falling back to main_state")
                print(f"[DEBUG] Main state type: {type(main_state)}")
                print(f"[DEBUG] Main state keys: {list(main_state.keys()) if hasattr(main_state, 'keys') else 'No keys attribute'}")
            
            # Safely get final state
            final_state = main_final_state if main_final_state is not None else main_state
            
            # Ensure final_state is a dictionary-like object
            if not hasattr(final_state, 'get') or not hasattr(final_state, 'keys'):
                print(f"[WARNING] Converting final_state to dictionary from {type(final_state)}")
                # Try different ways to convert to dict
                try:
                    if isinstance(final_state, dict):
                        pass  # Already a dict
                    elif hasattr(final_state, 'dict') and callable(getattr(final_state, 'dict')):
                        final_state = final_state.dict()
                    elif hasattr(final_state, '__dict__'):
                        final_state = final_state.__dict__
                    else:
                        # Last resort - try to create a dict from object attributes
                        final_state = {attr: getattr(final_state, attr) for attr in dir(final_state) 
                                     if not attr.startswith('_') and not callable(getattr(final_state, attr))}
                except Exception as e:
                    print(f"[ERROR] Error converting final_state to dict: {str(e)}")
                    final_state = {"final_answer": "Error: Could not process final state. See logs for details."}
                
                print(f"[DEBUG] After conversion, final_state type: {type(final_state)}")
                
        except Exception as e:
            print(f"[ERROR] Error in final state processing: {str(e)}", exc_info=True)
            final_state = {"final_answer": f"Error occurred during final state processing: {str(e)}. Check logs for details."}
        
        print(f"[DEBUG] === FINAL STATE PROCESSING COMPLETE ===\n")
        
        # Display results section
        st.subheader("📋 Research Results")
        
        # Display the final answer with robust error handling
        try:
            print("\n[DEBUG] === FINAL ANSWER RENDERING START ===")
            
            # Safely extract final_answer with multiple fallback methods
            final_answer = None
            
            # Method 1: Try direct attribute access
            try:
                if hasattr(final_state, 'get') and callable(final_state.get):
                    final_answer = final_state.get('final_answer')
                    print("[DEBUG] Extracted final_answer using .get() method")
            except Exception as e:
                print(f"[DEBUG] Error with .get() method: {str(e)}")
            
            # Method 2: Try dictionary access
            if final_answer is None and isinstance(final_state, dict):
                try:
                    final_answer = final_state.get('final_answer')
                    print("[DEBUG] Extracted final_answer using dict access")
                except Exception as e:
                    print(f"[DEBUG] Error with dict access: {str(e)}")
            
            # Method 3: Try attribute access
            if final_answer is None and hasattr(final_state, 'final_answer'):
                try:
                    final_answer = final_state.final_answer
                    print("[DEBUG] Extracted final_answer using attribute access")
                except Exception as e:
                    print(f"[DEBUG] Error with attribute access: {str(e)}")
            
            # If we still don't have an answer, try to convert to dict
            if final_answer is None:
                try:
                    if hasattr(final_state, 'dict') and callable(getattr(final_state, 'dict')):
                        final_state_dict = final_state.dict()
                        final_answer = final_state_dict.get('final_answer')
                        print("[DEBUG] Converted to dict and extracted final_answer")
                except Exception as e:
                    print(f"[DEBUG] Error converting to dict: {str(e)}")
            
            print(f"[DEBUG] Final answer type: {type(final_answer) if final_answer is not None else 'None'}")
            print(f"[DEBUG] Final answer preview: {str(final_answer)[:200]}..." if final_answer else "[DEBUG] No final answer found")
            
            # Display the final answer with error handling
            if final_answer:
                st.markdown("### 📄 Final Report")
                try:
                    # First try to render as markdown
                    st.markdown(final_answer, unsafe_allow_html=True)
                    print("[DEBUG] Successfully rendered final answer as markdown")
                except Exception as e:
                    print(f"[ERROR] Error rendering markdown: {str(e)}")
                    try:
                        # Fallback to text area if markdown fails
                        st.text_area("Final Report (Raw Text)", str(final_answer), height=400)
                        print("[DEBUG] Rendered final answer in text area")
                    except Exception as e2:
                        print(f"[ERROR] Error in text area fallback: {str(e2)}")
                        st.error("Could not display the final report. Please check the logs for details.")
                
                # Add copy to clipboard button with error handling
                if st.button("📋 Copy Report to Clipboard"):
                    try:
                        copy_to_clipboard(final_answer)
                        st.success("Report copied to clipboard!")
                    except Exception as e:
                        print(f"  [ERROR] Error copying to clipboard: {str(e)}")
                        st.error(f"Could not copy to clipboard: {str(e)}")
            else:
                st.error("No final report was generated. Check logs for errors.")
                print("  [ERROR] No final_answer found in final_state")
        except Exception as e:
            st.error(f"Error displaying final report: {str(e)}")
            print(f"  [ERROR] Exception in final answer display: {str(e)}")
        
        # Display quality check results if available
        if enable_quality_checker and "quality_check" in final_state and final_state["quality_check"]:
            quality_check = final_state["quality_check"]
            
            # Check if this is the new LLM-based assessment
            if quality_check.get("assessment_type") == "llm_fidelity_assessment":
                st.markdown("### 🔍 LLM-Based Quality Assessment")
                
                # Display score and pass/fail status
                overall_score = quality_check.get("overall_score", 0)
                max_score = quality_check.get("max_score", 400)
                passes_quality = quality_check.get("passes_quality", False)
                
                # Create columns for score display
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col1:
                    st.metric(
                        label="Overall Score",
                        value=f"{overall_score}/{max_score}",
                        delta=f"{overall_score - 300} from threshold" if overall_score != 0 else None
                    )
                
                with col2:
                    status_color = "🟢" if passes_quality else "🔴"
                    status_text = "PASS" if passes_quality else "FAIL"
                    st.metric(
                        label="Assessment Result",
                        value=f"{status_color} {status_text}"
                    )
                
                with col3:
                    threshold = quality_check.get("threshold", 300)
                    st.metric(
                        label="Pass Threshold",
                        value=f"{threshold}/{max_score}"
                    )
                
                # Display full assessment in expandable section
                with st.expander("📊 Detailed Fidelity Assessment", expanded=False):
                    full_assessment = quality_check.get("full_assessment", "No detailed assessment available.")
                    st.markdown(full_assessment)
                    
            else:
                # Legacy quality check display
                st.markdown("### ✅ Quality Check Results")
                st.info(quality_check)
        
        # Store results in session state
        st.session_state.research_results = final_state
        
        return final_state
        
    except Exception as e:
        st.error(f"Error in main workflow: {str(e)}")
        return None


def generate_response(user_input, enable_web_search, report_structure, max_search_queries, 
                     report_llm, enable_quality_checker, quality_check_loops=1, 
                     use_ext_database=False, selected_database=None, k_results=3,
                     human_feedback="", additional_context=""):
    """
    Simplified response generation that delegates to appropriate workflow based on phase.
    This function is kept for backward compatibility but now uses the new two-phase approach.
    """
    
    # Check current workflow phase
    if st.session_state.workflow_phase == "hitl":
        # Execute HITL workflow
        success = execute_hitl_workflow(user_input, report_llm, additional_context, human_feedback)
        if success:
            st.session_state.workflow_phase = "main"  # Move to main phase
        return None
    else:
        # Execute main workflow
        return execute_main_workflow(enable_web_search, report_structure, max_search_queries, 
                                   enable_quality_checker, quality_check_loops, 
                                   use_ext_database, selected_database, k_results)

def clear_chat():
    """Clear the chat history and reset session state"""
    keys_to_clear = [
        'messages', 'research_results', 'current_query', 'hitl_feedback',
        'hitl_analysis', 'hitl_questions', 'hitl_context', 'hitl_result',
        'hitl_conversation_history', 'hitl_state', 'waiting_for_human_input',
        'conversation_ended', 'input_counter'
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    
    # Reset workflow phase to HITL
    st.session_state.workflow_phase = "hitl"
    st.rerun()

def copy_to_clipboard(text):
    """Safely copy text to clipboard if pyperclip is available"""
    try:
        import pyperclip
        pyperclip.copy(text)
        return True
    except ImportError:
        st.warning("pyperclip not available. Please install it for clipboard functionality.")
        return False

def main():
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "research_results" not in st.session_state:
        st.session_state.research_results = None
    
    if "current_query" not in st.session_state:
        st.session_state.current_query = ""
    
    # HITL session state
    if "hitl_feedback" not in st.session_state:
        st.session_state.hitl_feedback = ""
    
    if "hitl_analysis" not in st.session_state:
        st.session_state.hitl_analysis = ""
    
    if "hitl_questions" not in st.session_state:
        st.session_state.hitl_questions = ""
    
    if "hitl_context" not in st.session_state:
        st.session_state.hitl_context = ""
    
    # Session state for storing HITL results (similar to test_st-multigraph.py)
    if "hitl_result" not in st.session_state:
        st.session_state.hitl_result = None
    
    # Workflow phase tracking
    if "workflow_phase" not in st.session_state:
        st.session_state.workflow_phase = "hitl"  # "hitl" or "main"
    
    # Model selection session state
    if "report_llm" not in st.session_state:
        report_llm_models = get_report_llm_models()
        # Set default to the first model in the list (from report_llms.md)
        st.session_state.report_llm = report_llm_models[0] if report_llm_models else "deepseek-r1:latest"
    
    if "summarization_llm" not in st.session_state:
        summarization_llm_models = get_summarization_llm_models()
        # Set default to qwen3:1.7b if available, otherwise first model
        if "qwen3:1.7b" in summarization_llm_models:
            st.session_state.summarization_llm = "qwen3:1.7b"
        else:
            st.session_state.summarization_llm = summarization_llm_models[0] if summarization_llm_models else "deepseek-r1:latest"
    
    # Create header with two columns (matching app_v1_1.py)
    header_col1, header_col2 = st.columns([0.6, 0.4])
    with header_col1:
        st.markdown(
    '<h1>🔍 Br<span style="color:darkorange;"><b>AI</b></span>n: Human-In-The-Loop (HITL) RAG Researcher V2.0</h1>',
    unsafe_allow_html=True
)
        # Add license information under the title (exact implementation from basic_HITL_app.py)
        st.markdown('<p style="font-size:12px; font-weight:bold; color:darkorange; margin-top:0px;">LICENCE</p>', 
                    unsafe_allow_html=True, help=get_license_content())
    with header_col2:
        st.image("Header für Chatbot.png", use_container_width=True)
    
    # Load model options from global configuration
    report_llm_models = get_report_llm_models()
    summarization_llm_models = get_summarization_llm_models()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model Selection
        st.subheader("🤖 Model Selection")
        
        # Report writing LLM
        st.session_state.report_llm = st.sidebar.selectbox(
            "Report Writing LLM",
            options=report_llm_models,
            index=report_llm_models.index(st.session_state.report_llm) if st.session_state.report_llm in report_llm_models else 0,
            help="Choose the LLM model to use for final report generation; loaded from global report_llms.md configuration"
        )
        
        # Summarization LLM
        st.session_state.summarization_llm = st.sidebar.selectbox(
            "Summarization LLM",
            options=summarization_llm_models,
            index=summarization_llm_models.index(st.session_state.summarization_llm) if st.session_state.summarization_llm in summarization_llm_models else (summarization_llm_models.index("qwen3:1.7b") if "qwen3:1.7b" in summarization_llm_models else 0),
            help="Choose the LLM model to use for document summarization; loaded from global summarization_llms.md configuration"
        )
        
        st.divider()
        
        # Research Configuration
        st.subheader("🔬 Research Settings")
        
        # Report structure
        report_structures = get_report_structures()
        report_structure = st.selectbox(
            "Report Structure",
            options=list(report_structures.keys()),
            index=0,
            help="Choose the structure for the final report"
        )
        
        # Max search queries
        max_search_queries = st.slider(
            "Number of Additional Research Queries",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of additional research queries to generate"
        )
        
        # Enable web search
        enable_web_search = st.checkbox(
            "Enable Web Search",
            value=False,
            help="Enable web search in addition to local document retrieval"
        )
        
        # Quality checker settings
        enable_quality_checker = st.checkbox(
            "Enable Quality Checker",
            value=False,
            help="Enable quality checking and improvement of the final report"
        )
        
        if enable_quality_checker:
            quality_check_loops = st.slider(
                "Quality Check Iterations",
                min_value=1,
                max_value=3,
                value=1,
                help="Number of quality check and improvement iterations"
            )
        else:
            quality_check_loops = 1
        
        st.divider()
        
        # External Database Configuration (matching app_v1_1.py)
        st.subheader("🗄️ External Database")
        
        # Define DATABASE_PATH like in app_v1_1.py
        DATABASE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "kb", "database")
        
        # Initialize session state for external database
        if "use_ext_database" not in st.session_state:
            st.session_state.use_ext_database = False
        if "selected_database" not in st.session_state:
            st.session_state.selected_database = ""
        if "k_results" not in st.session_state:
            st.session_state.k_results = 3
        
        # Enable external database checkbox
        st.session_state.use_ext_database = st.sidebar.checkbox(
            "Use ext. Database", 
            value=st.session_state.use_ext_database,
            help="Use an existing database for document retrieval"
        )
        
        # Database selection
        if st.session_state.use_ext_database:
            # Get available databases
            database_dir = Path(DATABASE_PATH)
            database_options = [d.name for d in database_dir.iterdir() if d.is_dir()] if database_dir.exists() else []
            
            if database_options:
                # Select database
                selected_db = st.sidebar.selectbox(
                    "Select Database",
                    options=database_options,
                    index=database_options.index(st.session_state.selected_database) if st.session_state.selected_database in database_options else 0,
                    help="Choose a database to use for retrieval"
                )
                st.session_state.selected_database = selected_db
                
                # Extract and update embedding model from database name
                embedding_model_name = extract_embedding_model(selected_db)
                if embedding_model_name:
                    # Update the global configuration to use this embedding model
                    from src.configuration_v1_1 import update_embedding_model
                    update_embedding_model(embedding_model_name)
                    st.sidebar.info(f"Selected Database: {selected_db}")
                    st.sidebar.success(f"Updated embedding model to: {embedding_model_name}")
                
                # Number of results to retrieve
                st.session_state.k_results = st.sidebar.slider(
                    "Number of results to retrieve", 
                    min_value=1, 
                    max_value=10, 
                    value=st.session_state.k_results
                )
                
                selected_database = st.session_state.selected_database
                k_results = st.session_state.k_results
            else:
                st.sidebar.warning("No databases found. Please upload documents first.")
                st.session_state.use_ext_database = False
                selected_database = None
                k_results = 3
        else:
            selected_database = None
            k_results = 3
            
        use_ext_database = st.session_state.use_ext_database
        
        st.divider()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat", help="Clear all chat history and reset the session"):
            clear_chat()

    
    # Initialize HITL session state
    if "hitl_state" not in st.session_state:
        st.session_state.hitl_state = None
    if "hitl_conversation_history" not in st.session_state:
        st.session_state.hitl_conversation_history = []
    if "waiting_for_human_input" not in st.session_state:
        st.session_state.waiting_for_human_input = False
    if "conversation_ended" not in st.session_state:
        st.session_state.conversation_ended = False
    if "input_counter" not in st.session_state:
        st.session_state.input_counter = 0
    
    # Workflow Visualization Expander (moved here to be visible from the beginning)
    with st.expander("🔄 Show Workflow Graphs", expanded=False):
        st.markdown("### RAG Deep Researcher v2.0 - Workflow Graphs")
        
        # Display embedding model information
        from src.configuration_v1_1 import get_config_instance
        config_instance = get_config_instance()
        st.info(f"**Embedding Model:** {config_instance.embedding_model}")
        
        # Create two columns for HITL and Main graphs (side-by-side)
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🤝 Human-in-the-Loop (HITL) Workflow")
            try:
                # Generate HITL graph visualization
                hitl_png = hitl_graph.get_graph().draw_mermaid_png()
                st.image(hitl_png, caption="HITL Workflow Graph", width=350)
            except Exception as e:
                st.error(f"Could not generate HITL graph visualization: {str(e)}")
        
        with col2:
            st.markdown("#### 🔬 Main Research Workflow")
            try:
                # Generate main graph visualization
                main_png = main_graph.get_graph().draw_mermaid_png()
                st.image(main_png, caption="Main Research Workflow Graph", width=350)
            except Exception as e:
                st.error(f"Could not generate main graph visualization: {str(e)}")
    
    # Initialize session state for robust tab switching
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "HITL Phase"
    
    # Define tab options
    if st.session_state.workflow_phase == "hitl":
        tab_options = ["📝 HITL Phase (Active)", "🔬 Main Research Phase"]
        # Automatically switch to Main Research when HITL completes
        if st.session_state.active_tab == "Main Research Phase":
            # User manually switched, keep their choice
            pass
        else:
            st.session_state.active_tab = "HITL Phase"
    else:
        tab_options = ["📝 HITL Phase (Completed)", "🔬 Main Research Phase (Active)"]
        # Automatically switch to Main Research when workflow phase changes
        if st.session_state.active_tab == "HITL Phase":
            st.session_state.active_tab = "Main Research Phase"
            # Add a visual indicator that we've automatically switched
            if "phase_switch_notified" not in st.session_state:
                st.success("✨ **Automatically switched to Main Research Phase!** The HITL phase has been completed.")
                st.session_state.phase_switch_notified = True
    
    # Create radio button for tab selection with automatic switching
    selected_tab = st.radio(
        "Select workflow phase:",
        tab_options,
        index=tab_options.index([opt for opt in tab_options if "Main Research" in opt][0]) if st.session_state.active_tab == "Main Research Phase" else 0,
        key="workflow_tab_radio",
        horizontal=True
    )
    
    # Update session state based on selection
    if "HITL" in selected_tab:
        st.session_state.active_tab = "HITL Phase"
    else:
        st.session_state.active_tab = "Main Research Phase"
    
    # HITL Phase Content
    if st.session_state.active_tab == "HITL Phase":
        if st.session_state.workflow_phase == "hitl":
            st.info("📝 **Current Phase: Human-in-the-Loop** - Interactive conversation to refine your research needs.")
            
            # Initialize HITL session state variables if they don't exist
            if "hitl_conversation_history" not in st.session_state:
                st.session_state.hitl_conversation_history = []
            
            if "hitl_state" not in st.session_state:
                st.session_state.hitl_state = None
            
            if "waiting_for_human_input" not in st.session_state:
                st.session_state.waiting_for_human_input = False
            
            if "conversation_ended" not in st.session_state:
                st.session_state.conversation_ended = False
            
            # Initial query input (following basic_HITL_app.py pattern)
            if not st.session_state.hitl_state:
                st.markdown("""
                
                This system will first ask you **clarifying questions** to better understand your research needs,
                then proceed with enhanced document retrieval and report generation.
                
                Type `/end` at any point to finish the conversation and proceed to main research.
                """)
                
                user_query = st.text_area(
                    "Enter your initial research query:", 
                    height=100,
                    placeholder="e.g., 'What are the latest developments in quantum computing and their potential applications in cryptography?'"
                )
                submit_button = st.button("🚀 Submit Query", type="primary")
                
                if submit_button and user_query:
                    # Initialize the HITL state
                    st.session_state.hitl_state = initialize_hitl_state(
                        user_query, 
                        st.session_state.report_llm, 
                        st.session_state.summarization_llm,
                        max_search_queries  # Pass the max_search_queries parameter
                    )
                    
                    # Add user message to conversation history
                    st.session_state.hitl_conversation_history.append({
                        "role": "user",
                        "content": user_query
                    })
                    
                    # Process initial query
                    combined_response = process_initial_query(st.session_state.hitl_state)
                    
                    # Add AI message to conversation history
                    st.session_state.hitl_conversation_history.append({
                        "role": "assistant",
                        "content": combined_response
                    })
                    
                    # Set waiting for human input
                    st.session_state.waiting_for_human_input = True
                    
                    # Force a rerun to update the UI
                    st.rerun()
            
            # Display conversation history
            if st.session_state.hitl_conversation_history:
                st.subheader("💬 Conversation History")
                for message in st.session_state.hitl_conversation_history:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
            
            # Display debug information about the current state
            if st.session_state.hitl_state:
                with st.expander("Debug: Current HITL State", expanded=False):
                    # Create a deep copy of the state to display
                    display_state = {}
                    for key, value in st.session_state.hitl_state.items():
                        display_state[key] = value
                    st.json(display_state)
            
            # Handle human feedback (following basic_HITL_app.py pattern)
            if st.session_state.waiting_for_human_input and not st.session_state.conversation_ended:
                # Use a dynamic key that changes after each submission to force widget reset
                human_feedback = st.text_area(
                    "Your response (type /end to finish and proceed to main research):", 
                    value="", 
                    height=100, 
                    key=f"human_feedback_area_{st.session_state.input_counter}"
                )
                submit_feedback_button = st.button("Submit Response")
                
                if submit_feedback_button and human_feedback:
                    # Check if the user wants to end the conversation
                    if human_feedback.strip().lower() == "/end":
                        st.session_state.conversation_ended = True
                        
                        # Add user message to conversation history
                        st.session_state.hitl_conversation_history.append({
                            "role": "user",
                            "content": "/end - Conversation ended"
                        })
                        
                        # Finalize HITL conversation
                        final_response = finalize_hitl_conversation(st.session_state.hitl_state)
                        
                        # Add AI message to conversation history
                        st.session_state.hitl_conversation_history.append({
                            "role": "assistant",
                            "content": final_response
                        })
                        
                        # Store HITL results in session state for handover to main workflow
                        st.session_state.hitl_result = {
                            "user_query": st.session_state.hitl_state["user_query"],
                            "current_position": st.session_state.hitl_state["current_position"],
                            "detected_language": st.session_state.hitl_state["detected_language"],
                            "additional_context": st.session_state.hitl_state["additional_context"],
                            "report_llm": st.session_state.hitl_state["report_llm"],
                            "summarization_llm": st.session_state.hitl_state["summarization_llm"],
                            "research_queries": st.session_state.hitl_state.get("research_queries", []),
                            "analysis": st.session_state.hitl_state["analysis"],
                            "follow_up_questions": st.session_state.hitl_state["follow_up_questions"],
                            "human_feedback": st.session_state.hitl_state["human_feedback"]
                        }
                        
                        # Move to main workflow phase
                        st.session_state.workflow_phase = "main"
                        
                        # Increment input counter to reset widgets
                        st.session_state.input_counter += 1
                        st.rerun()
                    else:
                        # Add user message to conversation history
                        st.session_state.hitl_conversation_history.append({
                            "role": "user",
                            "content": human_feedback
                        })
                        
                        # Process human feedback
                        combined_response = process_human_feedback(st.session_state.hitl_state, human_feedback)
                        
                        # Add AI message to conversation history
                        st.session_state.hitl_conversation_history.append({
                            "role": "assistant",
                            "content": combined_response
                        })
                        
                        # Increment input counter to reset widgets
                        st.session_state.input_counter += 1
                        st.rerun()
        else:
            # HITL phase completed - show summary without conversation history
            st.markdown("📋 HITL Phase Results (Completed)")
            st.success("✅ HITL Phase completed successfully!")
                
            if st.session_state.hitl_result:
                st.markdown("### 📝 Query Analysis")
                st.write(f"**Original Query:** {st.session_state.hitl_result['user_query']}")
                st.write(f"**Detected Language:** {st.session_state.hitl_result['detected_language']}")
                
                # Display deep analysis and knowledge base questions following basic_HITL_app.py pattern
                if 'additional_context' in st.session_state.hitl_result and st.session_state.hitl_result['additional_context']:
                    st.markdown("### 🧠 Deep Analysis of Your Information Needs")
                    st.write(st.session_state.hitl_result['additional_context'])
                    
                    st.markdown("### 🎯 Targeted Knowledge Base Search Questions")
                    st.write("Based on our conversation and the analysis above, here are targeted knowledge base search questions:")
                    # Display the research queries in a formatted way
                    for i, query in enumerate(st.session_state.hitl_result['research_queries'], 1):
                        st.write(f"{i}. {query}")
                
                # Note: Conversation history is preserved in session state but not displayed in GUI during Main phase
                st.info("💬 Conversation history has been preserved but is hidden during the Main Research phase for a cleaner interface.")
    
    # Main Research Phase Content
    elif st.session_state.active_tab == "Main Research Phase":
        if st.session_state.workflow_phase == "main":
            st.info("🔬 **Current Phase: Main Research** - The system will now execute the full research workflow using your HITL input.")
            with st.expander("### 🧠 Deep Analysis of Your Information Needs"):
                st.write(st.session_state.hitl_result['additional_context'])
                st.markdown("### 🎯 Targeted Knowledge Base Search Questions")
                st.write("Based on our conversation and the analysis above, here are targeted knowledge base search questions:")
                # Display the research queries in a formatted way
                for i, query in enumerate(st.session_state.hitl_result['research_queries'], 1):
                    st.write(f"{i}. {query}")
            # Main Research Phase
            if not st.session_state.hitl_result:
                st.error("No HITL results found. Please restart and complete the HITL phase first.")
                if st.button("🔄 Restart HITL Phase"):
                    st.session_state.workflow_phase = "hitl"
                    st.rerun()
                return
            
            # Execute main research workflow automatically
            if not st.session_state.research_results:
                st.info("🔬 Starting main research workflow automatically...")
                with st.spinner("Executing main research workflow..."):
                    results = execute_main_workflow(
                        enable_web_search=enable_web_search,
                        report_structure=report_structure,
                        max_search_queries=max_search_queries,
                        enable_quality_checker=enable_quality_checker,
                        quality_check_loops=quality_check_loops,
                        use_ext_database=use_ext_database,
                        selected_database=selected_database,
                        k_results=k_results
                    )
                    
                    if results:
                        st.session_state.research_results = results
            else:
                st.success("✅ Main research workflow completed! Results are displayed above.")
                
                # Option to restart
                if st.button("🔄 Start New Research"):
                    clear_chat()
                    st.rerun()
        else:
            # Main phase not active - show message
            st.info("🕰️ Waiting for HITL phase to complete before starting main research...")
    
    # Display chat history (outside of tabs)
    if st.session_state.messages:
        st.subheader("📝 Research History")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

if __name__ == "__main__":
    main()
