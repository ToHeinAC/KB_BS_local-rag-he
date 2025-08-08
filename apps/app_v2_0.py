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
from src.graph_retrieval_summarization import create_retrieval_summarization_graph
from src.utils_v1_1 import get_report_structures, process_uploaded_files, clear_cuda_memory
from src.rag_helpers_v1_1 import (
    get_report_llm_models, 
    get_summarization_llm_models, 
    get_all_available_models,
    get_license_content,
    extract_embedding_model
)

# Set page configuration
st.set_page_config(
    page_title="RAG Deep Researcher v2.0 - Human-in-the-Loop (HITL)",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

clear_cuda_memory()
# Function to clean model names for display
def clean_model_name(model_name):
    """Clean model name by removing common prefixes and suffixes for better display"""
    return model_name.replace(":latest", "").replace("_", " ").title()


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
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'dev'))
from basic_HITL_app import detect_language

# Import from hyphenated filename using importlib
import importlib.util
rerank_reporter_path = os.path.join(os.path.dirname(__file__), '..', 'dev', 'basic_rerank-reporter_app.py')
spec = importlib.util.spec_from_file_location("basic_rerank_reporter_app", rerank_reporter_path)
basic_rerank_reporter_app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(basic_rerank_reporter_app)
create_rerank_reporter_graph = basic_rerank_reporter_app.create_rerank_reporter_graph
from src.graph_v2_0 import (
    analyse_user_feedback as _analyse_user_feedback,
    generate_follow_up_questions as _generate_follow_up_questions,
    generate_knowledge_base_questions as _generate_knowledge_base_questions
)
from src.graph_retrieval_summarization import create_retrieval_summarization_graph
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


def execute_retrieval_summarization_phase(use_ext_database=False, selected_database=None, k_results=3):
    """
    Execute Phase 2: Retrieval-Summarization workflow using HITL results from session state.
    Returns the state with retrieved documents and summaries.
    """
    if not st.session_state.hitl_result:
        st.error("No HITL results found. Please complete the HITL workflow first.")
        return None
    
    # Clear CUDA memory before starting
    clear_cuda_memory()
    
    # Configuration for the retrieval-summarization graph
    config = {
        "configurable": {
            "use_ext_database": use_ext_database,
            "selected_database": selected_database,
            "k_results": k_results
        }
    }
    
    try:
        # Initialize retrieval-summarization state using HITL results
        retrieval_state = ResearcherStateV2(
            user_query=st.session_state.hitl_result["user_query"],
            current_position=0,
            detected_language=st.session_state.hitl_result["detected_language"],
            research_queries=st.session_state.hitl_result["research_queries"],
            retrieved_documents={},
            search_summaries={},
            final_answer="",
            quality_check=None,
            additional_context=st.session_state.hitl_result["additional_context"],
            report_llm=st.session_state.hitl_result["report_llm"],
            summarization_llm=st.session_state.hitl_result["summarization_llm"],
            enable_quality_checker=False,  # Not used in this phase
            # Transfer HITL fields
            human_feedback=st.session_state.hitl_result["human_feedback"],
            analysis=st.session_state.hitl_result["analysis"],
            follow_up_questions=st.session_state.hitl_result["follow_up_questions"]
        )
        
        # Create progress tracking
        progress_container = st.container()
        with progress_container:
            st.subheader("🔍 Phase 2: Retrieval & Summarization")
            retrieval_progress_bar = st.progress(0)
            retrieval_status_text = st.empty()
        
        # Create retrieval-summarization graph
        retrieval_graph = create_retrieval_summarization_graph()
        
        # Execute retrieval-summarization graph
        retrieval_status_text.text("🔍 Retrieving relevant documents...")
        
        retrieval_final_state = retrieval_state
        step_count = 0
        total_steps = 2  # retrieve_documents, summarize_documents
        
        for step_output in retrieval_graph.stream(retrieval_state, config):
            step_count += 1
            progress = min(step_count / total_steps, 1.0)
            retrieval_progress_bar.progress(progress)
            
            # Update status based on current step
            if step_count == 1:
                retrieval_status_text.text("📋 Summarizing research findings...")
            elif step_count == 2:
                retrieval_status_text.text("✅ Retrieval and summarization completed")
            
            # Get the latest state from the step output
            for node_name, node_state in step_output.items():
                if node_state is not None:
                    retrieval_final_state = node_state
        
        # Complete retrieval-summarization phase
        retrieval_progress_bar.progress(1.0)
        retrieval_status_text.text("✅ Retrieval and summarization completed")
        
        # Display results
        st.subheader("📋 Phase 2 Results")
        
        # Display retrieved documents
        if "retrieved_documents" in retrieval_final_state and retrieval_final_state["retrieved_documents"]:
            with st.expander("📄 Retrieved Documents by Query", expanded=False):
                retrieved_docs = retrieval_final_state["retrieved_documents"]
                for query, documents in retrieved_docs.items():
                    st.markdown(f"**Query:** {query}")
                    st.markdown(f"**Documents found:** {len(documents)}")
                    
                    for i, doc in enumerate(documents, 1):
                        with st.expander(f"Document {i}", expanded=False):
                            if hasattr(doc, 'page_content'):
                                st.text_area(f"Content", doc.page_content, height=150, key=f"ret_doc_{hash(query)}_{i}")
                            if hasattr(doc, 'metadata'):
                                st.json(doc.metadata)
                    st.divider()
        
        # Display summaries
        if "search_summaries" in retrieval_final_state and retrieval_final_state["search_summaries"]:
            search_summaries = retrieval_final_state["search_summaries"]
            for query, summaries in search_summaries.items():
                st.markdown(f"**Number of summaries for query {query}:** {len(summaries)}")
            with st.expander("📝 Generated Summaries", expanded=False):    
                for query, summaries in search_summaries.items():
                    st.markdown(f"**Query:** {query}")
                    for i, summary in enumerate(summaries, 1):
                        if hasattr(summary, 'page_content'):
                            st.markdown(f"**Summary {i}:**")
                            st.markdown(summary.page_content)
                        elif isinstance(summary, str):
                            st.markdown(f"**Summary {i}:**")
                            st.markdown(summary)
                    st.divider()
            
        # Store results in session state for next phase
        st.session_state.retrieval_summarization_result = retrieval_final_state
        
        return retrieval_final_state
        
    except Exception as e:
        st.error(f"Error in retrieval-summarization phase: {str(e)}")
        print(f"[ERROR] Retrieval-summarization phase error: {str(e)}")
        return None


def execute_reporting_phase(enable_web_search=False):
    """
    Execute Phase 3: Reporting workflow using results from Phase 2.
    Returns the final research results with reranked summaries and final report.
    """
    if not st.session_state.retrieval_summarization_result:
        st.error("No retrieval-summarization results found. Please complete Phase 2 first.")
        return None
    
    # Clear CUDA memory before starting
    clear_cuda_memory()
    
    try:
        # Initialize reporting state using Phase 2 results
        reporting_state = ResearcherStateV2(
            **st.session_state.retrieval_summarization_result,
            # Add reporting-specific fields
            web_search_enabled=enable_web_search,
            all_reranked_summaries=None,
            reflection_count=0,
            internet_result=None,
            internet_search_term=None
        )
        
        # Create progress tracking
        progress_container = st.container()
        with progress_container:
            st.subheader("📊 Phase 3: Reranking & Report Generation")
            reporting_progress_bar = st.progress(0)
            reporting_status_text = st.empty()
        
        # Create rerank-reporter graph
        reporting_graph = create_rerank_reporter_graph()
        
        # Execute reporting graph
        reporting_status_text.text("🚀 Starting reranking and report generation...")
        
        reporting_final_state = reporting_state
        step_count = 0
        
        for step_output in reporting_graph.stream(reporting_state):
            step_count += 1
            progress = min(step_count / 4, 1.0)  # Approximate steps: reranker, report_writer, quality_checker
            reporting_progress_bar.progress(progress)
            
            # Update status based on current step
            for node_name, node_state in step_output.items():
                if node_name == "reranker":
                    reporting_status_text.text("🔄 Reranking summaries by relevance...")
                elif node_name == "web_tavily_searcher":
                    reporting_status_text.text("🌐 Searching the internet for additional information...")
                elif node_name == "report_writer":
                    reporting_status_text.text("✍️ Generating final report...")
                elif node_name == "quality_checker":
                    reporting_status_text.text("✅ Checking report quality...")
                
                if node_state is not None:
                    reporting_final_state = node_state
        
        # Complete reporting phase
        reporting_progress_bar.progress(1.0)
        reporting_status_text.text("✅ Report generation completed")
        
        # Display results
        st.subheader("📋 Phase 3 Results")
        
        # Display reranked summaries
        if "all_reranked_summaries" in reporting_final_state and reporting_final_state["all_reranked_summaries"]:
            with st.expander("🏆 Reranked Summaries", expanded=False):
                reranked_summaries = reporting_final_state["all_reranked_summaries"]
                for i, summary in enumerate(reranked_summaries, 1):
                    score = summary.get('score', 0)
                    query = summary.get('query', 'Unknown')
                    content = summary.get('content', 'No content')
                    
                    st.markdown(f"**Rank #{i} - Score: {score:.1f}**")
                    st.markdown(f"**Query:** {query}")
                    with st.expander(f"Summary Content", expanded=False):
                        st.markdown(content)
                    st.divider()
        
        # Display internet search results if available
        if enable_web_search and "internet_result" in reporting_final_state and reporting_final_state["internet_result"]:
            with st.expander("🌐 Internet Search Results", expanded=False):
                if "internet_search_term" in reporting_final_state and reporting_final_state["internet_search_term"]:
                    st.markdown(f"🔍 **Generated Search Term:** `{reporting_final_state['internet_search_term']}`")
                st.markdown(reporting_final_state["internet_result"])
        
        # Display final report
        if "final_answer" in reporting_final_state and reporting_final_state["final_answer"]:
            st.markdown("### 📄 Final Report")
            st.markdown(reporting_final_state["final_answer"])
            
            # Add copy to clipboard button
            if st.button("📋 Copy Report to Clipboard"):
                try:
                    copy_to_clipboard(reporting_final_state["final_answer"])
                    st.success("Report copied to clipboard!")
                except Exception as e:
                    st.error(f"Could not copy to clipboard: {str(e)}")
        
        # Display quality check results if available
        if "quality_check" in reporting_final_state and reporting_final_state["quality_check"]:
            quality_check = reporting_final_state["quality_check"]
            
            # Check if this is the new LLM-based assessment
            if quality_check.get("assessment_type") == "llm_fidelity_assessment":
                st.markdown("### 🔍 Quality Assessment")
                
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
                with st.expander("📊 Detailed Assessment", expanded=False):
                    full_assessment = quality_check.get("full_assessment", "No detailed assessment available.")
                    st.markdown(full_assessment)
        
        # Clean the final answer from <think> blocks if present
        if "final_answer" in reporting_final_state and reporting_final_state["final_answer"]:
            import re
            raw_answer = reporting_final_state["final_answer"]
            
            # Remove <think> blocks from the final answer
            clean_answer = re.sub(r'<think>.*?(?:</think>|<think>)', '', raw_answer, flags=re.DOTALL | re.IGNORECASE)
            clean_answer = clean_answer.strip()
            
            # Store both raw and clean versions
            reporting_final_state["final_answer_raw"] = raw_answer
            reporting_final_state["final_answer"] = clean_answer
        
        # Store final results in session state
        st.session_state.reporting_result = reporting_final_state
        st.session_state.research_results = reporting_final_state  # Keep for backward compatibility
        st.session_state.workflow_phase = "completed"  # Mark workflow as completed
        
        return reporting_final_state
        
    except Exception as e:
        st.error(f"Error in reporting phase: {str(e)}")
        print(f"[ERROR] Reporting phase error: {str(e)}")
        return None


def generate_response(user_input, enable_web_search, report_structure, max_search_queries, 
                     report_llm, enable_quality_checker, quality_check_loops=1, 
                     use_ext_database=False, selected_database=None, k_results=3,
                     human_feedback="", additional_context=""):
    """
    Simplified response generation that delegates to appropriate workflow based on phase.
    This function is kept for backward compatibility but now uses the new three-phase approach.
    """
    
    # Check current workflow phase
    if st.session_state.workflow_phase == "hitl":
        # HITL phase is handled in the main GUI
        return None
    elif st.session_state.workflow_phase == "retrieval_summarization":
        # Execute retrieval-summarization phase
        return execute_retrieval_summarization_phase(use_ext_database, selected_database, k_results)
    elif st.session_state.workflow_phase == "reporting":
        # Execute reporting phase
        return execute_reporting_phase(enable_web_search)
    else:
        st.error(f"Unknown workflow phase: {st.session_state.workflow_phase}")
        return None


def clear_chat():
    """Clear the chat history and reset session state"""
    keys_to_clear = [
        'messages', 'research_results', 'current_query', 'hitl_feedback',
        'hitl_analysis', 'hitl_questions', 'hitl_context', 'hitl_result',
        'hitl_conversation_history', 'hitl_state', 'waiting_for_human_input',
        'conversation_ended', 'input_counter', 'retrieval_summarization_result'
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
    
    # Input field state control - use processing flags instead of visibility flags
    if "processing_initial_query" not in st.session_state:
        st.session_state.processing_initial_query = False
    
    if "processing_feedback" not in st.session_state:
        st.session_state.processing_feedback = False
    
    # Session state for storing HITL results (similar to test_st-multigraph.py)
    if "hitl_result" not in st.session_state:
        st.session_state.hitl_result = None
    
    # Session state for storing phase results
    if "retrieval_summarization_result" not in st.session_state:
        st.session_state.retrieval_summarization_result = None
    
    if "reporting_result" not in st.session_state:
        st.session_state.reporting_result = None
    
    # Workflow phase tracking
    if "workflow_phase" not in st.session_state:
        st.session_state.workflow_phase = "hitl"  # "hitl", "retrieval_summarization", "reporting"
    
    # Model selection session state
    if "report_llm" not in st.session_state:
        report_llm_models = get_report_llm_models()
        # Set default to the first model in the list (from report_llms.md)
        st.session_state.report_llm = report_llm_models[0] if report_llm_models else "gpt-oss:20b"
    
    if "summarization_llm" not in st.session_state:
        summarization_llm_models = get_summarization_llm_models()
        # Set default to qwen3:latest if available, otherwise first model
        if "qwen3:latest" in summarization_llm_models:
            st.session_state.summarization_llm = "qwen3:latest"
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
        # Define DATABASE_PATH like in app_v1_1.py
        DATABASE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "kb", "database")
    
        with st.expander("🗄️ External Database", expanded=False): # External Database Configuration (moved to top)    
            # Initialize session state for external database
            if "use_ext_database" not in st.session_state:
                st.session_state.use_ext_database = True
            if "selected_database" not in st.session_state:
                st.session_state.selected_database = ""
            if "k_results" not in st.session_state:
                st.session_state.k_results = 3

            # Enable external database checkbox
            st.session_state.use_ext_database = st.checkbox(
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
                    selected_db = st.selectbox(
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
                        st.info(f"Selected Database: {selected_db}")
                        st.success(f"Updated embedding model to: {embedding_model_name}")
                    
                    # Number of results to retrieve
                    st.session_state.k_results = st.slider(
                        "Number of results to retrieve", 
                        min_value=1, 
                        max_value=10, 
                        value=st.session_state.k_results
                    )
                    
                    selected_database = st.session_state.selected_database
                    k_results = st.session_state.k_results
                else:
                    st.warning("No databases found. Please upload documents first.")
                    st.session_state.use_ext_database = False
                    selected_database = None
                    k_results = 3
            else:
                selected_database = None
                k_results = 3
        
        
        # Model Selection (moved to bottom)
        with st.expander("🤖 LLM Model Selection", expanded=False):
            # Report writing LLM
            st.session_state.report_llm = st.selectbox(
                "Report Writing LLM",
                options=report_llm_models,
                index=report_llm_models.index(st.session_state.report_llm) if st.session_state.report_llm in report_llm_models else 0,
                help="Choose the LLM model to use for final report generation; loaded from global report_llms.md configuration"
            )
            
            # Summarization LLM
            st.session_state.summarization_llm = st.selectbox(
                "Summarization LLM",
                options=summarization_llm_models,
                index=summarization_llm_models.index(st.session_state.summarization_llm) if st.session_state.summarization_llm in summarization_llm_models else (summarization_llm_models.index("qwen3:latest") if "qwen3:latest" in summarization_llm_models else 0),
                help="Choose the LLM model to use for document summarization; loaded from global summarization_llms.md configuration"
            )
        
        use_ext_database = st.session_state.use_ext_database
   
        # Research Configuration
        with st.expander("🔬 Advanced Research Settings", expanded=False):
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
                "Number of Research Queries",
                min_value=1,
                max_value=10,
                value=3,
                help="Number of research queries to generate"
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
                value=True,
                help="Enable LLM-based quality assessment of the final report"
            )

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
    
       
    # Three-Phase Workflow Visualization Expander (moved here to be visible from the beginning)
    with st.expander("🔄 Show Three-Phase Workflow Graphs", expanded=False):
        st.markdown("### RAG Deep Researcher v2.0 - Three-Phase Workflow")
        
        # Display embedding model information
        try:
            if st.session_state.get('use_ext_database', False) and st.session_state.get('selected_database', ''):
                # Use the selected external database
                embedding_model_name = extract_embedding_model(st.session_state.selected_database)
                st.info(f"📊 **Current Embedding Model:** `{embedding_model_name}` (from selected database: `{st.session_state.selected_database}`)") 
            else:
                # No external database selected, show default
                st.info(f"📊 **Current Embedding Model:** Default (no external database selected)")
        except Exception as e:
            st.warning(f"Could not determine embedding model: {str(e)}")
        
        # Create three columns for the three phase graphs (side-by-side)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🤝 Phase 1: HITL Workflow")
            try:
                # Generate HITL graph visualization
                hitl_graph = create_hitl_graph()
                hitl_png = hitl_graph.get_graph(xray=True).draw_mermaid_png()
                st.image(hitl_png, caption="HITL Workflow Graph", use_container_width=True)
            except Exception as e:
                st.error(f"Could not generate HITL graph: {str(e)}")
        
        with col2:
            st.markdown("#### 📚 Phase 2: Retrieve & Summry")
            try:
                # Generate retrieval-summarization graph visualization
                retrieval_graph = create_retrieval_summarization_graph()
                retrieval_png = retrieval_graph.get_graph(xray=True).draw_mermaid_png()
                st.image(retrieval_png, caption="Retrieval & Summarize Graph", use_container_width=True)
            except Exception as e:
                st.error(f"Could not generate retrieval-summarization graph: {str(e)}")
        
        with col3:
            st.markdown("#### 📄 Phase 3: QA & Reporting")
            try:
                # Generate reporting graph visualization
                reporting_graph = create_rerank_reporter_graph()
                reporting_png = reporting_graph.get_graph(xray=True).draw_mermaid_png()
                st.image(reporting_png, caption="QA & Reporting Graph", use_container_width=True)
            except Exception as e:
                st.error(f"Could not generate reporting graph: {str(e)}")
    
    # Three-Phase Tabs
    tab1, tab2, tab3 = st.tabs(["🤝 Phase 1: HITL", "📚 Phase 2: Retrieval-Summarization", "📄 Phase 3: Reporting"])
    
    # Phase 1: HITL
    with tab1:
        # Dynamic phase info for HITL tab
        st.warning("🤝 **Current Phase: Human-in-the-Loop** - Interactive conversation to refine your research needs.")
        
        if st.session_state.workflow_phase == "hitl":
            
            # Initialize HITL session state variables if they don't exist
            if "hitl_conversation_history" not in st.session_state:
                st.session_state.hitl_conversation_history = []
            
            if "hitl_state" not in st.session_state:
                st.session_state.hitl_state = None
            
            if "waiting_for_human_input" not in st.session_state:
                st.session_state.waiting_for_human_input = False
            
            if "conversation_ended" not in st.session_state:
                st.session_state.conversation_ended = False
            
            # Initial query input - only show if no conversation has started and not processing
            if (len(st.session_state.hitl_conversation_history) == 0 and 
                not st.session_state.processing_initial_query):  
                #add a few blank lines          
                st.markdown("\n\n\n\n\n")          
                user_query = st.chat_input(
                    "Enter your initial research query"
                )
                
                if user_query:
                    # Set processing flag to hide input on next render
                    st.session_state.processing_initial_query = True
                    
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
            
            #keep this for debugging if needed later
            # Display debug information about the current state
            def debug_hitl_state():
                if st.session_state.hitl_state:
                    with st.expander("Debug: Current HITL State", expanded=False):
                        # Create a deep copy of the state to display
                        display_state = {}
                        for key, value in st.session_state.hitl_state.items():
                            display_state[key] = value
                        return st.json(display_state)
            
            # Handle human feedback - only show when waiting for input and not processing
            if (st.session_state.waiting_for_human_input and 
                not st.session_state.conversation_ended and
                not st.session_state.processing_feedback and
                len(st.session_state.hitl_conversation_history) > 0 and 
                st.session_state.hitl_conversation_history[-1]["role"] == "assistant"):
                
                # Use a dynamic key that changes after each submission to force widget reset
                human_feedback = st.chat_input(
                    "Your response (type '/end' to finish and proceed to main research)",
                    key=f"human_feedback_input_{st.session_state.input_counter}"
                )
                
                if human_feedback:
                    # Set processing flag to hide input on next render
                    st.session_state.processing_feedback = True
                    
                    # Check if the user wants to end the conversation
                    if human_feedback.strip().lower() == "/end":
                        st.session_state.conversation_ended = True
                        
                        # Add user message to conversation history
                        st.session_state.hitl_conversation_history.append({
                            "role": "user",
                            "content": "/end - Conversation ended"
                        })
                        
                        # Set flags
                        st.session_state.waiting_for_human_input = False
                        
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
                        
                        # Move to retrieval-summarization workflow phase
                        st.session_state.workflow_phase = "retrieval_summarization"
                        
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
                        
                        # Reset processing flag and continue waiting for input
                        st.session_state.processing_feedback = False
                        st.session_state.waiting_for_human_input = True
                        
                        # Increment input counter to reset widgets
                        st.session_state.input_counter += 1
                        st.rerun()
        else:
            # HITL phase completed - show summary
            st.success("✅ HITL Phase completed successfully!")
            # Display the main results
            research_queries = st.session_state.hitl_result.get("research_queries", [])
            lenqueries = len(research_queries)
            st.markdown(f"**Original Query:** {st.session_state.hitl_result.get("user_query", "N/A")}")
            st.markdown(f"**Generated {lenqueries} Additional Research Queries**")
            st.markdown(f"**Additional Context generated by HITL** ")
        
            if st.session_state.hitl_result:
                with st.expander("📋 HITL Phase Results (Completed)", expanded=False):
                    st.markdown("**Research Queries Generated:**")
                    for i, query in enumerate(research_queries, 1):
                        st.markdown(f"**{i}.** {query}")
                    
                    st.markdown("**Original Query:**")
                    st.markdown(st.session_state.hitl_result.get("user_query", "N/A"))
                    
                    st.markdown("**Additional Context:**")
                    st.markdown(st.session_state.hitl_result.get("additional_context", "N/A"))
    
    # Phase 2: Retrieval-Summarization
    with tab2:
        # Dynamic phase info for Retrieval-Summarization tab
        st.warning("📚 **Current Phase: Retrieval & Summarization** - Retrieving documents and generating summaries.")
        
        if st.session_state.workflow_phase == "retrieval_summarization":
            st.markdown("### 📚 Retrieval & Summarization Phase")
            st.markdown("The system will now retrieve relevant documents and generate summaries based on your HITL input.")
            
            if not st.session_state.hitl_result:
                st.warning("⚠️ Please complete the HITL phase first.")
            else:
                # Show HITL context
                with st.expander("🔗 Using HITL Results", expanded=False):
                    research_queries = st.session_state.hitl_result.get("research_queries", [])
                    st.markdown(f"**Research Queries ({len(research_queries)}):**")
                    for i, query in enumerate(research_queries, 1):
                        st.markdown(f"**{i}.** {query}")
                
                # Execute button
                if st.button("🔍 Start Retrieval & Summarization", type="primary"):
                    result = execute_retrieval_summarization_phase(
                        use_ext_database=use_ext_database,
                        selected_database=selected_database,
                        k_results=k_results
                    )
                    if result:
                        st.session_state.workflow_phase = "reporting"
                        st.rerun()
        
        elif st.session_state.workflow_phase in ["reporting", "completed"] and st.session_state.retrieval_summarization_result:
            # Show completed retrieval-summarization results
            st.success("✅ Retrieval & Summarization Phase completed successfully!")

            # Display the main results
            # Show retrieved documents summary
            result = st.session_state.retrieval_summarization_result
            retrieved_docs = result.get("retrieved_documents", {})
            total_docs = sum(len(docs) for docs in retrieved_docs.values())
            st.markdown(f"**Total Documents Retrieved:** {total_docs}")
            
            # Show summaries summary
            search_summaries = result.get("search_summaries", {})
            total_summaries = sum(len(summaries) for summaries in search_summaries.values())
            st.markdown(f"**Total Summaries Generated:** {total_summaries}")
            


            with st.expander("📚 Retrieval & Summarization Results (Completed)", expanded=False):              
                # Show detailed results for each research query
                for i, (query, docs) in enumerate(retrieved_docs.items(), 1):
                    st.markdown(f"### 🔍 Query {i}: {query[:100]}{'...' if len(query) > 100 else ''}")
                    
                    # Retrieved documents for this query
                    with st.expander(f"📄 Retrieved Documents ({len(docs)} documents)", expanded=False):
                        if docs:
                            for j, doc in enumerate(docs):
                                st.markdown(f"**Document {j+1}:**")
                                # Show content preview
                                content_preview = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                                st.text(content_preview)
                                
                                # Show metadata if available
                                if hasattr(doc, 'metadata') and doc.metadata:
                                    with st.expander(f"📋 Document {j+1} Metadata", expanded=False):
                                        st.json(doc.metadata)
                                st.divider()
                        else:
                            st.warning("No documents retrieved for this query.")
                    
                    # Summary for this query
                    query_summaries = search_summaries.get(query, [])
                    if query_summaries:
                        with st.expander(f"📝 Generated Summary", expanded=True):
                            for summary_doc in query_summaries:
                                st.markdown("**Summary Content:**")
                                st.markdown(summary_doc.page_content)
                                
                                # Show summary metadata
                                if hasattr(summary_doc, 'metadata') and summary_doc.metadata:
                                    with st.expander("📊 Summary Metadata", expanded=False):
                                        metadata_display = {
                                            "LLM Model": summary_doc.metadata.get("llm_model", "N/A"),
                                            "Language": summary_doc.metadata.get("language", "N/A"),
                                            "Document Count": summary_doc.metadata.get("document_count", "N/A"),
                                            "Source Documents": summary_doc.metadata.get("source_documents", []),
                                            "Source Paths": summary_doc.metadata.get("source_paths", [])
                                        }
                                        st.json(metadata_display)
                                
                                # Copy button for summary
                                if st.button(f"📋 Copy Summary {i}", key=f"copy_summary_{i}"):
                                    try:
                                        import pyperclip
                                        pyperclip.copy(summary_doc.page_content)
                                        st.success(f"Summary {i} copied to clipboard!")
                                    except ImportError:
                                        st.warning("pyperclip not available. Please install it to enable copying.")
                    else:
                        st.warning("No summary generated for this query.")
                    
                    st.divider()
        
        else:
            st.info("📝 Complete the HITL phase first to proceed with retrieval and summarization.")
    
    # Phase 3: Reporting
    with tab3:
        # Dynamic phase info for Reporting tab
        st.warning("📄 **Current Phase: Reporting** - Reranking summaries and generating final report.")
        
        if st.session_state.workflow_phase == "reporting":
            st.markdown("### 📄 Reporting Phase")
            st.markdown("The system will now rerank the summaries and generate a comprehensive final report.")
            
            if not st.session_state.retrieval_summarization_result:
                st.warning("⚠️ Please complete the Retrieval & Summarization phase first.")
            else:
                # Show retrieval-summarization context
                with st.expander("🔗 Using Retrieval & Summarization Results", expanded=False):
                    result = st.session_state.retrieval_summarization_result
                    search_summaries = result.get("search_summaries", {})
                    total_summaries = sum(len(summaries) for summaries in search_summaries.values())
                    st.markdown(f"**Total Summaries to Process:** {total_summaries}")
                
                # Execute button
                if st.button("📊 Start Reporting Phase", type="primary"):
                    result = execute_reporting_phase(enable_web_search=enable_web_search)
                    if result:
                        st.success("✅ All phases completed successfully!")
                        st.rerun()
        
        elif st.session_state.reporting_result:
            # Show completed reporting results
            st.success("✅ Reporting Phase completed successfully!")
            
            result = st.session_state.reporting_result
            
            # Processing details section
            with st.expander("🔍 Processing Details", expanded=False):
                st.write(f"**Current Position:** {result.get('current_position', 'unknown')}")
                st.write(f"**Research Queries Processed:** {len(result.get('research_queries', []))}")
                st.write(f"**Reflection Count:** {result.get('reflection_count', 0)}")
                
                # Show reranked summaries statistics
                reranked_summaries = result.get("all_reranked_summaries", [])
                if reranked_summaries:
                    st.subheader("📊 Reranked Documents")
                    
                    # Summary statistics
                    total_docs = len(reranked_summaries)
                    avg_score = sum(item.get('score', 0) for item in reranked_summaries) / total_docs if total_docs > 0 else 0
                    max_score = max([item.get('score', 0) for item in reranked_summaries], default=0)
                    
                    col_stats1, col_stats2, col_stats3 = st.columns(3)
                    with col_stats1:
                        st.metric("Total Documents", total_docs)
                    with col_stats2:
                        st.metric("Average Score", f"{avg_score:.2f}/10")
                    with col_stats3:
                        st.metric("Highest Score", f"{max_score:.2f}/10")
                    
                    # Display reranked documents in ranked order
                    for rank, item in enumerate(reranked_summaries, 1):
                        score = item.get('score', 0)
                        summary_data = item.get('summary', {})
                        
                        # Handle different summary formats
                        if isinstance(summary_data, dict):
                            content = summary_data.get('Content', str(summary_data))
                            source = summary_data.get('Source', 'Unknown')
                        else:
                            # Handle Document objects or string content
                            if hasattr(summary_data, 'page_content'):
                                content = summary_data.page_content
                                source = summary_data.metadata.get('source', 'Unknown') if hasattr(summary_data, 'metadata') else 'Unknown'
                            else:
                                content = str(summary_data)
                                source = 'Unknown'
                        
                        with st.expander(f"🥇 Rank #{rank} (Score: {score:.2f}/10)", expanded=rank <= 3):
                            st.markdown(f"**Score:** {score:.2f}/10")
                            st.markdown(f"**Query:** {item.get('query', 'N/A')}")
                            st.markdown(f"**Source:** {source}")
                            st.markdown(f"**Original Index:** {item.get('original_index', 'N/A')}")
                            
                            st.markdown("**Content:**")
                            st.markdown(content)
                            
                            # Add separator except for last item
                            if rank < len(reranked_summaries):
                                st.divider()
                else:
                    st.warning("No reranked summaries found. Check input data.")
            
            # Internet search results section
            internet_result = result.get("internet_result")
            internet_search_term = result.get("internet_search_term")
            if internet_result and internet_result.strip():
                st.subheader("🌐 Internet Search Results")
                
                # Check if web search was enabled
                web_search_enabled = result.get("web_search_enabled", False)
                if web_search_enabled:
                    st.success("✅ Web search was enabled and executed successfully")
                else:
                    st.info("ℹ️ Web search was not enabled for this query")
                
                # Display the generated search term if available
                if internet_search_term:
                    st.info(f"🔍 **Generated Search Term:** `{internet_search_term}`")
                
                # Display the internet search results
                with st.expander("📄 Internet Search Summary", expanded=False):
                    st.markdown(internet_result)
            elif result.get("web_search_enabled", False):
                st.subheader("🌐 Internet Search Results")
                st.warning("⚠️ Web search was enabled but no results were obtained. Check your Tavily API key and internet connection.")
            
            # Quality assessment section
            quality_check = result.get("quality_check", {})
            if quality_check and quality_check.get("enabled", False):
                st.subheader("🔍 Quality Assessment")
                
                # Get values with backward compatibility
                overall_score = quality_check.get("quality_score", quality_check.get("overall_score", 0))
                max_score = quality_check.get("max_score", 400)
                passes_quality = quality_check.get("is_accurate", quality_check.get("passes_quality", False))
                needs_improvement = quality_check.get("improvement_needed", quality_check.get("needs_improvement", False))
                improvement_suggestions = quality_check.get("improvement_suggestions", "")
                
                # New JSON structure fields
                issues_found = quality_check.get("issues_found", [])
                missing_elements = quality_check.get("missing_elements", [])
                citation_issues = quality_check.get("citation_issues", [])
                
                # Display quality metrics
                col_q1, col_q2, col_q3 = st.columns(3)
                with col_q1:
                    st.metric("Quality Score", f"{overall_score}/{max_score}")
                with col_q2:
                    status_color = "🟢" if passes_quality else "🔴"
                    st.metric("Assessment", f"{status_color} {'PASS' if passes_quality else 'FAIL'}")
                with col_q3:
                    if needs_improvement:
                        improvement_status = "🔄 Needs Improvement"
                    else:
                        improvement_status = "✅ Quality Passed"
                    st.metric("Status", improvement_status)
                
                # Display detailed quality analysis if available
                if issues_found or missing_elements or citation_issues:
                    st.markdown("#### 📊 Detailed Quality Analysis")
                    
                    col_detail1, col_detail2, col_detail3 = st.columns(3)
                    
                    with col_detail1:
                        if issues_found:
                            st.markdown("**🚨 Issues Found:**")
                            # Handle both string and list formats
                            if isinstance(issues_found, str):
                                st.markdown(f"• {issues_found}")
                            elif isinstance(issues_found, list):
                                for issue in issues_found:
                                    st.markdown(f"• {issue}")
                            else:
                                st.markdown(f"• {str(issues_found)}")
                        else:
                            st.markdown("**✅ No Issues Found**")
                    
                    with col_detail2:
                        if missing_elements:
                            st.markdown("**❓ Missing Elements:**")
                            # Handle both string and list formats
                            if isinstance(missing_elements, str):
                                st.markdown(f"• {missing_elements}")
                            elif isinstance(missing_elements, list):
                                for element in missing_elements:
                                    st.markdown(f"• {element}")
                            else:
                                st.markdown(f"• {str(missing_elements)}")
                        else:
                            st.markdown("**✅ All Elements Present**")
                    
                    with col_detail3:
                        if citation_issues:
                            st.markdown("**📚 Citation Issues:**")
                            # Handle both string and list formats
                            if isinstance(citation_issues, str):
                                st.markdown(f"• {citation_issues}")
                            elif isinstance(citation_issues, list):
                                for citation in citation_issues:
                                    st.markdown(f"• {citation}")
                            else:
                                st.markdown(f"• {str(citation_issues)}")
                        else:
                            st.markdown("**✅ Citations OK**")
                
                # Show improvement suggestions if they were generated
                if improvement_suggestions:
                    with st.expander("📝 Improvement Suggestions", expanded=False):
                        st.markdown(improvement_suggestions)
                
                # Show full assessment if available
                full_assessment = quality_check.get("full_assessment", "")
                if full_assessment:
                    with st.expander("🔍 Full Quality Assessment", expanded=False):
                        st.text(full_assessment)
            
            # Final report section
            final_answer = result.get("final_answer", "")
            with st.expander("📋 Final Report", expanded=False):
                if final_answer and final_answer.strip():
                    # Show improvement notice if quality checker triggered reflection loop
                    if quality_check and quality_check.get("needs_improvement", False):
                        st.info("ℹ️ This report has been regenerated based on quality assessment feedback through reflection loop.")
                    
                    # Get thinking process from structured output
                    thinking_process = result.get("thinking_process", "")
                    
                    # Show thinking process in a collapsed expander if available
                    if thinking_process and thinking_process.strip():
                        with st.expander("🧠 LLM Thinking Process", expanded=False):
                            st.text(thinking_process.strip())
                    
                    # Display the final answer (already clean from structured output)
                    if final_answer:
                        st.markdown(final_answer)
                    else:
                        st.warning("The answer appears to be empty. Please check the LLM response.")
                    
                    # Copy to clipboard and download buttons
                    col_btn1, col_btn2 = st.columns(2)
                    with col_btn1:
                        if st.button("📋 Copy Report to Clipboard"):
                            if copy_to_clipboard(final_answer):
                                st.success("Report copied to clipboard!")
                            else:
                                st.error("Could not copy to clipboard.")
                    
                    with col_btn2:
                        from datetime import datetime
                        st.download_button(
                            label="📥 Download Report",
                            data=final_answer,
                            file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                            mime="text/markdown"
                        )
                else:
                    st.warning("No report generated. This could be due to:")
                    st.markdown("- No reranked summaries found")
                    st.markdown("- All summaries scored below relevance threshold")
                    st.markdown("- Report generation failed")
        
        else:
            st.info("📚 Complete the Retrieval & Summarization phase first to proceed with reporting.")
    
    # ========================================
    # FINAL ANSWER DISPLAY (MAIN WINDOW)
    # ========================================
    
    # Check if we have a completed reporting phase with final answer
    if ((st.session_state.workflow_phase in ["reporting", "completed"]) and 
        hasattr(st.session_state, 'reporting_result') and 
        st.session_state.reporting_result) or (
        hasattr(st.session_state, 'reporting_result') and 
        st.session_state.reporting_result and 
        st.session_state.reporting_result.get("final_answer")):
        
        result = st.session_state.reporting_result
        final_answer = result.get("final_answer", "")
        
        if final_answer and final_answer.strip():
            # Add some spacing
            st.markdown("---")
            
            # Main final answer section - prominently displayed
            st.markdown("# 🎯 Final Research Report")
            
            # Show improvement notice if quality checker triggered reflection loop
            quality_check = result.get("quality_check", {})
            if quality_check and quality_check.get("needs_improvement", False):
                st.info("ℹ️ This report has been regenerated based on quality assessment feedback through reflection loop.")
            
            # Get thinking process from structured output
            thinking_process = result.get("thinking_process", "")
            
            # Show thinking process in a collapsed expander if available
            if thinking_process and thinking_process.strip():
                with st.expander("🧠 LLM Thinking Process", expanded=False):
                    st.text(thinking_process.strip())
            
            # Display the final answer prominently using chat message
            with st.chat_message("assistant"):
                if final_answer:
                    st.markdown(final_answer)
                else:
                    st.warning("The answer appears to be empty. Please check the LLM response.")
            
            # Action buttons in columns
            col_btn1, col_btn2, col_btn3 = st.columns(3)
            
            with col_btn1:
                if st.button("📋 Copy Report to Clipboard", key="main_copy_btn"):
                    if copy_to_clipboard(final_answer):
                        st.success("Report copied to clipboard!")
                    else:
                        st.error("Could not copy to clipboard.")
            
            with col_btn2:
                from datetime import datetime
                st.download_button(
                    label="📥 Download Report",
                    data=final_answer,
                    file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    key="main_download_btn"
                )
            
            with col_btn3:
                if st.button("🔄 Start New Research", key="main_new_research_btn"):
                    clear_chat()
                    st.rerun()
            
            # Show thinking process in a collapsed expander if found
            if think_matches:
                with st.expander("🧠 LLM Thinking Process", expanded=False):
                    for i, think_content in enumerate(think_matches, 1):
                        if len(think_matches) > 1:
                            st.markdown(f"**Thinking Block {i}:**")
                        st.text(think_content.strip())
                        if i < len(think_matches):
                            st.divider()
            
            # Show quality assessment if available
            if quality_check:
                with st.expander("📊 Quality Assessment Details", expanded=False):
                    if "overall_score" in quality_check:
                        score = quality_check["overall_score"]
                        max_score = 400  # Based on the 4-dimensional scoring system
                        score_percentage = (score / max_score) * 100
                        
                        # Score display with color coding
                        if score >= 300:
                            st.success(f"✅ Quality Score: {score}/{max_score} ({score_percentage:.1f}%) - PASSED")
                        else:
                            st.warning(f"⚠️ Quality Score: {score}/{max_score} ({score_percentage:.1f}%) - NEEDS IMPROVEMENT")
                    
                    # Show full assessment if available
                    full_assessment = quality_check.get("full_assessment", "")
                    if full_assessment:
                        st.text(full_assessment)

if __name__ == "__main__":
    main()
