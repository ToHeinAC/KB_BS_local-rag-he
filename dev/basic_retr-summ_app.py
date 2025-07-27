#!/usr/bin/env python3
"""
Basic Retrieval-Summarization App

This app focuses on testing the retrieval-summarization step from app_v2_0.py
with different LLM models for summarization evaluation.

Features:
1. User query input and vector DB selection (same as app_v2_0.py)
2. Reads summarization prompts from dev/summ_prompts.py
3. Loops over all LLMs from summarization_llms.md for evaluation
4. Shows results per LLM in expandable sections
"""

import streamlit as st
import os
import sys
import time
from pathlib import Path
from datetime import datetime

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import necessary modules from the main app
from src.state_v2_0 import ResearcherStateV2
from src.graph_v2_0 import main_graph
from src.rag_helpers_v1_1 import get_summarization_llm_models, source_summarizer_ollama
from src.configuration_v1_1 import get_config_instance
from src.vector_db_v1_1 import get_or_create_vector_db, search_documents
from src.utils_v1_1 import format_documents_with_metadata, clear_cuda_memory

# Import summarization prompts from dev/summ_prompts.py
import importlib.util
summ_prompts_path = os.path.join(os.path.dirname(__file__), 'summ_prompts.py')
spec = importlib.util.spec_from_file_location("summ_prompts", summ_prompts_path)
summ_prompts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summ_prompts)
SUMMARIZER_SYSTEM_PROMPT = summ_prompts.SUMMARIZER_SYSTEM_PROMPT
SUMMARIZER_HUMAN_PROMPT = summ_prompts.SUMMARIZER_HUMAN_PROMPT

clear_cuda_memory()

def extract_embedding_model(db_dir_name):
    """
    Extract the embedding model name from the database directory name.
    Handles various database naming conventions including:
    - Standard format: "organization/model_name"
    - Directory format with separators: "Qwen--Qwen3-Embedding-0.6B--3000--600"
    - Directory format with path: "Qwen/Qwen--Qwen3-Embedding-0.6B--3000--600"
    """
    # Handle the specific case of database names with '--' separators
    if '--' in db_dir_name:
        parts = db_dir_name.split('--')
        
        if len(parts) >= 2:
            # For format like "Qwen--Qwen3-Embedding-0.6B--3000--600"
            if '/' not in parts[0]:
                return f"{parts[0]}/{parts[1]}"
            
            # For format like "Qwen/Qwen--Qwen3-Embedding-0.6B--3000--600"
            else:
                org = parts[0].split('/')[0]  # Extract "Qwen" from "Qwen/Qwen"
                return f"{org}/{parts[1]}"
    
    # Handle the case where the name already has a proper format like "Qwen/Qwen3-Embedding-0.6B"
    if '/' in db_dir_name and '--' not in db_dir_name:
        return db_dir_name
    
    # Fallback: replace double hyphens with slashes
    model_name = db_dir_name.replace("--", "/")
    
    return model_name.split('/')[0] + '/' + model_name.split('/')[1]

def retrieve_documents_for_query(query, selected_database, k_results=3):
    """
    Retrieve documents for a single query using the same logic as app_v2_0.py
    """
    try:
        # Set up database path
        DATABASE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "kb", "database")
        db_path = os.path.join(DATABASE_PATH, selected_database)
        
        # Import the specific function for this database path
        from src.rag_helpers_v1_1 import get_tenant_vectorstore
        from langchain_huggingface import HuggingFaceEmbeddings
        
        # Extract embedding model from selected database name
        embedding_model = extract_embedding_model(selected_database)
        
        # Create embeddings instance
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model, model_kwargs={'device': 'cpu'})
        
        # Get vector database for this specific path
        vector_db = get_tenant_vectorstore(
            tenant_id="default",
            embed_llm=embeddings,
            persist_directory=db_path,
            similarity="cosine",
            normal=True
        )
        
        # Search documents using similarity_search_for_tenant directly
        from src.rag_helpers_v1_1 import similarity_search_for_tenant
        
        docs = vector_db.similarity_search(query, k=k_results)
        
        return docs
    except Exception as e:
        st.error(f"Error retrieving documents: {str(e)}")
        return []

def test_summarization_with_llm(query, docs, llm_model, language="English", human_feedback="", formatted_docs=None):
    """
    Test summarization with a specific LLM model using the exact same logic as app_v2_0.py
    """
    clear_cuda_memory()
    try:
        # Format documents if not already formatted
        if formatted_docs is None:
            formatted_docs = format_documents_with_metadata(docs)
        
        # Transform the docs into the format expected by source_summarizer_ollama
        transformed_docs = []
        for doc in docs:
            transformed_docs.append({
                'content': doc.page_content,
                'metadata': doc.metadata
            })
        
        # Use the source_summarizer_ollama function with the prompts from summ-prompts.py
        summary_result = source_summarizer_ollama(
            user_query=query,
            context_documents=transformed_docs,  # Pass the transformed documents with metadata
            llm_model=llm_model,
            human_feedback=human_feedback,
            language=language,  # Use the selected language
            system_message=SUMMARIZER_SYSTEM_PROMPT  # Add the system_message parameter
        )
        
        return summary_result
    except Exception as e:
        return f"Error with {llm_model}: {str(e)}"

def main():
    st.set_page_config(
        page_title="Basic Retrieval-Summarization Tester",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Basic Retrieval-Summarization Tester")
    st.markdown("Test retrieval and summarization with different LLM models")
    
    # Initialize session state
    if "test_results" not in st.session_state:
        st.session_state.test_results = {}
    
    # Configuration Section
    st.header("⚙️ Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # User query input
        user_query = st.text_area(
            "Enter your query:",
            placeholder="What would you like to research?",
            height=100
        )
    
    with col2:
        # Vector database selection (same as app_v2_0.py)
        DATABASE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "kb", "database")
        database_dir = Path(DATABASE_PATH)
        database_options = [d.name for d in database_dir.iterdir() if d.is_dir()] if database_dir.exists() else []
        
        if database_options:
            selected_database = st.selectbox(
                "Select Vector Database:",
                options=database_options,
                help="Choose a database for document retrieval"
            )
            
            # Show embedding model info
            embedding_model_name = extract_embedding_model(selected_database)
            st.info(f"**Embedding Model:** {embedding_model_name}")
            
            # Number of results to retrieve
            k_results = st.slider(
                "Number of documents to retrieve:",
                min_value=1,
                max_value=10,
                value=3
            )
        else:
            st.error("No databases found. Please create a database first.")
            return
    
    # Language selection
    languages = ["English", "German", "Spanish", "French", "Italian", "Chinese", "Japanese", "Russian"]
    selected_language = st.selectbox(
        "Select language for summarization:",
        options=languages,
        index=0,  # Default to English
        help="Language to use for the summarization response"
    )
    
    # Human feedback (optional)
    human_feedback = st.text_area(
        "Human feedback (optional):",
        placeholder="Additional context or feedback for summarization...",
        height=80
    )
    
    # Test button
    if st.button("🚀 Run Retrieval-Summarization Test", type="primary"):
        if not user_query.strip():
            st.error("Please enter a query.")
            return
        
        if not database_options:
            st.error("No databases available.")
            return
        
        # Clear previous results
        st.session_state.test_results = {}
        
        # Step 1: Retrieve documents
        st.subheader("📄 Document Retrieval")
        with st.spinner("Retrieving documents..."):
            docs = retrieve_documents_for_query(user_query, selected_database, k_results)
            
            # Save the formatted documents for display in the prompts expander
            if docs:
                formatted_docs = format_documents_with_metadata(docs)
                st.session_state.formatted_docs = formatted_docs
                st.session_state.last_run_data = {
                    "user_query": user_query,
                    "human_feedback": human_feedback,
                    "language": selected_language
                }
        
        if not docs:
            st.error("No documents retrieved.")
            return
        
        st.success(f"Retrieved {len(docs)} documents")
        
        # Format documents for display and use in all tests
        formatted_docs = format_documents_with_metadata(docs)
        
        # Show retrieved documents in an expander
        with st.expander(f"📋 Retrieved Documents ({len(docs)} documents)", expanded=False):
            for i, doc in enumerate(docs):
                st.markdown(f"**Document {i+1}:**")
                st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                if hasattr(doc, 'metadata') and doc.metadata:
                    st.json(doc.metadata)
                st.divider()
        
        # Display the prompts that will be used for all LLMs
        st.subheader("🔤 Prompts Used For All LLMs")
        with st.expander("📋 View System & Human Prompts", expanded=True):
            # Prepare the prompts with actual values for display
            system_prompt_filled = SUMMARIZER_SYSTEM_PROMPT.format(language=selected_language)
            human_prompt_filled = SUMMARIZER_HUMAN_PROMPT.format(
                user_query=user_query,
                human_feedback=human_feedback,
                documents=formatted_docs,
                language=selected_language
            )
            
            # Display system prompt
            st.markdown("### System Prompt")
            st.code(system_prompt_filled, language="markdown")
            
            # Display human prompt
            st.markdown("### Human Prompt")
            human_prompt_short = human_prompt_filled[:1000] + "..." if len(human_prompt_filled) > 1000 else human_prompt_filled
            st.code(human_prompt_short, language="markdown")
            
            # Add a note about the prompts
            st.info("These prompts will be used for all LLM models in the test. The only difference between runs is the LLM model used.")
        
        # Step 2: Test summarization with all LLMs
        st.subheader("📝 Summarization Results")
        
        # Get all summarization LLMs
        summarization_llms = get_summarization_llm_models()
        st.info(f"Testing with {len(summarization_llms)} LLM models")
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Test each LLM
        for i, llm_model in enumerate(summarization_llms):
            status_text.text(f"Testing {llm_model}...")
            
            # Run summarization
            start_time = time.time()
            summary_result = test_summarization_with_llm(
                user_query, 
                docs, 
                llm_model,
                selected_language,
                human_feedback,
                formatted_docs  # Pass the pre-formatted documents
            )
            end_time = time.time()
            
            # Store results
            st.session_state.test_results[llm_model] = {
                'summary_data': summary_result,
                'processing_time': end_time - start_time,
                'timestamp': datetime.now().strftime("%H:%M:%S")
            }
            
            # Update progress
            progress_bar.progress((i + 1) / len(summarization_llms))
        
        status_text.text("✅ All tests completed!")
        progress_bar.progress(1.0)
    
    # Display results
    if st.session_state.test_results:
        # We don't need the global prompts expander anymore as we'll show them per model
        # The prompts are now included in each model's results
        
        st.subheader("📊 Summarization Results by LLM")
        
        # Create expandable sections for each LLM result
        for llm_model, result in st.session_state.test_results.items():
            with st.expander(f"🤖 {llm_model} - {result['processing_time']:.2f}s", expanded=False):
                # Summary result
                st.markdown("### 📊 Summary Result")
                
                # Display the raw JSON in a collapsible section for debugging
                with st.expander("View Raw JSON", expanded=False):
                    st.json(result['summary_data'])
                
                # Display processing information
                st.markdown(f"**Processing Time:** {result['processing_time']:.2f} seconds")
                st.markdown(f"**Completed at:** {result['timestamp']}")
                
                st.markdown("### Summary Content")
                # Display the formatted content once, not the raw JSON again
                if isinstance(result['summary_data'], str) and result['summary_data'].startswith("Error"):
                    st.error(result['summary_data'])
                else:
                    try:
                        # If it's a dictionary with 'content' key
                        if isinstance(result['summary_data'], dict) and 'content' in result['summary_data']:
                            st.markdown(result['summary_data']['content'])
                        # If it has a content attribute (object)
                        elif hasattr(result['summary_data'], 'content'):
                            st.markdown(result['summary_data'].content)
                        elif hasattr(result['summary'], 'content'):
                            st.markdown(result['summary'].content)
                        # If it's a string
                        elif isinstance(result['summary'], str):
                            st.markdown(result['summary'])
                        # Fallback
                        else:
                            st.markdown(str(result['summary']))
                    except Exception as e:
                        st.error(f"Error displaying summary: {str(e)}")
                        st.markdown(str(result['summary']))
                
                # Copy button for each result
                if st.button(f"📋 Copy {llm_model} Result", key=f"copy_{llm_model}"):
                    try:
                        import pyperclip
                        pyperclip.copy(str(result['summary']))
                        st.success(f"Copied {llm_model} result to clipboard!")
                    except ImportError:
                        st.warning("pyperclip not available. Please install it to enable copying.")
    
    # Footer
    st.divider()
    st.markdown("**Note:** This app uses the exact same retrieval and summarization logic as `app_v2_0.py`")
    st.markdown("**Prompts:** Loaded from `dev/summ_prompts.py`")
    st.markdown("**LLM Models:** Loaded from `summarization_llms.md`")
    if "last_run_data" in st.session_state:
        st.markdown(f"**Language:** Currently set to {st.session_state.last_run_data['language']}")

if __name__ == "__main__":
    main()
