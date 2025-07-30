#!/usr/bin/env python3
"""
Retrieval-Summarization Graph

This module implements the retrieval and summarization phase as a LangGraph workflow.
Based on the logic from dev/basic_retr-summ_app.py but implemented as a proper graph
using ResearcherStateV2 for state management.
"""

import os
import sys
from typing import Dict, List, Any
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langgraph.graph import StateGraph, END
from langgraph.types import RunnableConfig
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.state_v2_0 import ResearcherStateV2
from src.rag_helpers_v1_1 import get_tenant_vectorstore, source_summarizer_ollama
from src.utils_v1_1 import format_documents_with_metadata

# Import summarization prompts
import importlib.util
summ_prompts_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dev', 'summ_prompts.py')
spec = importlib.util.spec_from_file_location("summ_prompts", summ_prompts_path)
summ_prompts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summ_prompts)
SUMMARIZER_SYSTEM_PROMPT = summ_prompts.SUMMARIZER_SYSTEM_PROMPT
SUMMARIZER_HUMAN_PROMPT = summ_prompts.SUMMARIZER_HUMAN_PROMPT


def extract_embedding_model(db_dir_name: str) -> str:
    """
    Extract the embedding model name from the database directory name.
    Handles various database naming conventions.
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


def retrieve_documents_node(state: ResearcherStateV2, config: RunnableConfig) -> ResearcherStateV2:
    """
    LangGraph node that retrieves documents for all research queries.
    
    Args:
        state: ResearcherStateV2 containing research queries and configuration
        config: RunnableConfig for the graph
    
    Returns:
        Updated ResearcherStateV2 with retrieved_documents populated
    """
    print(f"\n[RETRIEVAL] Starting document retrieval for {len(state['research_queries'])} queries...")
    
    # Get configuration from state or use defaults
    selected_database = config.get("configurable", {}).get("selected_database", "default")
    k_results = config.get("configurable", {}).get("k_results", 3)
    
    # Set up database path
    DATABASE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "kb", "database")
    db_path = os.path.join(DATABASE_PATH, selected_database)
    
    try:
        # Extract embedding model from selected database name
        embedding_model = extract_embedding_model(selected_database)
        print(f"[RETRIEVAL] Using embedding model: {embedding_model}")
        
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
        
        retrieved_documents = {}
        
        # Retrieve documents for each research query
        for i, query in enumerate(state['research_queries']):
            print(f"[RETRIEVAL] Processing query {i+1}/{len(state['research_queries'])}: {query[:100]}...")
            
            try:
                # Perform similarity search
                docs = vector_db.similarity_search(query, k=k_results)
                retrieved_documents[query] = docs
                print(f"[RETRIEVAL] Retrieved {len(docs)} documents for query {i+1}")
                
            except Exception as e:
                print(f"[RETRIEVAL] Error retrieving documents for query {i+1}: {e}")
                retrieved_documents[query] = []
        
        print(f"[RETRIEVAL] Document retrieval completed. Total queries processed: {len(retrieved_documents)}")
        
        return {
            **state,
            "retrieved_documents": retrieved_documents,
            "current_position": state.get("current_position", 0) + 1
        }
        
    except Exception as e:
        print(f"[RETRIEVAL] Critical error during document retrieval: {e}")
        return {
            **state,
            "retrieved_documents": {},
            "current_position": state.get("current_position", 0) + 1
        }


def summarize_documents_node(state: ResearcherStateV2, config: RunnableConfig) -> ResearcherStateV2:
    """
    LangGraph node that summarizes retrieved documents for all research queries.
    
    Args:
        state: ResearcherStateV2 containing retrieved documents and configuration
        config: RunnableConfig for the graph
    
    Returns:
        Updated ResearcherStateV2 with search_summaries populated
    """
    print(f"\n[SUMMARIZATION] Starting document summarization...")
    
    retrieved_documents = state.get("retrieved_documents", {})
    if not retrieved_documents:
        print("[SUMMARIZATION] No retrieved documents found. Skipping summarization.")
        return {
            **state,
            "search_summaries": {},
            "current_position": state.get("current_position", 0) + 1
        }
    
    # Get summarization configuration
    summarization_llm = state.get("summarization_llm", "qwen3:latest")
    user_query = state.get("user_query", "")
    human_feedback = state.get("human_feedback", "")
    detected_language = state.get("detected_language", "English")
    
    # Handle detected_language if it's a dict (from memories, this was a known issue)
    if isinstance(detected_language, dict):
        detected_language = detected_language.get("detected_language", "English")
    
    print(f"[SUMMARIZATION] Using LLM: {summarization_llm}")
    print(f"[SUMMARIZATION] Language: {detected_language}")
    
    search_summaries = {}
    
    # Process each query and its retrieved documents
    for i, (query, docs) in enumerate(retrieved_documents.items()):
        print(f"[SUMMARIZATION] Processing query {i+1}/{len(retrieved_documents)}: {query[:100]}...")
        
        if not docs:
            print(f"[SUMMARIZATION] No documents for query {i+1}. Skipping.")
            search_summaries[query] = []
            continue
        
        try:
            # Format documents for summarization
            formatted_docs = format_documents_with_metadata(docs)
            
            # Call the summarization function with correct parameters
            from src.prompts_v1_1 import SUMMARIZER_SYSTEM_PROMPT
            summary_result = source_summarizer_ollama(
                user_query=query,  # Use the specific research query as user_query
                context_documents=formatted_docs,
                language=detected_language,
                system_message=SUMMARIZER_SYSTEM_PROMPT,
                llm_model=summarization_llm,
                human_feedback=human_feedback
            )
            
            # Extract content string from the returned dictionary
            if isinstance(summary_result, dict) and 'content' in summary_result:
                summary_content = summary_result['content']
                # Extract metadata if available
                summary_metadata = summary_result.get('metadata', {})
            else:
                # Fallback: treat as string if not a dict
                summary_content = str(summary_result)
                summary_metadata = {}
            
            # Create summary document with extracted content
            summary_doc = Document(
                page_content=summary_content,
                metadata={
                    "query": query,
                    "source": "summarization",
                    "llm_model": summarization_llm,
                    "language": detected_language,
                    "document_count": len(docs),
                    # Include original document metadata if available
                    "source_documents": summary_metadata.get('name', []),
                    "source_paths": summary_metadata.get('path', [])
                }
            )
            
            search_summaries[query] = [summary_doc]
            print(f"[SUMMARIZATION] Completed summary for query {i+1} (length: {len(summary_content)} chars)")
            
        except Exception as e:
            print(f"[SUMMARIZATION] Error summarizing query {i+1}: {e}")
            search_summaries[query] = []
    
    print(f"[SUMMARIZATION] Summarization completed. Total summaries: {len(search_summaries)}")
    
    return {
        **state,
        "search_summaries": search_summaries,
        "current_position": state.get("current_position", 0) + 1
    }


def create_retrieval_summarization_graph() -> StateGraph:
    """
    Create the LangGraph workflow for document retrieval and summarization.
    
    Returns:
        Compiled LangGraph workflow
    """
    # Create the state graph
    workflow = StateGraph(ResearcherStateV2)
    
    # Add nodes
    workflow.add_node("retrieve_documents", retrieve_documents_node)
    workflow.add_node("summarize_documents", summarize_documents_node)
    
    # Define the workflow
    workflow.set_entry_point("retrieve_documents")
    workflow.add_edge("retrieve_documents", "summarize_documents")
    workflow.add_edge("summarize_documents", END)
    
    # Compile the graph
    return workflow.compile()


# Export the main function for use in other modules
__all__ = ["create_retrieval_summarization_graph"]
