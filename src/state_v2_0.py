import operator
from typing import Annotated, List, Dict, Any, Optional
from typing_extensions import TypedDict
# Updated import path for Document to fix ModuleNotFoundError
from langchain_core.documents import Document

class ResearcherStateV2(TypedDict):
    """Extended ResearcherState with Human-in-the-Loop (HITL) capabilities."""
    
    # Original ResearcherState fields
    user_query: str
    current_position: int
    detected_language: str  # Added field to store detected language
    research_queries: list[str]
    retrieved_documents: dict[str, list[Document]]  
    search_summaries: dict[str, list[Document]]
    final_answer: str
    quality_check: Optional[dict[str, Any]]  # Added field to store quality check results
    # Persist user-selected LLM models throughout the graph workflow
    report_llm: str  # LLM model used for report writing
    summarization_llm: str  # LLM model used for document summarization
    enable_quality_checker: bool = False  # Flag to enable/disable quality checker
    
    # Reranking fields for basic_rerank-reporter_app.py
    all_reranked_summaries: Optional[list[dict[str, Any]]]  # Reranked summaries with scores
    reflection_count: int = 0  # Counter for quality checker reflection loops
    internet_result: Optional[str] = None  # Internet search results from Tavily
    internet_search_term: Optional[str] = None  # Generated search term used for Tavily search
    web_search_enabled: bool = False  # Flag to enable/disable web search
    
    # New HITL fields (extending from InitState in basic_HITL_app.py)
    human_feedback: Optional[str]  # Latest human feedback message
    analysis: Optional[str]  # Analysis of the conversation, including the latest human feedback
    follow_up_questions: Optional[str]  # Follow-up questions generated based on last human feedback
    additional_context: Optional[str]  # Additional context from human feedback (overrides the list version above)


# Keep the original InitState for compatibility with basic_HITL_app.py
class InitState(TypedDict):
    """State for the human feedback loop briefing phase."""
    user_query: str  # The initial user query
    current_position: int  # Current position in the workflow
    detected_language: str  # Detected language of the query
    additional_context: str  # Additional context from human feedback
    human_feedback: str  # Latest human feedback message
    analysis: str  # Analysis of the conversation, including the latest human feedback
    follow_up_questions: str  # Follow-up questions generated based on last human feedback
    report_llm: str  # LLM model used for report writing
    summarization_llm: str  # LLM model used for document summarization
    research_queries: list[str] # List of research queries for the handover to the main graph
