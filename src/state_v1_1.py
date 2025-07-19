import operator
from typing import Annotated, List, Dict, Any, Optional
from typing_extensions import TypedDict
# Updated import path for Document to fix ModuleNotFoundError
from langchain_core.documents import Document

class ResearcherState(TypedDict):
    user_query: str
    current_position: int
    detected_language: str  # Added field to store detected language
    research_queries: list[str]
    retrieved_documents: dict[str, list[Document]]  
    search_summaries: dict[str, list[Document]]
    final_answer: str
    quality_check: Optional[dict[str, Any]]  # Added field to store quality check results
    additional_context: Optional[list[Document]]  # Added field to store additional context from document retrieval
    # Persist user-selected LLM models throughout the graph workflow
    report_llm: str  # LLM model used for report writing
    summarization_llm: str  # LLM model used for document summarization
    # For handling duplicate research queries
    query_mapping: Optional[dict[str, str]]  # Maps indexed queries to original queries
    enable_quality_checker: bool = False  # Flag to enable/disable quality checker


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
