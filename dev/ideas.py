
from langchain_community.llms import Ollama
from langchain.tools import tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage


class ResearcherStateV2(TypedDict):
    """ResearcherState for tool calling agent"""
    user_query: str # The original user query
    query_category: dict # Category of the user query and if research is needed
    research_plan: str # Research plan for the research process
    research_queries: list[str] # List of research queries for the research process
    retrieved_documents: list[str] # List of retrieved information based ao the respective research queries
    search_summaries: list[str] # List of search summaries based ao the respective research queries and retrieved information
    draft_answer: str # Draft answer based ao the search_summaries
    gap_analysis: str # Gap analysis of the draft answer to the user query and the search_summaries
    additional_context: Optional[str]  # Additional context
    final_answer: str # Final answer based ao the draft answer and gap analysis

def categorizer_node(state: ResearcherStateV2) -> ResearcherStateV2:
    """Categorize the user query and determine if research is needed.
    Uses the following state variables:
    - user_query
    Updates the following state variables:
    - query_category
    """
    
def research_plan_node(state: ResearcherStateV2) -> ResearcherStateV2:
    """Generate an initial research plan based on the user query and the query category.
    Uses the following state variables:
    - user_query
    - query_category
    Updates the following state variables:
    - research_plan
    """

def research_supervisor_node(state: ResearcherStateV2) -> ResearcherStateV2:
    """Supervise the research process.
    Has access to the following tools:
    - research_query_writer
    - document_retriever
    - document_summarizer
    - draft_writer
    - gap_analyzer
    - research_plan_updater
    Uses the following state variables:
    - user_query
    - query_category
    - research_plan
    - research_queries
    - retrieved_documents
    - search_summaries
    - draft_answer
    - gap_analysis
    Updates the following state variables:
    - research_plan
    - research_queries
    - retrieved_documents
    - search_summaries
    - draft_answer
    - gap_analysis
    Rules:
    - start with research_query_writer
    - continue with document_retriever if research_queries are not empty
    - continue with document_summarizer if retrieved_documents are not empty
    - continue with draft_writer if search_summaries are not empty
    - continue with gap_analyzer only if a draft answer is not empty
    - continue with research_plan_updater only if a gap_analysis is not empty
    - route to final_answer_node if a gap_analysis hints that no relevant gap is left over
    """

def final_answer_node(state: ResearcherStateV2) -> ResearcherStateV2:
    """Generate the final answer based on the draft answer in case the gap_analyser hints that no relevant gap is left over
    Uses the following state variables:
    - user_query
    - query_category
    - draft_answer
    Updates the following state variables:
    - final_answer
    """

