import streamlit as st
import os
import sys
import torch
from typing import TypedDict, Dict, List, Any, Annotated, Optional, Literal, Union
from typing_extensions import Literal
from dotenv import load_dotenv
import json
from datetime import datetime
import gc

#uv run streamlit run dev/test_st-router.py --server.port 8501 --server.headless False --server.fileWatcherType none

# Add parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# CUDA memory management function
def clear_cuda_memory():
    """
    Clear CUDA memory cache to free up GPU resources between queries.
    Only has an effect if CUDA is available.
    """
    if torch.cuda.is_available():
        # Empty the cache
        torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
        print("CUDA memory cache cleared")
    return

# Available LLM models - predefine working models
AVAILABLE_MODELS = [
    "mistral-small3.2:latest", 
    "qwen3:1.7b", 
    "qwen3:latest", 
    "llama3.2", 
    "deepseek-r1:7b"
]

# Default model to use
DEFAULT_LLM_MODEL = "mistral-small3.2:latest"

# Import LangGraph components
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

# Import LangChain components
from langchain_core.tools import tool
from langchain_ollama.chat_models import ChatOllama
from pydantic import BaseModel, Field

# Import Tavily for web search
from tavily import TavilyClient

# Import project utilities
from src.utils_v1_1 import invoke_ollama, parse_output

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="LangGraph Query Router Demo",
    page_icon="🔀",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .category-box {
        border: 2px solid #63b463;
        border-radius: 10px;
        padding: 15px;
        background-color: #2e5c2e;
        color: white;
        margin: 10px 0;
    }
    .search-results {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 10px;
        background-color: #f9f9f9;
        margin: 5px 0;
    }
    .info-box {
        background-color: #2e5c2e;
        border-left: 5px solid #63b463;
        color: white;
        border-radius: 5px;
        padding: 10px 15px;
        margin: 10px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2);
    }
    .memory-box {
        background-color: #2e5c2e;
        color: white;
        border-radius: 5px;
        padding: 10px 15px;
        margin: 10px 0;
        border: 1px solid #63b463;
    }
    .answer-box {
        background-color: #2e5c2e;
        border-left: 5px solid #63b463;
        border-radius: 5px;
        padding: 15px;
        margin: 15px 0;
        color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Define Pydantic schemas for structured output
class QueryCategoryResponse(BaseModel):
    """Structured response for query categorization."""
    category: str = Field(description="The category that best fits the user's query")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0", ge=0.0, le=1.0)
    reasoning: str = Field(description="Brief explanation for why this category was chosen")

class SupervisorDecisionResponse(BaseModel):
    """Structured response for supervisor routing decisions."""
    action: str = Field(description="The next action to take: search, answer, or end")
    reasoning: str = Field(description="Explanation for why this action was chosen")

# Define tools using LangChain @tool decorator
@tool
def categorize_query_tool(query: str) -> str:
    """Categorize a user query using sophisticated LLM-based analysis.
    
    Args:
        query: The user query to categorize
        
    Returns:
        A JSON string with category, confidence, and reasoning
    """
    try:
        # Clear GPU memory before starting
        clear_cuda_memory()
        
        # Use the sophisticated categorization prompt and LLM analysis
        response = invoke_ollama(
            model=DEFAULT_LLM_MODEL,
            system_prompt=CATEGORY_CLASSIFICATION_PROMPT,
            user_prompt=f"Please categorize this query: {query}"
        )
        
        # Extract the category from the response (should be in quotes)
        import re
        category_match = re.search(r'"([^"]+)"', response)
        
        if category_match:
            category = category_match.group(1)
            
            # Determine confidence based on category characteristics
            confidence = 0.9  # High confidence for LLM-based categorization
            
            # Generate reasoning based on the category
            if category == "recent news":
                reasoning = "Query requires real-time or time-sensitive information"
            elif category == "general purpose":
                reasoning = "Query can be answered with static knowledge"
            elif category == "practical guide":
                reasoning = "Query asks for step-by-step instructions or how-to guidance"
            elif category == "legal guide":
                reasoning = "Query involves legal matters or regulatory information"
            elif category == "casual conversation":
                reasoning = "Query appears to be conversational or social interaction"
            else:
                reasoning = "Category determined through LLM analysis"
                
            return f'{{"category": "{category}", "confidence": {confidence}, "reasoning": "{reasoning}"}}'
        else:
            # Fallback to general purpose if no clear category found
            return '{"category": "general purpose", "confidence": 0.6, "reasoning": "Could not determine specific category, defaulting to general purpose"}'
            
    except Exception as e:
        # Fallback to simple heuristic if LLM fails
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["news", "recent", "latest", "current", "today", "breaking", "weather", "forecast"]):
            return '{"category": "recent news", "confidence": 0.7, "reasoning": "Fallback: Query contains time-sensitive keywords"}'
        elif any(word in query_lower for word in ["how to", "tutorial", "guide", "step", "instructions"]):
            return '{"category": "practical guide", "confidence": 0.8, "reasoning": "Fallback: Query asks for instructions"}'
        elif any(word in query_lower for word in ["legal", "law", "regulation", "court", "attorney", "lawyer"]):
            return '{"category": "legal guide", "confidence": 0.7, "reasoning": "Fallback: Query contains legal terms"}'
        elif any(word in query_lower for word in ["hello", "hi", "how are you", "good morning", "thanks"]):
            return '{"category": "casual conversation", "confidence": 0.9, "reasoning": "Fallback: Query appears conversational"}'
        else:
            return '{"category": "general purpose", "confidence": 0.6, "reasoning": "Fallback: Default categorization due to error"}'
    
    finally:
        # Clear GPU memory after completion
        clear_cuda_memory()

@tool
def supervisor_decision_tool(query: str, category: str, search_attempts: int, max_attempts: int, has_search_results: bool) -> str:
    """Make a supervisor decision about the next workflow action.
    
    Args:
        query: The user query
        category: The categorized query type
        search_attempts: Number of search attempts made
        max_attempts: Maximum allowed search attempts
        has_search_results: Whether search results are available
        
    Returns:
        A JSON string with action and reasoning
    """
    if category in ["general purpose", "casual conversation"]:
        return '{"action": "answer", "reasoning": "General purpose and casual queries can be answered directly without search"}'
    elif search_attempts >= max_attempts:
        return '{"action": "answer", "reasoning": "Maximum search attempts reached, proceeding to answer generation"}'
    elif category in ["recent news", "practical guide", "legal guide"] and not has_search_results:
        return '{"action": "search", "reasoning": "This category requires current information, performing web search"}'
    elif has_search_results:
        return '{"action": "answer", "reasoning": "Search results available, proceeding to generate final answer"}'
    else:
        return '{"action": "search", "reasoning": "Need to gather more information via web search"}'

# Define Pydantic models for state management
class QueryCategory(BaseModel):
    """Query categorization result."""
    category: str
    confidence: float
    reasoning: str

class SupervisorDecision(BaseModel):
    """Supervisor decision result."""
    action: str
    reasoning: str

# Define the state for our router
class RouterState(TypedDict):
    user_query: str
    category: Optional[str]
    category_confidence: Optional[float]
    category_reasoning: Optional[str]
    search_results: Optional[str]
    final_answer: Optional[str]
    thinking_process: Optional[str]
    needs_search: bool
    search_attempts: int
    max_search_attempts: int
    messages: List[Any]  # For tool calling

# Define category classification schema for structured output
CATEGORY_CLASSIFICATION_PROMPT = """
You are a thorogh query categorization expert. 
**GOAL:**
- Determine the category of the user query.

**Categories for Reference:**
- "recent news": Requires real-time data (e.g., forecasts, live scores, breaking news).
- "general purpose": Static, factual knowledge.
- "practical guide": How-to instructions.
- "legal guide": Legal questions.
- "casual conversation": Greetings, small talk.

**Tasks (step by step):**
- Analyse the users query carefully. Your primary goal is to determine if a query requires up-to-date information, e.g. from a web research.
- Based on your analysis, determine the category of the query.
- Respond with ONLY the category name in quotes.

**Examples:**

1.  **Query**: "what is the weather forecast for Paris?" leads to "recent news"

2.  **Query**: "latest news on the stock market" leads to "recent news"

3.  **Query**: "what is the boiling point of water?" leads to "general purpose"

4.  **Query**: "how do I change a tire?" leads to "practical guide"
"""

SUPERVISOR_DECISION_PROMPT = """
You are a supervisor deciding the next action based on the query category and current context.

Query: {query}
Category: {category}
Search Attempts: {search_attempts}
Max Attempts: {max_attempts}
Previous Search Results: {search_results}

Decision Rules:
1. If category is "general purpose" or "casual conversation" → go directly to final answer
2. If category is "recent news", "practical guide", or "legal guide" → web search first, then answer
3. If search results exist, evaluate if they are sufficient for a good answer
4. If search attempts >= max attempts → go to final answer regardless

Respond with ONLY one of these actions:
- "search" - perform web search
- "answer" - generate final answer
- "end" - end the workflow
"""

# Human prompt template for answer generation
ANSWER_GENERATION_PROMPT = """
I need to answer this user query: "{query}"

Query Category: {category}

Here are the search results I should incorporate:
{search_results}

Please provide a comprehensive and accurate response SOLELY based on the users initial query, the query category and search results.
"""

@tool
def tavily_search_tool(query: str, include_raw_content: bool = True, max_results: int = 3) -> str:
    """Search the web using the Tavily API for recent information.

    Args:
        query: The search query to execute
        include_raw_content: Whether to include the raw_content from Tavily in the formatted string
        max_results: Maximum number of results to return (default: 3)

    Returns:
        Formatted string containing search results with titles, URLs, and content
    """
    try:
        # Get current date for context
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Enhance search query with current date context
        enhanced_query = f"{query} (current date: {current_date})"
        
        tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        response = tavily_client.search(
            enhanced_query,
            max_results=max_results,
            include_raw_content=include_raw_content
        )
        
        if response and "results" in response and response["results"]:
            formatted_results = []
            for result in response["results"]:
                formatted_result = f"""**Title:** {result.get('title', 'N/A')}
**URL:** {result.get('url', 'N/A')}
**Content:** {result.get('content', 'N/A')}"""
                if result.get('raw_content'):
                    formatted_result += f"\n**Raw Content:** {result['raw_content'][:500]}..."
                formatted_results.append(formatted_result)
            
            return "\n---\n".join(formatted_results)
        else:
            return "No search results found."
            
    except Exception as e:
        return f"Error in Tavily search: {str(e)}"

def categorize_query(state: RouterState) -> RouterState:
    """Categorize the user query using LangChain-Ollama with structured output."""
    try:
        # Clear GPU memory before starting
        clear_cuda_memory()
        
        # Get the selected model or use default
        current_model = state.get("llm_model", DEFAULT_LLM_MODEL)
        
        # Improved system prompt for more precise tool usage
        system_prompt = CATEGORY_CLASSIFICATION_PROMPT
        
        # Initialize Ollama model with tool calling and structured output
        ollama_model = ChatOllama(model=current_model, temperature=0.1)
        model_with_tools = ollama_model.bind_tools([categorize_query_tool])
        structured_model = model_with_tools.with_structured_output(QueryCategoryResponse)
        
        # Prepare messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Please categorize this query: {state['user_query']}"}
        ]
        
        # Get structured response
        response: QueryCategoryResponse = structured_model.invoke(messages)
        
        state["category"] = response.category
        state["category_confidence"] = response.confidence
        state["category_reasoning"] = response.reasoning
        
        st.info(f"🏷️ Query categorized as: **{response.category}** (confidence: {response.confidence:.2f})")
        with st.expander("🤔 Categorization Reasoning", expanded=False):
            st.write(response.reasoning)
        
        return state
        
    except Exception as e:
        st.error(f"Error in categorization: {str(e)}")
        # Fallback: use the tool directly
        try:
            import json
            result_str = categorize_query_tool.invoke({"query": state["user_query"]})
            result = json.loads(result_str)
            
            state["category"] = result["category"]
            state["category_confidence"] = result["confidence"]
            state["category_reasoning"] = result["reasoning"]
        except:
            state["category"] = "general purpose"
            state["category_confidence"] = 0.5
            state["category_reasoning"] = "Fallback due to error"
        return state

def supervisor_decision(state: RouterState) -> str:
    """Supervisor node that decides the next action using LangChain-Ollama with structured output."""
    try:
        # Clear GPU memory before starting
        clear_cuda_memory()
        
        # Get the selected model or use default
        current_model = state.get("llm_model", DEFAULT_LLM_MODEL)
        
        # Direct decision making based on rules (avoiding LLM confusion with tool calling)
        # This provides reliable routing without tool calling confusion
        
        # Rule 1: General purpose or casual conversation queries go directly to answer
        if state["category"] in ["general purpose", "casual conversation"]:
            action = "answer"
            reasoning = f"Category '{state['category']}' can be answered directly without search."
        # Rule 2: If maximum search attempts reached, go to answer
        elif state["search_attempts"] >= state["max_search_attempts"]:
            action = "answer"
            reasoning = f"Maximum search attempts ({state['max_search_attempts']}) reached. Using available information to answer."
        # Rule 3: Information-seeking categories with no search results yet need search
        elif state["category"] in ["recent news", "practical guide", "legal guide"] and not state.get("search_results"):
            action = "search"
            reasoning = f"'{state['category']}' query requires current information. Performing search."
        # Rule 4: If we have search results, go to answer
        elif state.get("search_results"):
            action = "answer"
            reasoning = "Search results available. Generating answer based on retrieved information."
        # Default: perform search
        else:
            action = "search"
            reasoning = "Need to gather information via search before answering."
            
        # Display decision in UI
        st.info(f"🏹 Supervisor decision: **{action}**")
        with st.expander("🧠 Decision Reasoning", expanded=False):
            st.write(reasoning)
            
        # Make sure we return a valid routing value
        return action
    
    except Exception as e:
        st.error(f"Error in supervisor decision: {str(e)}")
        # Default logic based on category
        if state["category"] in ["general purpose", "casual conversation"]:
            return "answer"
        elif state["search_attempts"] >= state["max_search_attempts"]:
            return "answer"
        else:
            return "search"

def perform_search(state: RouterState) -> RouterState:
    """Perform web search using LangChain tool calling."""
    try:
        # Clear GPU memory before starting
        clear_cuda_memory()
        
        st.info(f"🔍 Performing web search (attempt {state['search_attempts'] + 1})...")
        
        # Display search query for transparency
        st.info(f"📅 Search query: {state['user_query']}")
        
        # Use the tool directly - with enhanced query based on category
        query_enhancement = ""
        if state["category"] == "recent news":
            query_enhancement = "recent information"
        elif state["category"] == "practical guide":
            query_enhancement = "step-by-step guide"
        elif state["category"] == "legal guide":
            query_enhancement = "legal information"
            
        enhanced_query = state["user_query"]
        if query_enhancement:
            enhanced_query += f" (Looking for {query_enhancement})"
        
        search_results = tavily_search_tool.invoke({
            "query": enhanced_query,
            "include_raw_content": True,
            "max_results": 3
        })
        
        state["search_results"] = search_results
        state["search_attempts"] += 1
        
        # Display search results in UI
        with st.expander(f"🔍 Search Results (Attempt {state['search_attempts']})", expanded=True):
            st.markdown(search_results)
        
        # Clear GPU memory after search
        clear_cuda_memory()
        
        return state
        
    except Exception as e:
        st.error(f"Error in search: {str(e)}")
        state["search_results"] = f"Search error: {str(e)}"
        state["search_attempts"] += 1
        return state

def generate_final_answer(state: RouterState) -> RouterState:
    """Generate the final answer based on query and available information."""
    try:
        # Clear GPU memory before starting
        clear_cuda_memory()
        
        st.info("💭 Generating answer...")
        
        # Get the selected model or use default
        current_model = state.get("llm_model", DEFAULT_LLM_MODEL)
        
        # Enhanced system prompt with clear instructions
        system_prompt = """
You are an expert researcher with deep analytical skills. Your task is to give a final answer to a user query under the following instructions:

1. ANALYZE the user query and provided search results thoroughly
2. EXTRACT key relevant information, i.e. the query category and search results that addresses the query
3. SYNTHESIZE a detailled and concise answer

Your response MUST be structured in two clearly separated parts:

<think>
In this section, provide your detailed analysis process:
- Break down the user's query to understand core information needs
- Evaluate which search results are most relevant and why
- Identify key facts, data points, or insights from the search results
- Note any gaps in information or potential biases
- Explain your reasoning for including specific information in the final answer
This should be thoughtful analysis, not just a restatement of the query or search results.
</think>

After your thinking section, provide your actual answer to the user:
- Make it direct, clear, and tailored to the query category
- Include specific information from search results with proper attribution when available
- Be helpful, informative, accurate and concise
- Format with appropriate paragraphs, bullet points, or headings if needed
- For "recent news" queries, include dates and sources
- For "practical guide" queries, include clear steps or instructions
- For "legal guide" queries, note limitations of your advice when appropriate

The quality of your answer will be judged on both the thoroughness of your analysis and the clarity of your final response.
"""

        # Format the human prompt with specific query information
        user_prompt = ANSWER_GENERATION_PROMPT.format(
            query=state["user_query"],
            category=state["category"],
            search_results=state.get("search_results", "No search results available.")
        )
        
        response = invoke_ollama(
            model=current_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )
        
        # Parse the response to separate thinking and answer
        parsed_output = parse_output(response)
        thinking = parsed_output.get("reasoning")
        final_answer = parsed_output.get("response")
        
        # If the LLM didn't follow the <think></think> format, use the full response as the answer
        if not thinking and not final_answer:
            final_answer = response
        elif not final_answer:
            final_answer = response
            
        state["thinking_process"] = thinking
        state["final_answer"] = final_answer
        
        # Clear GPU memory after completion
        clear_cuda_memory()
        
        return state
        
    except Exception as e:
        st.error(f"Error in answer generation: {str(e)}")
        state["final_answer"] = f"Error generating answer: {str(e)}"
        state["thinking_process"] = None
        return state

def create_router_graph() -> StateGraph:
    """Create the LangGraph workflow for query routing."""
    
    # Create the graph
    workflow = StateGraph(RouterState)
    
    # Add nodes
    workflow.add_node("categorize", categorize_query)
    workflow.add_node("search", perform_search)
    workflow.add_node("answer", generate_final_answer)
    
    # Add edges
    workflow.add_edge(START, "categorize")
    
    # Add conditional edges from categorize to supervisor decision
    workflow.add_conditional_edges(
        "categorize",
        supervisor_decision,
        {
            "search": "search",
            "answer": "answer",
            "end": END
        }
    )
    
    # Add conditional edges from search back to supervisor
    workflow.add_conditional_edges(
        "search",
        supervisor_decision,
        {
            "search": "search",
            "answer": "answer", 
            "end": END
        }
    )
    
    # Answer always ends
    workflow.add_edge("answer", END)
    
    return workflow.compile(checkpointer=MemorySaver())

def main():
    """Main Streamlit application."""
    
    st.title("🔀 LangGraph Query Router Demo")
    st.markdown("This demo showcases a LangGraph-based query router that categorizes queries and routes them through different workflows.")
    
    # Initialize session state
    if "router_graph" not in st.session_state:
        st.session_state.router_graph = create_router_graph()
    
    # Display workflow visualization
    with st.expander("🔄 Workflow Visualization", expanded=False):
        try:
            graph_png = st.session_state.router_graph.get_graph().draw_mermaid_png()
            st.image(graph_png, caption="LangGraph Query Router Workflow", use_container_width=True)
        except Exception as e:
            st.error(f"Could not generate workflow visualization: {str(e)}")
    
    # User input
    st.subheader("📝 Enter Your Query")
    user_query = st.text_area(
        "What would you like to know?",
        placeholder="Enter your question here...",
        height=100
    )
    
    # Configuration
    st.subheader("⚙️ Configuration")
    col1, col2 = st.columns(2)
    with col1:
        max_search_attempts = st.slider("Max Search Attempts", 1, 5, 2)
        
        # Clear CUDA memory button
        if st.button("🔄 Clear CUDA Memory"):
            clear_cuda_memory()
            st.success("CUDA memory cleared successfully!")
            
    with col2:
        # Model selection dropdown
        selected_model = st.selectbox(
            "🤖 Select LLM Model",
            options=AVAILABLE_MODELS,
            index=AVAILABLE_MODELS.index(DEFAULT_LLM_MODEL) if DEFAULT_LLM_MODEL in AVAILABLE_MODELS else 0,
            help="Select the LLM model to use for query processing"
        )
        
        st.info(f"🤖 Using Model: **{selected_model}**")
        
        # Memory usage note with improved visibility
        st.markdown("""<div class="memory-box">
                    💾 <b>Memory Management:</b> This app automatically clears CUDA memory between operations.
                    </div>""", unsafe_allow_html=True)
    
    if st.button("🚀 Process Query", type="primary"):
        if user_query.strip():
            # Clear GPU memory before starting a new query
            clear_cuda_memory()
            
            # Initialize state - include the selected model
            initial_state = {
                "user_query": user_query.strip(),
                "category": None,
                "category_confidence": None,
                "category_reasoning": None,
                "search_results": None,
                "final_answer": None,
                "thinking_process": None,
                "needs_search": False,
                "search_attempts": 0,
                "max_search_attempts": max_search_attempts,
                "messages": [],
                "llm_model": selected_model
            }
            
            # Create a unique thread ID for this conversation
            thread_id = f"query_{hash(user_query)}"
            config = {"configurable": {"thread_id": thread_id}}
            
            # Execute the workflow
            st.subheader("🔄 Workflow Execution")
            
            try:
                # Run the graph
                final_state = None
                for state in st.session_state.router_graph.stream(initial_state, config):
                    final_state = state
                
                # Display results
                if final_state:
                    # Get the last state from the stream
                    last_node, last_state = next(iter(final_state.items()))
                    
                    st.subheader("📊 Results")
                    
                    # Display category with improved visibility
                    if last_state.get("category"):
                        st.markdown(f"""
                        <div class="info-box">
                        <h4>🏷️ Query Category: {last_state['category'].title()}</h4>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Display the final answer in a dedicated section
                    if last_state.get("final_answer"):
                        st.markdown(f"""
                        <div class="answer-box">
                        <h3>💡 Final Answer</h3>
                        <div>{last_state.get("final_answer")}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Also store the answer in session state for persistence
                        st.session_state["last_answer"] = last_state.get("final_answer")
                    
                    # Display execution summary
                    with st.expander("📈 Execution Summary", expanded=False):
                        st.json({
                            "query": last_state.get("user_query"),
                            "category": last_state.get("category"),
                            "search_attempts": last_state.get("search_attempts", 0),
                            "had_search_results": bool(last_state.get("search_results")),
                            "final_node": last_node
                        })
                
            except Exception as e:
                st.error(f"Error executing workflow: {str(e)}")
                st.exception(e)
        else:
            st.warning("Please enter a query to process.")
    
    # Example queries
    st.subheader("💡 Example Queries")
    examples = [
        ("General Purpose", "What is machine learning?"),
        ("Recent News", "What are the latest developments in AI?"),
        ("Practical Guide", "How do I set up a Python virtual environment?"),
        ("Legal Guide", "What are the GDPR requirements for data processing?"),
        ("Casual Conversation", "Hello, how are you today?")
    ]
    
    cols = st.columns(len(examples))
    for i, (category, example) in enumerate(examples):
        with cols[i]:
            if st.button(f"{category}\n\n*{example[:30]}...*", key=f"example_{i}"):
                st.rerun()

if __name__ == "__main__":
    main()
