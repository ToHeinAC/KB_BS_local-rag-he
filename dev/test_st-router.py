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
from langchain_community.llms import Ollama
from langchain.agents import Tool, initialize_agent
from langchain.schema import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

# Import Tavily for web search
from tavily import TavilyClient

# Import project utilities
from src.utils_v1_1 import parse_output

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

@tool
def supervisor_decision_tool(category: str, search_attempts: int = 0, max_search_attempts: int = 2, search_results: Optional[str] = None) -> str:
    """Make a supervisor decision about the next workflow action.
    
    Args:
        category: The category of the query
        search_attempts: Number of search attempts made so far
        max_search_attempts: Maximum number of search attempts allowed
        search_results: Previous search results if any
        
    Returns:
        A JSON string with action and reasoning
    """
    st.write("Used supervisor_decision_tool ...")
    
    # Use the provided values
    has_search_results = bool(search_results)
    
    # Apply decision logic
    if category in ["general purpose", "casual conversation"]:
        action = "answer"
        reasoning = "General purpose and casual queries can be answered directly without search"
    elif search_attempts >= max_search_attempts:
        action = "answer"
        reasoning = f"Maximum search attempts ({max_search_attempts}) reached, proceeding to answer generation"
    elif category in ["recent news", "practical guide", "legal guide"] and not has_search_results:
        action = "search"
        reasoning = "This category requires current information, performing web search"
    elif has_search_results:
        action = "answer"
        reasoning = "Search results available, proceeding to generate final answer"
    else:
        action = "search"
        reasoning = "Need to gather more information via web search"
    
    # Display decision in UI
    st.info(f"🏹 Supervisor decision: **{action}**")
    with st.expander("🧠 Decision Reasoning", expanded=False):
        st.write(reasoning)
    
    return f'{{"action": "{action}", "reasoning": "{reasoning}"}}'

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

Please help me create a comprehensive and accurate response.
"""

@tool
def tavily_search_tool(state: dict) -> str:
    """Search the web using the Tavily API for recent information.

    Args:
        state: The RouterState dictionary containing user_query and other state

    Returns:
        Formatted string containing search results with titles, URLs, and content
    """
    try:
        st.write("Used tavily_search_tool ...")
        
        # Extract query and other parameters from state
        query = state.get("user_query", "")
        category = state.get("category", "general purpose")
        search_attempts = state.get("search_attempts", 0)
        
        # Display search query for transparency
        st.info(f"📅 Search query: {query}")
        
        # Use the tool directly - with enhanced query based on category
        query_enhancement = ""
        if category == "recent news":
            query_enhancement = "recent information"
        elif category == "practical guide":
            query_enhancement = "step-by-step guide"
        elif category == "legal guide":
            query_enhancement = "legal information"
            
        enhanced_query = query
        if query_enhancement:
            enhanced_query += f" (Looking for {query_enhancement})"
        
        # Get current date for context
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Initialize Tavily client
        tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        
        # Perform the search
        search_response = tavily_client.search(
            query=enhanced_query,
            search_depth="advanced",
            max_results=3,
            include_raw_content=True
        )
        
        # Format the results
        if search_response and "results" in search_response:
            results = search_response["results"]
            formatted_results = []
            
            for i, result in enumerate(results, 1):
                title = result.get("title", "No title")
                url = result.get("url", "No URL")
                content = result.get("content", "No content available")
                
                formatted_result = f"**Result {i}: {title}**\n"
                formatted_result += f"URL: {url}\n"
                formatted_result += f"Content: {content}\n"
                
                # Include raw content if available and requested
                if "raw_content" in result:
                    raw_content = result["raw_content"]
                    if raw_content and len(raw_content) > 100:  # Only include if substantial
                        formatted_result += f"Raw Content: {raw_content[:500]}...\n"
                
                formatted_results.append(formatted_result)
            
            search_results = "\n---\n".join(formatted_results)
            
            # Display search results in UI
            with st.expander(f"🔍 Search Results (Attempt {search_attempts + 1})", expanded=True):
                st.markdown(search_results)
            
            return search_results
        else:
            return "No search results found."
            
    except Exception as e:
        error_msg = f"Error in Tavily search: {str(e)}"
        st.error(error_msg)
        return error_msg

# Helper function for categorization logic
def _categorize_query_logic(user_query: str, llm_model: str = DEFAULT_LLM_MODEL) -> dict:
    """Helper function that contains the actual categorization logic."""
    try:
        # Clear GPU memory before starting
        clear_cuda_memory()
        
        # Create the categorization prompt
        system_prompt = """
You are a query categorization expert. Analyze the user query and categorize it into one of these categories:

1. "general purpose" - General questions, explanations, how-to guides that don't require recent information
2. "recent news" - Questions about current events, recent developments, breaking news
3. "practical guide" - Step-by-step instructions, tutorials, practical how-to content
4. "legal guide" - Legal advice, regulations, compliance, legal procedures
5. "casual conversation" - Greetings, small talk, casual interactions

Provide your response in JSON format with:
- "category": one of the above categories
- "confidence": float between 0.0 and 1.0
- "reasoning": brief explanation of your decision
"""
        
        user_prompt = f"Categorize this query: {user_query}"
        
        # Get response from LLM using Ollama directly
        llm = Ollama(model=llm_model)
        full_prompt = f"{system_prompt}\n\nUser: {user_prompt}\nAssistant: I'll categorize this query and provide a JSON response."
        response = llm.invoke(full_prompt)
        
        # Parse JSON response - handle potential formatting issues
        try:
            # Try to find JSON in the response
            import re
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                result = json.loads(json_match.group())
            else:
                # If no JSON found, create a structured response
                raise ValueError("No JSON found in response")
        except (json.JSONDecodeError, ValueError):
            # Fallback: try to extract category from response text
            response_lower = response.lower()
            if "recent news" in response_lower or "news" in response_lower:
                category = "recent news"
            elif "practical guide" in response_lower or "guide" in response_lower or "how to" in response_lower:
                category = "practical guide"
            elif "legal" in response_lower:
                category = "legal guide"
            elif "casual" in response_lower or "conversation" in response_lower:
                category = "casual conversation"
            else:
                category = "general purpose"
            
            result = {
                "category": category,
                "confidence": 0.7,
                "reasoning": f"Extracted from response: {response[:100]}..."
            }
        
        # Validate required fields
        if "category" not in result or "confidence" not in result or "reasoning" not in result:
            raise ValueError("Missing required fields in response")
            
        return result
        
    except Exception as e:
        st.error(f"Error in categorization logic: {str(e)}")
        return {
            "category": "general purpose",
            "confidence": 0.5,
            "reasoning": f"Fallback due to error: {str(e)}"
        }

def categorize_query(state: RouterState) -> RouterState:
    """Categorize the user query using tool calling with LangChain agent."""
    try:
        # Clear GPU memory before starting
        clear_cuda_memory()
        
        # Get current model and query
        current_model = state.get("llm_model", DEFAULT_LLM_MODEL)
        query = state.get("user_query", "")
        
        # Create categorization tool that uses helper function
        def categorize_tool_func(query: str) -> str:
            """Categorize a query into predefined categories."""
            result = _categorize_query_logic(query, current_model)
            return json.dumps(result)
        
        categorize_tool = Tool(
            name="categorize_query",
            func=categorize_tool_func,
            description="Categorize a user query into predefined categories. ONLY use this tool when you need to categorize the user's query."
        )
        
        # Create LLM and agent using idiomatic pattern
        llm = Ollama(model=current_model)
        system_prompt = """You are a query categorization assistant. Analyze the user query and use the categorize_query tool to categorize it.

ONLY call the categorize_query tool when you need to categorize the user's query.

Available tools:
- categorize_query: Use this to categorize the user's query into predefined categories"""
        
        agent = initialize_agent(
            tools=[categorize_tool],
            llm=llm,
            agent_type="zero-shot-react-description",
            system_message=SystemMessage(content=system_prompt)
        )
        
        # Execute agent with tool calling
        response = agent.invoke(f"Please categorize this query: {query}")
        
        # Parse the result from agent output
        output = response.get('output', '') if isinstance(response, dict) else str(response)
        
        # Try to extract JSON from the output
        try:
            # Look for JSON in the output
            import re
            json_match = re.search(r'\{[^}]+\}', output)
            if json_match:
                result_json = json.loads(json_match.group())
            else:
                # Fallback parsing
                result_json = {"category": "general purpose", "confidence": 0.5, "reasoning": "Could not parse agent output"}
        except:
            result_json = {"category": "general purpose", "confidence": 0.5, "reasoning": "JSON parsing failed"}
        
        # Update state with results
        state["category"] = result_json.get("category", "general purpose")
        state["category_confidence"] = result_json.get("confidence", 0.5)
        state["category_reasoning"] = result_json.get("reasoning", "Categorization completed")
        
    except Exception as e:
        st.error(f"Error in categorization: {str(e)}")
        # Fallback
        state["category"] = "general purpose"
        state["category_confidence"] = 0.5
        state["category_reasoning"] = "Fallback due to error"
    
    # Display results
    st.info(f"🏷️ Query categorized as: **{state['category']}** (confidence: {state['category_confidence']:.2f})")
    with st.expander("🤔 Categorization Reasoning", expanded=False):
        st.write(state['category_reasoning'])
        
    return state

# Helper function for supervisor decision logic
def _supervisor_decision_logic(category: str, search_attempts: int = 0, max_search_attempts: int = 2, search_results: Optional[str] = None) -> dict:
    """Helper function that contains the actual supervisor decision logic."""
    try:
        # Clear GPU memory before starting
        clear_cuda_memory()
        
        # Create the decision prompt
        system_prompt = """
You are a workflow supervisor. Based on the query category and current state, decide the next action:

Actions:
- "search": Search for information (for recent news, guides that need current info)
- "answer": Generate answer directly (for general questions, when search limit reached, or when search results are sufficient)
- "end": End the workflow (only if explicitly requested)

Decision Rules:
1. "general purpose" or "casual conversation" → usually "answer"
2. "recent news", "practical guide", "legal guide" → "search" first, then "answer"
3. If search_attempts >= max_search_attempts → "answer"
4. If search_results exist and are sufficient → "answer"

Provide response in JSON format:
- "action": one of "search", "answer", "end"
- "reasoning": explanation of your decision
"""
        
        user_prompt = f"""
Category: {category}
Search attempts: {search_attempts}/{max_search_attempts}
Has search results: {bool(search_results)}
Search results preview: {search_results[:200] if search_results else "None"}

What should be the next action?
"""
        
        # Get response from LLM using Ollama directly
        llm = Ollama(model=DEFAULT_LLM_MODEL)
        full_prompt = f"{system_prompt}\n\nUser: {user_prompt}\nAssistant: I'll analyze the situation and provide a JSON response."
        response = llm.invoke(full_prompt)
        
        # Parse JSON response - handle potential formatting issues
        try:
            # Try to find JSON in the response
            import re
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                result = json.loads(json_match.group())
            else:
                # If no JSON found, create a structured response
                raise ValueError("No JSON found in response")
        except (json.JSONDecodeError, ValueError):
            # Fallback: analyze response text for decision
            response_lower = response.lower()
            if "search" in response_lower and "answer" not in response_lower:
                action = "search"
                reasoning = f"Extracted 'search' from response: {response[:100]}..."
            elif "end" in response_lower:
                action = "end"
                reasoning = f"Extracted 'end' from response: {response[:100]}..."
            else:
                action = "answer"
                reasoning = f"Defaulted to 'answer' from response: {response[:100]}..."
            
            result = {
                "action": action,
                "reasoning": reasoning
            }
        
        # Validate required fields
        if "action" not in result or "reasoning" not in result:
            raise ValueError("Missing required fields in response")
            
        return result
        
    except Exception as e:
        st.error(f"Error in supervisor decision logic: {str(e)}")
        # Fallback logic
        if category in ["general purpose", "casual conversation"]:
            return {"action": "answer", "reasoning": f"Fallback: Direct answer for {category}"}
        elif search_attempts >= max_search_attempts:
            return {"action": "answer", "reasoning": "Fallback: Max search attempts reached"}
        elif search_results:
            return {"action": "answer", "reasoning": "Fallback: Search results available"}
        else:
            return {"action": "search", "reasoning": "Fallback: Need to search first"}

def supervisor_decision(state: RouterState) -> str:
    """Supervisor node that decides the next action using tool calling with LangChain agent."""
    try:
        # Clear GPU memory before starting
        clear_cuda_memory()
        
        # Get state values
        category = state.get("category", "general purpose")
        search_attempts = state.get("search_attempts", 0)
        max_attempts = state.get("max_search_attempts", 2)
        search_results = state.get("search_results", None)
        current_model = state.get("llm_model", DEFAULT_LLM_MODEL)
        
        # Create supervisor decision tool that uses helper function
        def supervisor_tool_func(category: str, search_attempts: int = 0, max_attempts: int = 2, has_results: bool = False) -> str:
            """Make a supervisor decision about the next workflow action."""
            result = _supervisor_decision_logic(
                category=category,
                search_attempts=search_attempts,
                max_search_attempts=max_attempts,
                search_results=search_results if has_results else None
            )
            return json.dumps(result)
        
        decision_tool = Tool(
            name="make_decision",
            func=supervisor_tool_func,
            description="Make a supervisor decision about the next workflow action. ONLY use this tool when you need to decide the next action."
        )
        
        # Create LLM and agent using idiomatic pattern
        llm = Ollama(model=current_model)
        system_prompt = """You are a workflow supervisor. Analyze the current state and use the make_decision tool to decide the next action.

ONLY call the make_decision tool when you need to decide the next workflow action.

Available tools:
- make_decision: Use this to decide the next workflow action based on current state"""
        
        agent = initialize_agent(
            tools=[decision_tool],
            llm=llm,
            agent_type="zero-shot-react-description",
            system_message=SystemMessage(content=system_prompt)
        )
        
        # Execute agent with tool calling
        has_search_results = bool(search_results)
        response = agent.invoke(f"Category: {category}, Search attempts: {search_attempts}/{max_attempts}, Has results: {has_search_results}. What should be the next action?")
        
        # Parse the result from agent output
        output = response.get('output', '') if isinstance(response, dict) else str(response)
        
        # Try to extract JSON from the output
        try:
            # Look for JSON in the output
            import re
            json_match = re.search(r'\{[^}]+\}', output)
            if json_match:
                result_json = json.loads(json_match.group())
            else:
                # Fallback parsing
                result_json = {"action": "answer", "reasoning": "Could not parse agent output"}
        except:
            result_json = {"action": "answer", "reasoning": "JSON parsing failed"}
        
        action = result_json.get("action", "answer")
        reasoning = result_json.get("reasoning", "Decision made by agent")
        
        # Display decision in UI
        st.info(f"🏹 Supervisor decision: **{action}**")
        with st.expander("🧠 Decision Reasoning", expanded=False):
            st.write(reasoning)
            
        # Update state
        state["supervisor_reasoning"] = reasoning
        
        return action
        
    except Exception as e:
        st.error(f"Error in supervisor decision: {str(e)}")
        # Default to answering if there's an error
        return "answer"

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
        
        # Generate final answer using LLM with Ollama directly
        llm = Ollama(model=current_model)
        full_prompt = f"{system_prompt}\n\nUser: {user_prompt}\nAssistant:"
        response = llm.invoke(full_prompt)
        
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

def categorize_query_with_agent(state: RouterState) -> RouterState:
    """Categorize the user query using LangChain agent with tools."""
    try:
        # Clear GPU memory before starting
        clear_cuda_memory()
        
        # Get the selected model or use default
        current_model = state.get("llm_model", DEFAULT_LLM_MODEL)
        
        # Create LangChain agent with categorization tool
        from langchain_community.llms import Ollama
        from langchain.agents import Tool, initialize_agent
        from langchain.schema import SystemMessage
        
        llm = Ollama(model=current_model)
        
        # Create categorization tool
        def categorize_tool_func(query: str) -> str:
            """Categorize a query into predefined categories."""
            return categorize_query_tool.invoke({"user_query": query, "llm_model": current_model})
        
        categorize_tool = Tool(
            name="categorize_query",
            func=categorize_tool_func,
            description="Categorize a user query into one of these categories: recent news, general purpose, practical guide, legal guide, casual conversation"
        )
        
        system_prompt = """You are a query categorization expert. Use the categorize_query tool to analyze and categorize the user's query.
        
        ALWAYS use the categorize_query tool for every query to get the proper categorization with confidence and reasoning.
        
        After using the tool, extract the category, confidence, and reasoning from the response."""
        
        agent = initialize_agent(
            tools=[categorize_tool],
            llm=llm,
            agent_type="zero-shot-react-description",
            system_message=SystemMessage(content=system_prompt)
        )
        
        # Use the agent to categorize
        response = agent.invoke(f"Please categorize this query: {state['user_query']}")
        
        # Extract categorization from agent response
        import json
        import re
        
        # Try to extract JSON from the response
        json_match = re.search(r'\{[^}]+\}', str(response.get('output', '')))
        if json_match:
            try:
                result = json.loads(json_match.group())
                state["category"] = result.get("category", "general purpose")
                state["category_confidence"] = result.get("confidence", 0.7)
                state["category_reasoning"] = result.get("reasoning", "Categorized using agent")
            except:
                # Fallback
                state["category"] = "general purpose"
                state["category_confidence"] = 0.6
                state["category_reasoning"] = "Fallback categorization"
        else:
            # Fallback
            state["category"] = "general purpose"
            state["category_confidence"] = 0.6
            state["category_reasoning"] = "Fallback categorization"
        
        # Display results
        st.info(f"🏷️ Query categorized as: **{state['category']}** (confidence: {state['category_confidence']:.2f})")
        with st.expander("🤔 Categorization Reasoning", expanded=False):
            st.write(state['category_reasoning'])
        
        # Clear GPU memory after completion
        clear_cuda_memory()
        
        return state
    except Exception as e:
        st.error(f"Error in categorization: {str(e)}")
        # Fallback
        state["category"] = "general purpose"
        state["category_confidence"] = 0.5
        state["category_reasoning"] = "Fallback due to error"
        return state

def perform_search_with_agent(state: RouterState) -> RouterState:
    """Perform web search using LangChain agent with tools."""
    try:
        # Clear GPU memory before starting
        clear_cuda_memory()
        
        # Get the selected model or use default
        current_model = state.get("llm_model", DEFAULT_LLM_MODEL)
        
        # Create LangChain agent with search tool
        from langchain_community.llms import Ollama
        from langchain.agents import Tool, initialize_agent
        from langchain.schema import SystemMessage
        
        llm = Ollama(model=current_model)
        
        # Create search tool
        def search_tool_func(query: str) -> str:
            """Search the web for information."""
            return tavily_search_tool({
                "user_query": query,
                "category": state.get("category", "general purpose"),
                "search_attempts": state.get("search_attempts", 0)
            })
        
        search_tool = Tool(
            name="web_search",
            func=search_tool_func,
            description="Search the web for current information, news, guides, or other relevant content"
        )
        
        system_prompt = """You are a web search assistant. Use the web_search tool to find relevant information for the user's query.
        
        ALWAYS use the web_search tool to gather current information before providing any response.
        
        After searching, return the search results."""
        
        agent = initialize_agent(
            tools=[search_tool],
            llm=llm,
            agent_type="zero-shot-react-description",
            system_message=SystemMessage(content=system_prompt)
        )
        
        # Use the agent to search
        response = agent.invoke(f"Search for information about: {state['user_query']}")
        
        # Extract search results from agent response
        search_results = response.get('output', 'No search results found')
        state["search_results"] = search_results
        state["search_attempts"] = state.get("search_attempts", 0) + 1
        
        # Clear GPU memory after completion
        clear_cuda_memory()
        
        return state
    except Exception as e:
        st.error(f"Error in search: {str(e)}")
        state["search_results"] = f"Search error: {str(e)}"
        state["search_attempts"] = state.get("search_attempts", 0) + 1
        return state

def supervisor_decision_with_agent(state: RouterState) -> str:
    """Make supervisor decision using LangChain agent with tools."""
    try:
        # Clear GPU memory before starting
        clear_cuda_memory()
        
        # Get the selected model or use default
        current_model = state.get("llm_model", DEFAULT_LLM_MODEL)
        
        # Create LangChain agent with supervisor decision tool
        from langchain_community.llms import Ollama
        from langchain.agents import Tool, initialize_agent
        from langchain.schema import SystemMessage
        from langchain_core.tools import tool, Tool
        
        llm = Ollama(model=current_model)
        
        # Create supervisor decision tool
        def supervisor_tool_func(category: str, search_attempts: int = 0, max_attempts: int = 2, has_results: bool = False) -> str:
            """Make a supervisor decision about the next workflow action."""
            return supervisor_decision_tool.invoke({
                "category": state.get("category", "general purpose"),
                "search_attempts": state.get("search_attempts", 0), 
                "max_search_attempts": state.get("max_search_attempts", 2),
                "search_results": state.get("search_results")
            })
        
        decision_tool = Tool(
            name="make_decision",
            func=supervisor_tool_func,
            description="Make a routing decision about whether to search, answer, or end the workflow"
        )
        
        system_prompt = f"""You are a workflow routing expert.
        
        Current state information:
        - Query: {state['user_query']}
        - Category: {state['category']}
        - Search attempts so far: {state['search_attempts']}
        - Maximum allowed search attempts: {state['max_search_attempts']}
        - Has search results: {'Yes' if state.get('search_results') else 'No'}
        
        Use the make_decision tool to determine the next action for this workflow.
        The tool will return one of these actions:
        - "search" - perform web search
        - "answer" - generate final answer
        - "end" - end the workflow
        """
        
        agent = initialize_agent(
            tools=[decision_tool],
            llm=llm,
            agent_type="zero-shot-react-description",
            system_message=SystemMessage(content=system_prompt)
        )
        
        # Invoke the agent
        response = agent.invoke("Based on the current state, what action should we take next?")
        
        # Extract the decision from the agent's output
        import re
        decision_match = re.search(r'"(search|answer|end)"', response.get('output', ''))

        if decision_match:
            action = decision_match.group(1)
            st.info(f"🏹 Supervisor decision: **{action}**")
            return action
        else:
            # Fallback to answer if no clear decision found
            st.info("🏹 Supervisor decision: **answer** (fallback)")
            return "answer"

    except Exception as e:
        st.error(f"Error in supervisor decision with agent: {str(e)}")
        # Default to answering if there's an error
        return "answer"

def create_router_graph() -> StateGraph:
    """Create the LangGraph workflow for query routing using function nodes."""

    # Create the graph
    workflow = StateGraph(RouterState)

    # Add function nodes using the new tool-driven workflow
    workflow.add_node("categorize", categorize_query)
    workflow.add_node("supervisor", supervisor_decision)
    workflow.add_node("search", perform_search)
    workflow.add_node("answer", generate_final_answer)
    
    # Add edges
    workflow.add_edge(START, "categorize")
    
    # Add conditional edges from categorize to supervisor decision
    workflow.add_conditional_edges(
        "categorize",
        supervisor_decision_with_agent,
        {
            "search": "search",
            "answer": "answer",
            "end": END
        }
    )
    
    # Add conditional edges from search back to supervisor
    workflow.add_conditional_edges(
        "search",
        supervisor_decision_with_agent,
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
