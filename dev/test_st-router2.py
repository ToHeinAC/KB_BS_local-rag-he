import streamlit as st
import os
from typing import Annotated, Literal
from dotenv import load_dotenv
import torch
from langchain_ollama import ChatOllama #accepts .bind_tools()
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt.chat_agent_executor import create_tool_calling_executor
import requests
import json

# Load environment variables
load_dotenv()

def clear_cuda_memory():
    """Clear CUDA memory cache to free up GPU resources between queries."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        print("CUDA memory cache cleared")

# Initialize LLM
@st.cache_resource
def get_llm():
    return ChatOllama(model='qwen3:1.7b', temperature=0.1)

# Define tools
@tool
def tavily_search(query: str) -> str:
    """Search for recent information using Tavily API.
    
    Args:
        query: The search query to find information about
    """
    try:
        st.write("Used tavily_search ...")
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return "Error: Tavily API key not found in environment variables"
        
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": api_key,
            "query": query,
            "search_depth": "basic",
            "include_answer": True,
            "include_raw_content": False,
            "max_results": 3
        }
        
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            data = response.json()
            results = []
            
            # Add the direct answer if available
            if data.get("answer"):
                results.append(f"Direct Answer: {data['answer']}")
            
            # Add search results
            for result in data.get("results", [])[:3]:
                results.append(f"Title: {result.get('title', 'N/A')}")
                results.append(f"Content: {result.get('content', 'N/A')}")
                results.append(f"URL: {result.get('url', 'N/A')}")
                results.append("---")
            
            return "\n".join(results) if results else "No search results found"
        else:
            return f"Search failed with status code: {response.status_code}"
    except Exception as e:
        return f"Search error: {str(e)}"

@tool
def answerer(context: str, query: str) -> str:
    """Generate an answer based on the provided context and user query.
    
    Args:
        context: The context information to base the answer on
        query: The original user query to answer
    """
    st.write("Used answerer ...")
    llm = get_llm()
    prompt = f"""Based on the following context, provide a comprehensive answer to the user's query.

Context:
{context}

User Query: {query}

Instructions:
- Use the context information to provide an accurate and helpful answer
- If the context doesn't contain enough information, acknowledge this
- Be concise but thorough
- Focus on directly answering the user's question

Answer:"""
    
    try:
        response = llm.invoke(prompt)
        return response
    except Exception as e:
        return f"Error generating answer: {str(e)}"

@tool
def check_answer(answer: str, original_query: str) -> str:
    """Check if the provided answer is sufficient and complete for the original query.
    
    Args:
        answer: The answer to evaluate
        original_query: The original user query
    """
    st.write("Used check_answer ...")
    llm = get_llm()
    prompt = f"""Evaluate the following answer for completeness and accuracy regarding the original query.

Original Query: {original_query}

Answer to Evaluate:
{answer}

Instructions:
- Determine if the answer adequately addresses the user's query
- Check for completeness, accuracy, and relevance
- If insufficient, suggest what additional information is needed
- Respond with either "SUFFICIENT" or "INSUFFICIENT" followed by your reasoning
- If insufficient, provide a specific search query suggestion to improve the answer

Evaluation:"""
    
    try:
        response = llm.invoke(prompt)
        return response
    except Exception as e:
        return f"Error checking answer: {str(e)}"

# Create tools list
tools = [tavily_search, answerer, check_answer]
tool_node = ToolNode(tools)

# Define the workflow state
class WorkflowState(MessagesState):
    query_category: str = ""
    search_results: str = ""
    current_answer: str = ""
    search_needed: bool = False
    max_iterations: int = 3
    current_iteration: int = 0

# Node functions
def categorizer(state: WorkflowState) -> WorkflowState:
    """Categorize the user query and determine if search is needed."""
    llm = get_llm()
    
    # Get the last human message
    last_message = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_message = msg.content
            break
    
    if not last_message:
        return {
            **state,
            "query_category": "unknown",
            "search_needed": False
        }
    
    prompt = f"""Analyze the following user query and categorize it. Also determine if web search is needed.

User Query: {last_message}

Categories:
- general_knowledge: Questions about well-known facts, concepts, or common knowledge
- recent_news: Questions about weather forecasts, current events, recent developments, or time-sensitive information (including weather queries)
- practical_guide: How-to questions, tutorials, or step-by-step instructions
- specific_search: Questions requiring specific, current, or detailed information not in general knowledge
- casual_conversation: Greetings, casual chat, or non-informational queries

Instructions:
- Choose ONE category that best fits the query
- Weather forecasts and current weather should always be categorized as 'recent_news'
- Determine if web search is needed (true/false)
- Recent news, weather queries, specific searches, and some practical guides typically need search
- General knowledge and casual conversation typically don't need search

Respond in this exact format:
Category: [category_name]
Search Needed: [true/false]
Reasoning: [brief explanation]"""
    
    try:
        response = llm.invoke(prompt)
        
        # Extract content from AIMessage if needed
        if hasattr(response, 'content'):
            response_content = response.content
        else:
            response_content = str(response)
        
        # Parse the response with more robust logic
        category = "general_knowledge"  # default
        search_needed = False  # default
        
        # Convert response to lowercase for easier parsing
        response_lower = response_content.lower()
        
        # Look for category indicators in the response
        if "recent_news" in response_lower or "recent news" in response_lower:
            category = "recent_news"
        elif "practical_guide" in response_lower or "practical guide" in response_lower:
            category = "practical_guide"
        elif "specific_search" in response_lower or "specific search" in response_lower:
            category = "specific_search"
        elif "casual_conversation" in response_lower or "casual conversation" in response_lower:
            category = "casual_conversation"
        elif "general_knowledge" in response_lower or "general knowledge" in response_lower:
            category = "general_knowledge"
        
        # Look for search needed indicators
        if "search needed: true" in response_lower or "true" in response_lower:
            search_needed = True
        elif "search needed: false" in response_lower or "false" in response_lower:
            search_needed = False
        
        # Special handling for weather queries - always recent_news and search needed
        if "weather" in last_message.lower() or "forecast" in last_message.lower():
            category = "recent_news"
            search_needed = True
        
        # Debug output
        st.write(f"🏷️ Query categorized as: {category}")
        st.write(f"🔍 Search needed: {search_needed}")
        st.write(f"📝 LLM Response: {response_content[:200]}..." if len(response_content) > 200 else f"📝 LLM Response: {response_content}")
        
        return {
            **state,
            "query_category": category,
            "search_needed": search_needed
        }
    except Exception as e:
        st.error(f"❌ Categorizer error: {str(e)}")
        # For weather queries, default to recent_news even on error
        if "weather" in last_message.lower() or "forecast" in last_message.lower():
            return {
                **state,
                "query_category": "recent_news",
                "search_needed": True
            }
        return {
            **state,
            "query_category": "general_knowledge",
            "search_needed": False
        }

def supervisor_router(state: WorkflowState) -> Literal["search_agent", "answer_agent", "check_agent", "end"]:
    """Simple supervisor that routes based on workflow state."""
    
    # Check if we've exceeded max iterations
    if state.get("current_iteration", 0) >= state.get("max_iterations", 5):
        st.write("🔄 Maximum iterations reached. Ending workflow.")
        return "end"
    
    # Check if answer was deemed insufficient and we need to search again (check this first)
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and "Answer needs improvement" in str(msg.content):
            st.write("🔄 Answer insufficient, routing back to search_agent")
            return "search_agent"
    
    # Check if we have completed validation (answer is sufficient)
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage):
            msg_content = str(msg.content) if hasattr(msg, 'content') else str(msg)
            if "Answer is sufficient - workflow complete" in msg_content:
                st.write("🏁 Answer validated as sufficient, ending workflow")
                return "end"
    
    # If search is needed and we haven't searched yet, go to search
    if state.get("search_needed", False) and not state.get("search_results"):
        st.write("🔍 Routing to search_agent for web search")
        return "search_agent"
    
    # If we have search results but no answer, generate answer
    if state.get("search_results") and not state.get("current_answer"):
        st.write("💭 Routing to answer_agent to generate response")
        return "answer_agent"
    
    # If search is not needed and we don't have an answer, generate one
    if not state.get("search_needed", False) and not state.get("current_answer"):
        st.write("💭 Routing to answer_agent (no search needed)")
        return "answer_agent"
    
    # If we have an answer but haven't checked it yet, check it
    if state.get("current_answer"):
        # Check if validation has already been done
        validation_done = any(
            "Answer validation:" in str(msg.content) if hasattr(msg, 'content') else False
            for msg in state["messages"] if isinstance(msg, AIMessage)
        )
        
        if not validation_done:
            st.write("✅ Routing to check_agent for answer validation")
            return "check_agent"
    
    # Default to end if we get here
    st.write("🏁 Workflow complete, ending")
    return "end"

def search_agent_node(state: WorkflowState) -> WorkflowState:
    """Search agent that handles web search tasks."""
    # Get the original query
    query = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            query = msg.content
            break
    
    if not query:
        return state
    
    # Check if we need to use a refined search query from check_answer
    search_query = query
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage) and "search query suggestion" in msg.content.lower():
            # Extract suggested query from the message
            lines = msg.content.split('\n')
            for line in lines:
                if 'suggest' in line.lower() or 'query' in line.lower():
                    # Simple extraction - in practice you might want more sophisticated parsing
                    if ':' in line:
                        search_query = line.split(':', 1)[1].strip()
                    break
            break
    
    # Execute the search
    try:
        result = tavily_search.invoke({"query": search_query})
        
        return {
            **state,
            "messages": state["messages"] + [AIMessage(content=f"Search completed for: {search_query}\n\nResults: {result}")],
            "search_results": result,
            "current_iteration": state.get("current_iteration", 0) + 1
        }
    except Exception as e:
        return {
            **state,
            "messages": state["messages"] + [AIMessage(content=f"Search failed: {str(e)}")],
            "current_iteration": state.get("current_iteration", 0) + 1
        }

def answer_agent_node(state: WorkflowState) -> WorkflowState:
    """Answer agent that generates responses based on context."""
    # Get the original query
    query = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            query = msg.content
            break
    
    if not query:
        return state
    
    # Get context (search results if available)
    context = state.get("search_results", "No additional context available.")
    
    # Execute the answerer
    try:
        result = answerer.invoke({"context": context, "query": query})
        
        return {
            **state,
            "messages": state["messages"] + [AIMessage(content=f"Generated answer: {result}")],
            "current_answer": result
        }
    except Exception as e:
        return {
            **state,
            "messages": state["messages"] + [AIMessage(content=f"Answer generation failed: {str(e)}")]
        }

def check_agent_node(state: WorkflowState) -> WorkflowState:
    """Check agent that validates answer quality and completeness."""
    st.write("🔍 Check agent starting validation...")
    
    # Get the original query and current answer
    query = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            query = msg.content
            break
    
    current_answer = state.get("current_answer", "")
    
    # Extract content from current_answer if it's an AIMessage
    if hasattr(current_answer, 'content'):
        answer_content = current_answer.content
    else:
        answer_content = str(current_answer)
    
    st.write(f"📝 Query: {query[:50]}..." if query and len(query) > 50 else f"📝 Query: {query}")
    st.write(f"💬 Answer: {answer_content[:100]}..." if answer_content and len(answer_content) > 100 else f"💬 Answer: {answer_content}")
    
    if not query or not answer_content:
        st.write("❌ Missing query or answer, skipping validation")
        return state
    
    # Execute the checker
    try:
        result = check_answer.invoke({"answer": answer_content, "original_query": query})
        
        # Extract content from AIMessage if needed
        if hasattr(result, 'content'):
            result_content = result.content
        else:
            result_content = str(result)
        
        st.write(f"🔍 Validation result: {result_content[:200]}..." if len(result_content) > 200 else f"🔍 Validation result: {result_content}")
        
        # If answer is insufficient, we might need to search again
        if "INSUFFICIENT" in result_content.upper():
            st.write("❌ Answer deemed insufficient, will search for more information")
            return {
                **state,
                "messages": state["messages"] + [AIMessage(content=f"Answer validation: {result_content}\n\nAnswer needs improvement - will search for more information.")],
                "search_needed": True,  # Enable another search round
                "current_answer": "",  # Reset answer to generate a new one
            }
        
        st.write("✅ Answer deemed sufficient, workflow should complete")
        return {
            **state,
            "messages": state["messages"] + [AIMessage(content=f"Answer validation: {result_content}\n\nAnswer is sufficient - workflow complete.")]
        }
    except Exception as e:
        st.write(f"❌ Check agent error: {str(e)}")
        return {
            **state,
            "messages": state["messages"] + [AIMessage(content=f"Answer checking failed: {str(e)}")]
        }

# Build the graph with simple conditional routing
def create_workflow():
    workflow = StateGraph(WorkflowState)
    
    # Add nodes
    workflow.add_node("categorizer", categorizer)
    workflow.add_node("search_agent", search_agent_node)
    workflow.add_node("answer_agent", answer_agent_node)
    workflow.add_node("check_agent", check_agent_node)
    
    # Add edges
    workflow.add_edge(START, "categorizer")
    
    # Conditional routing from categorizer
    workflow.add_conditional_edges(
        "categorizer",
        supervisor_router,
        {
            "search_agent": "search_agent",
            "answer_agent": "answer_agent",
            "check_agent": "check_agent",
            "end": END
        }
    )
    
    # All agents go back to supervisor router for next decision
    workflow.add_conditional_edges(
        "search_agent",
        supervisor_router,
        {
            "search_agent": "search_agent",
            "answer_agent": "answer_agent",
            "check_agent": "check_agent",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "answer_agent",
        supervisor_router,
        {
            "search_agent": "search_agent",
            "answer_agent": "answer_agent",
            "check_agent": "check_agent",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "check_agent",
        supervisor_router,
        {
            "search_agent": "search_agent",
            "answer_agent": "answer_agent",
            "check_agent": "check_agent",
            "end": END
        }
    )
    
    return workflow.compile(checkpointer=MemorySaver())

# Streamlit UI
def main():
    st.set_page_config(
        page_title="Smart Router with LangGraph",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Smart Router with LangGraph")
    st.markdown("An intelligent query router using LangGraph with categorization, search, and answer validation.")
    
    # Add custom CSS styling
    st.markdown("""
    <style>
    .answer-box {
        background-color: #2e5c2e;
        border-left: 5px solid #63b463;
        border-radius: 5px;
        padding: 15px;
        margin: 15px 0;
        color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .answer-box h3 {
        margin-top: 0;
        color: #63b463;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if "workflow" not in st.session_state:
        st.session_state.workflow = create_workflow()
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = "main_thread"
    
    # Workflow visualization
    with st.expander("🔄 Show Workflow Graph", expanded=False):
        try:
            graph_png = st.session_state.workflow.get_graph().draw_mermaid_png()
            st.image(graph_png, caption="LangGraph Workflow", use_container_width=True)
        except Exception as e:
            st.error(f"Could not generate workflow visualization: {e}")
    
    # Example queries
    st.sidebar.header("📝 Example Queries")
    example_queries = [
        "What's the weather like today?",
        "How do I make a perfect cup of coffee?",
        "What are the latest developments in AI?",
        "Explain quantum computing",
        "What happened in the stock market today?"
    ]
    
    for query in example_queries:
        if st.sidebar.button(query, key=f"example_{hash(query)}"):
            st.session_state.user_input = query
    
    # Configuration controls
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Main input
        user_query = st.text_input(
            "Enter your query:",
            value=st.session_state.get("user_input", ""),
            placeholder="Ask me anything...",
            key="query_input"
        )
    
    with col2:
        # Max steps control
        max_steps = st.number_input(
            "Max Steps:",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            help="Maximum number of workflow steps before termination"
        )
    
    if st.button("🚀 Process Query", type="primary") and user_query:
        clear_cuda_memory()
        
        with st.spinner("Processing your query..."):
            try:
                # Create initial state
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                initial_state = {
                    "messages": [HumanMessage(content=user_query)],
                    "query_category": "",
                    "search_results": "",
                    "current_answer": "",
                    "search_needed": False,
                    "max_iterations": max_steps,
                    "current_iteration": 0
                }
                
                # Execute workflow with recursion limit
                config["recursion_limit"] = max_steps * 2  # Allow some buffer
                result = st.session_state.workflow.invoke(initial_state, config)
                
                # Display results
                st.success("✅ Query processed successfully!")
                
                # Show categorization
                if result.get("query_category"):
                    st.info(f"**Query Category:** {result['query_category']}")
                    st.info(f"**Search Needed:** {'Yes' if result.get('search_needed') else 'No'}")
                
                # Show final answer with styled box
                final_answer = result.get("current_answer", "")
                
                # Extract content if it's an AIMessage
                if hasattr(final_answer, 'content'):
                    final_answer_content = final_answer.content
                elif final_answer:
                    final_answer_content = str(final_answer)
                else:
                    # Fallback: look for answer in messages
                    final_answer_content = None
                    for msg in reversed(result["messages"]):
                        if isinstance(msg, AIMessage) and "Generated answer:" in str(msg.content):
                            # Extract the answer part after "Generated answer:"
                            content = str(msg.content)
                            if "Generated answer:" in content:
                                final_answer_content = content.split("Generated answer:", 1)[1].strip()
                                break
                
                if final_answer_content:
                    st.markdown(f"""
                    <div class="answer-box">
                    <h3>💡 Final Answer</h3>
                    <div>{final_answer_content}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Debug expander for final answer
                    with st.expander("🐛 Final Answer Debug", expanded=False):
                        if hasattr(result.get("current_answer", ""), 'content'):
                            st.text_area(
                                "Raw Content:",
                                value=result["current_answer"].content,
                                height=200,
                                disabled=True
                            )
                        else:
                            st.text("No raw content available or content is not an AIMessage object")
                
                # Show workflow execution details
                with st.expander("🔍 Execution Details", expanded=False):
                    st.json({
                        "iterations": result.get("current_iteration", 0),
                        "category": result.get("query_category", ""),
                        "search_used": bool(result.get("search_results")),
                        "total_messages": len(result["messages"])
                    })
                    
                    # Show message history
                    st.markdown("**Message History:**")
                    for i, msg in enumerate(result["messages"]):
                        msg_type = type(msg).__name__
                        content = getattr(msg, 'content', str(msg))[:200] + "..." if len(str(getattr(msg, 'content', str(msg)))) > 200 else getattr(msg, 'content', str(msg))
                        st.text(f"{i+1}. {msg_type}: {content}")
                
            except Exception as e:
                st.error(f"❌ Error processing query: {str(e)}")
                st.exception(e)
        
        clear_cuda_memory()

if __name__ == "__main__":
    main()
