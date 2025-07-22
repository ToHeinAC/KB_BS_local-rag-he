"""
LangGraph Multi-Graph Showcase with Streamlit

This application demonstrates:
1. Using two LangGraph workflows with TypedDict states
2. A Classifier graph that categorizes user queries into 5 categories
3. An Answerer graph that uses the classifier's output to answer the query
4. State sharing between multiple LangGraph workflows
5. Visualization of both graph workflows
"""

import streamlit as st
from typing import Literal, List, Dict, Any
from typing_extensions import TypedDict, NotRequired
from IPython.display import Image, display
import io
import base64

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.llms import Ollama
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles

# Set page title and configuration
st.set_page_config(page_title="LangGraph Multi-Graph Showcase", layout="wide")
st.title("LangGraph Multi-Graph Showcase")

# Define TypedDict states for both workflows
class ClassifierState(TypedDict):
    """State for the Classifier workflow."""
    query: str
    category: NotRequired[Literal["general purpose", "recent news", "practical guide", "legal guide", "casual conversation"]]
    confidence: NotRequired[float]

class AnswererState(TypedDict):
    """State for the Answerer workflow that includes classifier state."""
    query: str
    category: Literal["general purpose", "recent news", "practical guide", "legal guide", "casual conversation"]
    answer: NotRequired[str]
    sources: NotRequired[List[str]]
    confidence: NotRequired[float]

# Set up Ollama with Llama3.2 for consistency
@st.cache_resource
def get_llm():
    """Initialize and cache the LLM."""
    return Ollama(model="llama3.2")

llm = get_llm()

# Define the classifier node function
def classify_query(state: ClassifierState) -> ClassifierState:
    """Classify the user query into one of 5 categories."""
    prompt = ChatPromptTemplate.from_template("""
    You are a query classifier that categorizes user questions into one of five categories:
    1. general purpose - factual information and general knowledge
    2. recent news - questions about current events or recent developments
    3. practical guide - how-to instructions or practical advice
    4. legal guide - information about laws, regulations, or legal procedures
    5. casual conversation - social chat, greetings, or personal interaction
    
    User query: {query}
    
    Respond with a JSON object with these fields:
    - category: one of ["general purpose", "recent news", "practical guide", "legal guide", "casual conversation"]
    - confidence: a number between 0.0 and 1.0 indicating your confidence in this classification
    
    JSON:
    """)
    
    chain = prompt | llm | JsonOutputParser()
    result = chain.invoke({"query": state["query"]})
    
    return {
        "query": state["query"],
        "category": result["category"],
        "confidence": result["confidence"]
    }

# Define the answerer node function
def answer_query(state: AnswererState) -> AnswererState:
    """Generate an answer for the user query based on its category."""
    prompt = ChatPromptTemplate.from_template("""
    You are a helpful AI assistant. The user has asked the following query:
    
    Query: {query}
    
    This query has been classified as: {category}
    
    Based on this classification, provide a helpful response. Consider the following guidelines:
    
    - For general purpose queries: Provide factual, informative responses based on general knowledge.
    - For recent news: Acknowledge that your knowledge has a cutoff date and suggest checking latest sources.
    - For practical guides: Provide step-by-step instructions or practical advice when applicable.
    - For legal guides: Include disclaimers about not providing legal advice and suggest consulting professionals.
    - For casual conversation: Engage in a friendly, conversational manner.
    
    Your answer:
    """)
    
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "query": state["query"],
        "category": state["category"]
    })
    
    return {
        "query": state["query"],
        "category": state["category"],
        "answer": answer,
        "confidence": state.get("confidence", 0.8)
    }

# Create the classifier graph
def create_classifier_graph():
    """Create and compile the classifier workflow graph."""
    workflow = StateGraph(ClassifierState)
    workflow.add_node("classifier", classify_query)
    workflow.add_edge(START, "classifier")
    workflow.add_edge("classifier", END)
    
    return workflow.compile()

# Create the answerer graph
def create_answerer_graph():
    """Create and compile the answerer workflow graph."""
    workflow = StateGraph(AnswererState)
    workflow.add_node("answerer", answer_query)
    workflow.add_edge(START, "answerer")
    workflow.add_edge("answerer", END)
    
    return workflow.compile()

# Create function to display graph visualization
def display_graph_image(graph, title):
    """Generate and display a graph visualization."""
    # Generate Mermaid PNG
    try:
        # Use basic mermaid drawing without custom styles
        png_data = graph.get_graph().draw_mermaid_png()
        # Display image in Streamlit
        st.subheader(title)
        st.image(png_data, use_column_width=True)
    except Exception as e:
        st.error(f"Error generating graph visualization: {e}")

# Create the two workflow graphs
classifier_graph = create_classifier_graph()
answerer_graph = create_answerer_graph()

# Streamlit UI layout
st.markdown("### Enter your query below:")
query = st.text_input("Query", key="query_input")

if st.button("Submit") and query:
    # Create two columns for results display
    col1, col2 = st.columns(2)
    
    with st.spinner("Processing..."):
        # Execute classifier graph
        classifier_result = classifier_graph.invoke({"query": query})
        
        # Execute answerer graph with classifier result
        answerer_input = {
            "query": classifier_result["query"],
            "category": classifier_result["category"]
        }
        answerer_result = answerer_graph.invoke(answerer_input)
        
        # Display results
        with col1:
            st.subheader("Classification Result")
            st.write(f"**Category:** {classifier_result['category']}")
            st.write(f"**Confidence:** {classifier_result['confidence']:.2f}")
            
            # Display classifier graph
            display_graph_image(classifier_graph, "Classifier Graph")
        
        with col2:
            st.subheader("Answer")
            st.write(answerer_result["answer"])
            
            # Display answerer graph
            display_graph_image(answerer_graph, "Answerer Graph")

# Add explanation section
with st.expander("How it works"):
    st.markdown("""
    ### LangGraph Multi-Graph Workflow
    
    This application demonstrates the use of multiple LangGraph workflows with state sharing:
    
    1. **Classifier Graph**: Takes a user query and categorizes it into one of five categories.
    2. **Answerer Graph**: Takes the output state from the classifier and generates an appropriate answer.
    
    The state is shared between the two graphs by passing the output of the first graph as input to the second.
    Each graph is visualized using LangGraph's built-in visualization capabilities.
    
    The LLM used for both classification and answering is Llama 3.2 via Ollama.
    """)
