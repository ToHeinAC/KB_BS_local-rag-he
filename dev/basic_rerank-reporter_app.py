import streamlit as st
import sys
import os
import json
import re
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path
from typing_extensions import TypedDict

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import required modules
from src.rag_helpers_v1_1 import get_report_llm_models
from src.utils_v1_1 import invoke_ollama
from src.state_v2_0 import ResearcherStateV2
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from langgraph.types import RunnableConfig

# Page configuration
st.set_page_config(
    page_title="Basic Rerank & Reporter",
    page_icon="📊",
    layout="wide"
)

def score_summary(initial_query: str, query: str, content: str, context: str, 
                  llm_model: str, language: str = "English") -> float:
    """
    Ask the LLM to score a single summary 0–10.
    """
    prompt = f"""
You are an expert evaluator of document summary relevance.

TASK: Make a deep analysis of the following summary and score it for its relevance and accuracy regarding the original query and the given context.
Ignore any scoring that is already within the SUMMARY TO ASSESS section.

ORIGINAL USER QUERY:
{initial_query}

SPECIFIC RESEARCH QUERY:
{query}

ADDITIONAL CONTEXT:
{context}

SUMMARY TO ASSESS:
{content}

SCORING CRITERIA (weights in parentheses):
1. Direct relevance to the original user query (50%) – Penalize mismatches in key terms (deduct 1-2 points if exact terms are not used, e.g., 'transportation' instead of 'local public transport').
2. Specificity and level of detail (20%) – Reward precise matches to query intent.
3. Alignment with the research query context (20%) – Ensure partial alignment doesn't inflate scores.
4. Factual accuracy and completeness (10%) – Deduct for inaccuracies or unrelated additions.
Example: "Score 10 only for perfect alignment; 7-8 for partial matches with some deviations; 5-6 for broad relevance without specifics."

INSTRUCTIONS:
Return ONLY a number between 0 and 10 using the following ranges:
- 10 = perfectly relevant and accurate
- 9-8 = very relevant with strong detail
- 7-6 = relevant but somewhat incomplete
- 5-4 = partially relevant
- 3-0 = poorly relevant or inaccurate
Be strictly critical: Reserve 10 for exact matches to the original query. 
Differentiate by prioritizing exact terms e.g. 'local public transport' from the user query and additional context over broader terms, e.g. 'transportation'. 
Compute sub-scores for each criterion, weight them, and average to final score. Output ONLY the final number (0-10, decimals allowed).
That is: Sub-score each criterion (0-10), multiply by weight, sum, and divide by 100 for final score. E.g., (Criterion1 * 0.5) + (Criterion2 * 0.2) + ...

One-Shot Example:
1. Perfect match to 'local public transport' matters: Score 10.
2. Partial relevance with public transport in general but not direct query: Score 7-8.
3. Discusses general transportation: Score 5-6 max.

Respond in {language}.
"""
    
    try:
        response = invoke_ollama(
            system_prompt="""You are a strict expert evaluator. 
            Provide only a single numerical score (0-10, with decimals) based on critical analysis; no explanations.""",
            user_prompt=prompt,
            model=llm_model
        )
        
        print(f"  [DEBUG] LLM scoring response: {response[:100]}...")
        
        # Extract numerical score from response
        match = re.search(r'\b(\d{1,2}(?:\.\d{1,2})?)\b', response)
        if match:
            score = float(match.group(1))
            # Ensure score is within valid range
            final_score = max(0.0, min(10.0, score))
            print(f"  [DEBUG] Extracted score: {final_score}")
            return final_score
        else:
            print(f"  [WARNING] Could not extract score from LLM response: {response[:100]}...")
            return 5.0  # Default to middle score instead of 1.0
    except Exception as e:
        print(f"  [ERROR] Failed to score summary: {str(e)}")
        return 5.0  # Default to middle score instead of 1.0


def reranker_node(state: ResearcherStateV2, config: RunnableConfig) -> ResearcherStateV2:
    """
    LangGraph node that reranks summaries based on relevance & accuracy.
    
    Args:
        state: ResearcherStateV2 containing search_summaries and other state
        config: RunnableConfig for the graph
    
    Returns:
        Updated ResearcherStateV2 with reranked summaries
    """
    print("--- Starting reranker node ---")
    
    # Extract state variables
    user_query = state.get("user_query", "")
    search_summaries = state.get("search_summaries", {})
    additional_context = state.get("additional_context", "")
    report_llm = state.get("report_llm", "qwen3:latest")
    detected_language = state.get("detected_language", "English")
    
    print(f"  [DEBUG] User query: {user_query}")
    print(f"  [DEBUG] Search summaries keys: {list(search_summaries.keys())}")
    print(f"  [DEBUG] Report LLM: {report_llm}")
    print(f"  [DEBUG] Language: {detected_language}")
    
    if not search_summaries:
        print("  [WARNING] No search summaries found for reranking")
        return state
    
    # Rerank summaries for each research query
    all_reranked_summaries = []
    
    for query, summaries in search_summaries.items():
        print(f"  [DEBUG] Processing query: {query} with {len(summaries)} summaries")
        
        if not summaries:
            print(f"  [WARNING] No summaries for query: {query}")
            continue
            
        # Rerank summaries for this query
        reranked = []
        for idx, doc in enumerate(summaries):
            content = doc.page_content
            print(f"  [DEBUG] Scoring summary {idx+1}/{len(summaries)} for query: {query[:50]}...")
            print(f"  [DEBUG] Content preview: {content[:100]}...")
            
            score = score_summary(
                initial_query=user_query,
                query=query,
                content=content,
                context=additional_context,
                llm_model=report_llm,
                language=detected_language
            )
            
            print(f"  [DEBUG] Score for summary {idx+1}: {score}")
            
            reranked.append({
                'summary': content,
                'score': score,
                'original_index': idx,
                'document': doc,
                'query': query
            })
        
        # Sort by score descending within this query
        reranked.sort(key=lambda x: x['score'], reverse=True)
        all_reranked_summaries.extend(reranked)
        print(f"  [DEBUG] Added {len(reranked)} reranked summaries for query: {query[:50]}...")
    
    # CRITICAL: Global sort by score after collecting all summaries from all queries
    all_reranked_summaries.sort(key=lambda x: x['score'], reverse=True)
    print(f"  [DEBUG] Applied global sort by score")
    
    # Store reranked summaries in state
    state["all_reranked_summaries"] = all_reranked_summaries
    state["current_position"] = "reranker"
    
    print(f"  [INFO] Reranked {len(all_reranked_summaries)} total summaries")
    print(f"  [DEBUG] Final scores after global sort: {[item['score'] for item in all_reranked_summaries]}")
    return state


def generate_final_answer_prompt(initial_query: str, reranked_summaries: List[Dict], 
                                additional_context: str = "", language: str = "English") -> str:
    """
    Create a prompt for generating the final answer using reranked summaries.
    """
    
    prompt = f"""You are an expert assistant providing comprehensive answers based on ranked document summaries.

TASK: Generate a complete and accurate answer to the user's query using the provided summaries. Prioritize information from higher-ranked summaries.

ORIGINAL QUERY:
{initial_query}

CONTEXT:
{additional_context}

RANKED SUMMARIES (ordered by relevance):

PRIMARY SOURCE (Highest Relevance - Score: {reranked_summaries[0]['score']:.1f}):
{reranked_summaries[0]['summary']}

SUPPORTING SOURCES:"""

    # Add remaining summaries as supporting sources
    for i, item in enumerate(reranked_summaries[1:], 2):
        prompt += f"""

Source {i} (Score: {item['score']:.1f}):
{item['summary']}"""

    prompt += f"""

INSTRUCTIONS:
• Base your answer PRIMARILY on the highest-ranked summary as it is most relevant to the query
• Use supporting sources to add context, details, or complementary information
• If supporting sources contradict the primary source, prioritize the primary source unless there's clear evidence of error
• Maintain accuracy and cite relevant legal references (§ sections) when mentioned
• Structure your response clearly with bullet points as preferred
• If information is incomplete, acknowledge limitations
• Focus on directly answering the original query
• Respond in {language} language

Generate a comprehensive answer that prioritizes the most relevant information while incorporating supporting details where appropriate."""

    return prompt


def report_writer_node(state: ResearcherStateV2, config: RunnableConfig) -> ResearcherStateV2:
    """
    LangGraph node that generates the final report using reranked summaries.
    
    Args:
        state: ResearcherStateV2 containing reranked summaries and other state
        config: RunnableConfig for the graph
    
    Returns:
        Updated ResearcherStateV2 with final answer
    """
    print("--- Starting report writer node ---")
    
    # Extract state variables
    user_query = state.get("user_query", "")
    all_reranked_summaries = state.get("all_reranked_summaries", [])
    additional_context = state.get("additional_context", "")
    report_llm = state.get("report_llm", "qwen3:latest")
    detected_language = state.get("detected_language", "English")
    
    print(f"  [DEBUG] User query: {user_query}")
    print(f"  [DEBUG] Reranked summaries count: {len(all_reranked_summaries)}")
    print(f"  [DEBUG] Report LLM: {report_llm}")
    print(f"  [DEBUG] Language: {detected_language}")
    
    if not all_reranked_summaries:
        print("  [WARNING] No reranked summaries found - cannot generate report")
        state["final_answer"] = "No relevant information found to generate a report."
        state["current_position"] = "report_writer"
        return state
    
    # Show sample of reranked summaries for debugging
    print(f"  [DEBUG] Sample reranked summaries:")
    for i, item in enumerate(all_reranked_summaries[:2]):
        print(f"    Summary {i+1}: Score={item.get('score', 'N/A')}, Query={item.get('query', 'N/A')[:50]}...")
        print(f"    Content preview: {str(item.get('summary', 'N/A'))[:100]}...")
    
    # Generate final report
    print("  [DEBUG] Generating final answer prompt...")
    prompt = generate_final_answer_prompt(
        initial_query=user_query,
        reranked_summaries=all_reranked_summaries,
        additional_context=additional_context,
        language=detected_language
    )
    
    print(f"  [DEBUG] Prompt length: {len(prompt)} characters")
    print(f"  [DEBUG] Prompt preview: {prompt[:200]}...")
    
    try:
        print(f"  [DEBUG] Calling invoke_ollama with model: {report_llm}")
        response = invoke_ollama(
            system_prompt="You are an expert research analyst. Provide comprehensive, well-structured reports based on the provided information.",
            user_prompt=prompt,
            model=report_llm
        )
        
        print(f"  [DEBUG] LLM response length: {len(response)} characters")
        print(f"  [DEBUG] Response preview: {response[:200]}...")
        
        state["final_answer"] = response
        print("  [INFO] Final report generated successfully")
        
    except Exception as e:
        print(f"  [ERROR] Exception in final report generation: {str(e)}")
        state["final_answer"] = f"Error occurred during final report generation: {str(e)}. Check logs for details."
        state["current_position"] = "report_writer_error"
        return state
    
    state["current_position"] = "report_writer"
    return state


def create_rerank_reporter_graph():
    """
    Create the LangGraph workflow for reranking and report generation.
    
    Returns:
        Compiled LangGraph workflow
    """
    print("Creating rerank-reporter graph...")
    
    # Create the workflow
    workflow = StateGraph(ResearcherStateV2)
    
    # Add nodes
    workflow.add_node("reranker", reranker_node)
    workflow.add_node("report_writer", report_writer_node)
    
    # Add edges
    workflow.add_edge("reranker", "report_writer")
    workflow.add_edge("report_writer", END)
    
    # Set entry point
    workflow.set_entry_point("reranker")
    
    return workflow.compile()


def create_sample_documents() -> Dict[str, List[Document]]:
    """Create sample documents for testing."""
    sample_docs = {
        "What are the benefits of renewable energy?": [
            Document(
                page_content="Renewable energy sources like solar and wind power offer significant environmental benefits by reducing greenhouse gas emissions and air pollution. They help combat climate change and improve public health.",
                metadata={"source": "environmental_report.pdf", "page": 1}
            ),
            Document(
                page_content="Economic advantages of renewable energy include job creation in manufacturing, installation, and maintenance sectors. The renewable energy industry has created millions of jobs worldwide.",
                metadata={"source": "economic_analysis.pdf", "page": 3}
            ),
            Document(
                page_content="Energy independence is a key benefit of renewable sources. Countries can reduce reliance on fossil fuel imports and achieve greater energy security through domestic renewable resources.",
                metadata={"source": "energy_policy.pdf", "page": 2}
            )
        ],
        "How does solar power work?": [
            Document(
                page_content="Solar photovoltaic cells convert sunlight directly into electricity through the photovoltaic effect. When photons hit the solar cell, they knock electrons loose from atoms, generating electrical current.",
                metadata={"source": "solar_tech.pdf", "page": 1}
            ),
            Document(
                page_content="Solar thermal systems use mirrors or lenses to concentrate sunlight to heat a fluid, which then generates steam to drive turbines for electricity production.",
                metadata={"source": "solar_thermal.pdf", "page": 2}
            )
        ]
    }
    return sample_docs


def generate_graph_visualization():
    """Generate and display the LangGraph workflow visualization."""
    try:
        graph = create_rerank_reporter_graph()
        
        # Try to get Mermaid PNG
        try:
            img = graph.get_graph().draw_mermaid_png()
            return img, "png"
        except Exception as e:
            # Fallback to Mermaid text
            mermaid_code = graph.get_graph().draw_mermaid()
            return mermaid_code, "mermaid"
            
    except Exception as e:
        return f"Error generating visualization: {str(e)}", "error"


def main():
    st.title("📊 LangGraph Rerank & Reporter")
    st.markdown("*Advanced reranking and report generation using LangGraph state logic*")
    
    # Initialize session state
    if "graph" not in st.session_state:
        st.session_state.graph = create_rerank_reporter_graph()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Load available models
        try:
            report_llm_models = get_report_llm_models()
        except Exception as e:
            st.error(f"Error loading models: {e}")
            report_llm_models = ["qwen3:latest", "deepseek-r1:latest"]
        
        # Model selection
        selected_model = st.selectbox(
            "Select LLM Model",
            options=report_llm_models,
            index=0
        )
        
        # Language selection
        language = st.selectbox(
            "Language",
            options=["English", "German"],
            index=0
        )
        
        # Additional context input
        additional_context = st.text_area(
            "Additional Context",
            placeholder="Enter any additional context or conversation history...",
            height=100
        )
        
        # Graph visualization in sidebar
        st.subheader("🕸️ Workflow Graph")
        try:
            graph_result, graph_type = generate_graph_visualization()
            
            if graph_type == "png":
                st.image(graph_result, caption="LangGraph Workflow", use_container_width=True)
            elif graph_type == "mermaid":
                st.code(graph_result, language="mermaid")
            else:
                st.error(graph_result)
                
        except Exception as e:
            st.warning(f"Could not generate graph visualization: {str(e)}")
            st.text("reranker → report_writer → END")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Input")
        
        # User query input
        initial_query = st.text_area(
            "Initial Query",
            placeholder="Enter your research question...",
            height=100,
            key="initial_query"
        )
        
        # Sample documents section
        st.subheader("📄 Sample Data")
        use_sample = st.checkbox("Use sample documents", value=True)
        
        if use_sample:
            sample_docs = create_sample_documents()
            
            # Display sample documents structure
            st.info(f"Using {len(sample_docs)} sample research queries with documents")
            for query, docs in sample_docs.items():
                with st.expander(f"Query: {query} ({len(docs)} docs)"):
                    for i, doc in enumerate(docs, 1):
                        st.write(f"**Document {i}:**")
                        st.write(doc.page_content[:200] + "...")
                        st.caption(f"Source: {doc.metadata.get('source', 'Unknown')}")
        
        else:
            st.info("Manual input mode - configure research queries and summaries below")
            
            # Manual input for research queries and summaries
            research_queries_input = st.text_area(
                "Research Queries (one per line)",
                placeholder="What are the benefits of renewable energy?\nHow does solar power work?",
                height=100,
                key="research_queries"
            )
            
            # Document content input
            st.subheader("📄 Document Summaries")
            st.write("Enter summaries for each query above:")
            
            if research_queries_input:
                research_queries = [q.strip() for q in research_queries_input.split('\n') if q.strip()]
                
                # Create input fields for each query
                custom_summaries = {}
                for i, query in enumerate(research_queries):
                    st.write(f"**Query {i+1}:** {query}")
                    summary_text = st.text_area(
                        f"Summary for query {i+1}",
                        placeholder="Enter document summary content...",
                        height=100,
                        key=f"summary_{i}"
                    )
                    if summary_text:
                        custom_summaries[query] = [Document(page_content=summary_text, metadata={"custom": True})]
    
    with col2:
        st.header("📊 Results")
        
        if st.button("🚀 Process with LangGraph", type="primary"):
            if not initial_query:
                st.error("Please enter an initial query.")
                return
            
            try:
                # Prepare data based on input method
                if use_sample:
                    search_summaries = create_sample_documents()
                else:
                    if not research_queries_input:
                        st.error("Please provide research queries.")
                        return
                    
                    research_queries = [q.strip() for q in research_queries_input.split('\n') if q.strip()]
                    search_summaries = {}
                    
                    for i, query in enumerate(research_queries):
                        summary_key = f"summary_{i}"
                        if summary_key in st.session_state and st.session_state[summary_key]:
                            docs = [Document(page_content=st.session_state[summary_key], metadata={"custom": True})]
                            search_summaries[query] = docs
                
                if not search_summaries:
                    st.error("No summaries provided. Please add document content.")
                    return
                
                # Process with LangGraph
                with st.spinner("🔄 Processing with LangGraph workflow..."):
                    # Prepare initial state
                    initial_state = {
                        "user_query": initial_query,
                        "search_summaries": search_summaries,
                        "additional_context": additional_context,
                        "report_llm": selected_model,
                        "detected_language": language,
                        "current_position": 0,
                        "research_queries": list(search_summaries.keys()),
                        "final_answer": "",
                        "all_reranked_summaries": []
                    }
                    
                    # Execute the graph
                    final_state = st.session_state.graph.invoke(initial_state)
                    
                    # Display results
                    st.success("✅ LangGraph processing complete!")
                    
                    # Display processing steps
                    with st.expander("🔍 Processing Details", expanded=True):
                        st.write(f"**Current Position:** {final_state.get('current_position', 'unknown')}")
                        st.write(f"**Research Queries Processed:** {len(final_state.get('research_queries', []))}")
                        
                        # Debug state keys
                        st.write("**State Keys:**", list(final_state.keys()))
                        st.write("**Reranked summaries count:**", len(final_state.get("all_reranked_summaries", [])))
                        st.write("**Reranked summaries (alt):**", len(final_state.get("reranked_summaries", [])))
                        
                        # Display reranked documents
                        reranked_summaries = final_state.get("all_reranked_summaries", [])
                        # Also check for reranked_summaries in the state
                        if not reranked_summaries:
                            reranked_summaries = final_state.get("reranked_summaries", [])
                        if reranked_summaries:
                            st.subheader("📊 Reranked Documents")
                            
                            # Summary statistics
                            total_docs = len(reranked_summaries)
                            avg_score = sum(item.get('score', 0) for item in reranked_summaries) / total_docs if total_docs > 0 else 0
                            
                            col_stats1, col_stats2, col_stats3 = st.columns(3)
                            with col_stats1:
                                st.metric("Total Documents", total_docs)
                            with col_stats2:
                                st.metric("Average Score", f"{avg_score:.2f}/10")
                            with col_stats3:
                                max_score = max([item.get('score', 0) for item in reranked_summaries], default=0)
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
                    
                    # Display final report
                    final_answer = final_state.get("final_answer", "")
                    if final_answer and final_answer.strip():
                        st.subheader("📋 Final Report")
                        st.markdown(final_answer)
                        
                        # Download button for the report
                        st.download_button(
                            label="📥 Download Report",
                            data=final_answer,
                            file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                            mime="text/markdown"
                        )
                    else:
                        st.subheader("📄 Final Report")
                        st.warning("No report generated. This could be due to:")
                        st.markdown("- No reranked summaries found")
                        st.markdown("- All summaries scored below relevance threshold")
                        st.markdown("- Report generation failed")
                        
                        # Show debug info
                        with st.expander("🔧 Debug Information"):
                            st.json({
                                "has_final_answer": bool(final_answer),
                                "final_answer_length": len(final_answer) if final_answer else 0,
                                "reranked_summaries_count": len(reranked_summaries),
                                "research_queries": final_state.get("research_queries", []),
                                "user_query": final_state.get("user_query", "")
                            })
                    
            except Exception as e:
                st.error(f"Error during processing: {str(e)}")
                st.exception(e)
        
    # Debug information
    with st.expander("🔧 Debug Information"):
        st.write("**Session State Keys:**")
        st.write(list(st.session_state.keys()))
        st.write("**Graph Available:**", st.session_state.graph is not None)

if __name__ == "__main__":
    main()
