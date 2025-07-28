import streamlit as st
import sys
import os
import json
import re
from typing import Dict, List, Any
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import required modules
from src.rag_helpers_v1_1 import get_report_llm_models, invoke_ollama
from langchain_core.documents import Document

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

TASK: Score the following summary for its relevance and accuracy regarding the original query and the given context.

ORIGINAL USER QUERY:
{initial_query}

SPECIFIC RESEARCH QUERY:
{query}

ADDITIONAL CONTEXT:
{context}

SUMMARY TO ASSESS:
{content}

SCORING CRITERIA (weights in parentheses):
1. Direct relevance to the original user query (40%)
2. Specificity and level of detail (25%)
3. Alignment with the research query context (20%)
4. Factual accuracy and completeness (15%)

INSTRUCTIONS:
Return ONLY a number between 0 and 10 using the following ranges:
- 10 = perfectly relevant and accurate
- 9-8 = very relevant with strong detail
- 7-6 = relevant but somewhat incomplete
- 5-4 = partially relevant
- 3-0 = poorly relevant or inaccurate

Respond in {language}.
"""
    
    try:
        response = invoke_ollama(
            system_prompt="You are an expert document evaluator. Provide only numerical scores.",
            user_prompt=prompt,
            model=llm_model
        )
        
        # Extract numerical score from response
        match = re.search(r"\b(\d+(?:\.\d+)?)\b", response)
        if match:
            score = float(match.group(1))
            # Ensure score is within valid range
            return max(0.0, min(10.0, score))
        else:
            st.warning(f"Could not extract score from LLM response: {response[:100]}...")
            return 1.0
    except Exception as e:
        st.error(f"Failed to score summary: {str(e)}")
        return 1.0


def rerank_query_summaries(initial_query: str, query: str, summaries: List[Document], 
                           additional_context: str, llm_model: str, language: str = "English") -> List[Dict]:
    """
    Rerank a list of Document summaries based on relevance & accuracy.
    
    Args:
        initial_query: The original user question.
        query: The specific research query being processed.
        summaries: List of Document objects with page_content.
        additional_context: Conversation history or domain context.
        llm_model: LLM model to use for scoring.
        language: Detected language for the evaluation.
    
    Returns:
        A list of dicts with keys: 'summary', 'score', 'original_index',
        sorted by descending score.
    """
    results = []
    
    with st.spinner(f"Reranking {len(summaries)} summaries for query: {query[:50]}..."):
        for idx, doc in enumerate(summaries):
            content = doc.page_content
            score = score_summary(initial_query, query, content, additional_context, llm_model, language)
            results.append({
                "summary": doc,
                "score": score,
                "original_index": idx,
                "query": query
            })
    
    # Sort highest score first
    return sorted(results, key=lambda x: x["score"], reverse=True)


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
{reranked_summaries[0]['summary'].page_content}

SUPPORTING SOURCES:"""

    # Add remaining summaries as supporting sources
    for i, item in enumerate(reranked_summaries[1:], 2):
        prompt += f"""

Source {i} (Score: {item['score']:.1f}):
{item['summary'].page_content}"""

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


def generate_final_report(initial_query: str, all_reranked_summaries: List[Dict], 
                         additional_context: str, llm_model: str, language: str = "English") -> str:
    """
    Generate the final report using reranked summaries.
    """
    if not all_reranked_summaries:
        return "Error: No summaries available for generating final report."
    
    # Generate the enhanced prompt using reranked summaries
    final_answer_prompt = generate_final_answer_prompt(
        initial_query=initial_query,
        reranked_summaries=all_reranked_summaries,
        additional_context=additional_context,
        language=language
    )
    
    try:
        with st.spinner(f"Generating final report using {llm_model}..."):
            # Generate final answer using the enhanced prompt
            final_answer = invoke_ollama(
                system_prompt=f"You are an expert assistant providing comprehensive answers. Respond in {language}.",
                user_prompt=final_answer_prompt,
                model=llm_model
            )
        
        return final_answer
        
    except Exception as e:
        st.error(f"Exception in final report generation: {str(e)}")
        return f"Error occurred during final report generation: {str(e)}. Check logs for details."


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


def main():
    st.title("📊 Basic Rerank & Reporter")
    st.markdown("*Test reranking and report generation functionality*")
    
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
            index=0,
            help="Choose the LLM model for reranking and report generation"
        )
        
        # Language selection
        language = st.selectbox(
            "Response Language",
            options=["English", "German", "French", "Spanish"],
            index=0
        )
        
        # Use sample data option
        use_sample_data = st.checkbox("Use Sample Data", value=True, help="Load sample documents for testing")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Input Data")
        
        if use_sample_data:
            st.info("Using sample data for demonstration")
            search_summaries = create_sample_documents()
            queries = list(search_summaries.keys())
            initial_query = st.selectbox("Select Initial Query", queries, index=0)
            additional_context = st.text_area(
                "Additional Context", 
                value="User is researching renewable energy for a sustainability report.",
                height=100
            )
        else:
            # Manual input
            initial_query = st.text_input("Initial Query", placeholder="Enter your main question...")
            
            additional_context = st.text_area(
                "Additional Context", 
                placeholder="Enter any additional context or conversation history...",
                height=100
            )
            
            # Query input
            st.subheader("Research Queries")
            num_queries = st.number_input("Number of Queries", min_value=1, max_value=10, value=2)
            
            queries = []
            search_summaries = {}
            
            for i in range(num_queries):
                query = st.text_input(f"Query {i+1}", key=f"query_{i}")
                if query:
                    queries.append(query)
                    
                    # Document input for this query
                    st.write(f"Documents for Query {i+1}:")
                    num_docs = st.number_input(f"Number of Documents for Query {i+1}", min_value=1, max_value=5, value=2, key=f"num_docs_{i}")
                    
                    docs = []
                    for j in range(num_docs):
                        doc_content = st.text_area(f"Document {j+1} Content", key=f"doc_{i}_{j}", height=100)
                        if doc_content:
                            docs.append(Document(
                                page_content=doc_content,
                                metadata={"source": f"doc_{i}_{j}", "query_index": i}
                            ))
                    
                    if docs:
                        search_summaries[query] = docs
    
    with col2:
        st.header("📊 Results")
        
        if st.button("🚀 Process Reranking & Report Generation", type="primary"):
            if not initial_query:
                st.error("Please provide an initial query")
                return
            
            if not search_summaries:
                st.error("Please provide search summaries")
                return
            
            # Process reranking
            st.subheader("🔄 Reranking Results")
            
            all_reranked_summaries = []
            
            # Create tabs for each query
            if len(search_summaries) > 1:
                tabs = st.tabs([f"Query {i+1}" for i in range(len(search_summaries))])
                
                for idx, (query, summaries) in enumerate(search_summaries.items()):
                    with tabs[idx]:
                        st.write(f"**Query:** {query}")
                        
                        # Rerank summaries for this query
                        reranked = rerank_query_summaries(
                            initial_query=initial_query,
                            query=query,
                            summaries=summaries,
                            additional_context=additional_context,
                            llm_model=selected_model,
                            language=language
                        )
                        
                        # Display ranked results
                        for rank, item in enumerate(reranked, 1):
                            with st.expander(f"Rank {rank} - Score: {item['score']:.1f}"):
                                st.write("**Content:**")
                                st.write(item['summary'].page_content)
                                st.write("**Metadata:**")
                                st.json(item['summary'].metadata)
                        
                        all_reranked_summaries.extend(reranked)
            else:
                # Single query
                query, summaries = next(iter(search_summaries.items()))
                st.write(f"**Query:** {query}")
                
                reranked = rerank_query_summaries(
                    initial_query=initial_query,
                    query=query,
                    summaries=summaries,
                    additional_context=additional_context,
                    llm_model=selected_model,
                    language=language
                )
                
                # Display ranked results
                for rank, item in enumerate(reranked, 1):
                    with st.expander(f"Rank {rank} - Score: {item['score']:.1f}"):
                        st.write("**Content:**")
                        st.write(item['summary'].page_content)
                        st.write("**Metadata:**")
                        st.json(item['summary'].metadata)
                
                all_reranked_summaries = reranked
            
            # Sort all summaries by score (highest first)
            all_reranked_summaries.sort(key=lambda x: x['score'], reverse=True)
            
            # Generate final report
            st.subheader("📋 Final Report")
            
            final_report = generate_final_report(
                initial_query=initial_query,
                all_reranked_summaries=all_reranked_summaries,
                additional_context=additional_context,
                llm_model=selected_model,
                language=language
            )
            
            st.markdown("### Generated Report:")
            st.markdown(final_report)
            
            # Summary statistics
            st.subheader("📈 Summary Statistics")
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.metric("Total Summaries", len(all_reranked_summaries))
            
            with col_b:
                avg_score = sum(item['score'] for item in all_reranked_summaries) / len(all_reranked_summaries)
                st.metric("Average Score", f"{avg_score:.2f}")
            
            with col_c:
                highest_score = max(item['score'] for item in all_reranked_summaries)
                st.metric("Highest Score", f"{highest_score:.1f}")


if __name__ == "__main__":
    main()
