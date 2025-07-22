# RAG Deep Researcher v2.0

## Overview

RAG Deep Researcher is a LangGraph-based application that implements a sophisticated Retrieval-Augmented Generation (RAG) workflow with Human-in-the-Loop (HITL) capabilities. The application is designed to perform deep research on user queries by leveraging vector databases, embedding models, and large language models.

## Architecture

The application consists of two main workflows that run sequentially:

1. **Human-in-the-Loop (HITL) Workflow**: Processes user queries with human feedback to generate high-quality research queries.
2. **Main Research Workflow**: Uses the research queries from the HITL workflow to retrieve documents, summarize information, and generate a comprehensive final answer.

### HITL Workflow

The HITL workflow consists of the following nodes:

1. **analyse_user_feedback**: Analyzes the user query and any feedback provided.
2. **generate_follow_up_questions**: Generates follow-up questions to refine the research direction.
3. **generate_knowledge_base_questions**: Generates final research queries that will be used in the main workflow.

The HITL workflow uses the `InitState` type for state management and produces `research_queries: list[str]` as its main output.

### Main Research Workflow

The main research workflow consists of the following nodes:

1. **retrieve_rag_documents**: Retrieves documents from the vector database based on the research queries.
2. **update_position**: Updates the current position in the research process.
3. **summarize_query_research**: Summarizes the retrieved documents.
4. **generate_final_answer**: Generates the final answer based on the summaries.
5. **quality_checker** (optional): Checks the quality of the final answer and may trigger additional research if needed.

The main workflow uses the `ResearcherStateV2` type for state management.

## Workflow Integration

The integration between the HITL and main workflows is handled by the `execute_sequential_workflow` function, which:

1. Runs the HITL workflow to completion
2. Extracts the research queries and other state information
3. Initializes the main workflow with the HITL results
4. Runs the main workflow starting with the `retrieve_rag_documents` node

This sequential execution ensures that the human feedback is properly incorporated into the research process.

## User Interface

The application uses Streamlit for the user interface, which includes:

- Chat interface for user queries and system responses
- Progress bars for both HITL and main workflows
- Status messages indicating the current step in each workflow
- Debug output showing the generated research queries

## Configuration

The application can be configured with the following options:

- **Embedding Model**: The model used for document embeddings (e.g., `Qwen/Qwen3-Embedding-0.6B`)
- **LLM Models**: Separate models for report generation and summarization
- **Quality Checker**: Can be enabled or disabled
- **External Database**: Option to use an external vector database

## Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Run the application: `streamlit run apps/app_v2_0.py`
3. Enter a query and follow the HITL workflow
4. After completing the HITL phase, the main research workflow will automatically start

## Development

The codebase is organized as follows:

- `apps/`: Contains the main application files
- `src/`: Contains the core functionality modules
  - `graph_v2_1.py`: Defines the HITL and main workflow graphs
  - `state_v2_0.py`: Defines the state types for both workflows
  - `configuration_v1_1.py`: Handles configuration and embedding model management
  - Other utility modules for RAG functionality

## License

See the LICENSE file for details.