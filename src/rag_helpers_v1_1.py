import os, re
from datetime import datetime
from typing import List
from pathlib import Path
from langchain_core.runnables import chain
from langchain_core.documents import Document
# Use updated import path to avoid deprecation warning
try:
    from langchain_chroma import Chroma
except ImportError:
    # Fallback to original import if package is not installed
    from langchain_community.vectorstores import Chroma
# Use updated import path to avoid deprecation warning
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    # Fallback to original import if package is not installed
    from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List, Dict, Any
import nltk
from langchain_community.llms import Ollama
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader

# Try to download nltk data if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

from nltk.tokenize import word_tokenize
from src.prompts_v1_1 import (
    # Document summarization prompts
    SUMMARIZER_HUMAN_PROMPT, SUMMARIZER_SYSTEM_PROMPT
)

# Define constants - must match the value in vector_db.py
VECTOR_DB_PATH = "database"

# Define clear_cuda_memory function
def clear_cuda_memory():
    """Clear CUDA memory if available"""
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Define extract_text_from_pdf function
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file"""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Define clean_ function
def clean_(text):
    """Clean text by removing unwanted characters and extra whitespace"""
    # Remove unwanted characters but preserve . , : § $ % &
    text = re.sub(r'[^a-zA-Z0-9\s.,:§$%&€@-µ²³üöäßÄÖÜ]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Define calculate_chunk_ids function
def calculate_chunk_ids(chunks):
    """Calculate human-readable chunk IDs for documents"""
    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get('source', 'unknown')
        page = chunk.metadata.get('page', 0)
        chunk_id = f"{source}:{page}:{i}"
        chunk.metadata['id'] = chunk_id
        # Also store as chunk_id for easier retrieval display
        chunk.metadata['chunk_id'] = i
    return chunks


def get_tenant_collection_name(tenant_id):
    """Get the collection name for a tenant."""
    return f"collection_{tenant_id}"


def get_tenant_vectorstore(tenant_id, embed_llm, persist_directory, similarity, normal=True):
    """Get the vector store for a tenant."""
    # Get tenant-specific directory
    tenant_vdb_dir = os.path.join(persist_directory, tenant_id)
    
    # Create directory if it doesn't exist
    os.makedirs(tenant_vdb_dir, exist_ok=True)
    
    # Get collection name for tenant
    collection_name = get_tenant_collection_name(tenant_id)
    
    return Chroma(
        persist_directory=tenant_vdb_dir,
        collection_name=collection_name,
        embedding_function=embed_llm,
        collection_metadata={"hnsw:space": similarity, "normalize_embeddings": normal}
    )


def load_embed(folder, vdbdir, embed_llm, similarity="cosine", c_size=1000, c_overlap=200, normal=True, clean=True, tenant_id=None):    
    # Clear CUDA memory before starting embedding process
    clear_cuda_memory()
    
    dirname = vdbdir
    # Now load and embed
    print(f"Step: Check for new data and embed new data to new vector DB '{dirname}'")
    # Load documents from the specified directory
    directory = folder
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(directory, filename)
            text = extract_text_from_pdf(pdf_path)
            documents.append(Document(page_content=text, metadata={'source': filename, 'path': pdf_path}))
        else:
            loader = DirectoryLoader(directory, exclude="**/*.pdf")
            loaded = loader.load()
            if loaded:
                # Add full path to metadata
                for doc in loaded:
                    if 'source' in doc.metadata:
                        doc.metadata['path'] = os.path.join(directory, doc.metadata['source'])
                documents.extend(loaded)
    
    docslen = len(documents)
    
    # multitenant
    if tenant_id is None:
        tenant_id = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')    
    print(f"Using tenant ID: {tenant_id}")
    vectorstore = get_tenant_vectorstore(tenant_id, embed_llm, persist_directory=dirname, similarity=similarity, normal=normal)
    print(f"Collection name: {vectorstore._collection.name}")
    print(f"Collection count before adding: {vectorstore._collection.count()}")
    
    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=c_size, chunk_overlap=c_overlap)
    chunks = []
    for document in documents:
        if clean:
            doc_chunks = text_splitter.create_documents([clean_(document.page_content)])
        else:
            doc_chunks = text_splitter.create_documents([document.page_content])
        for chunk in doc_chunks:
            chunk.metadata['source'] = document.metadata['source']
            chunk.metadata['page'] = document.metadata.get('page', 0)  # Assuming page metadata is available
            chunk.metadata['path'] = document.metadata.get('path', '')
        chunks.extend(doc_chunks)

    # Calculate human-readable chunk IDs
    chunks = calculate_chunk_ids(chunks)

    # Extract vector IDs from chunks
    vector_ids = [chunk.metadata['id'] for chunk in chunks]

    # Check for existing vector IDs in the database
    existing_ids = vectorstore.get()['ids']

    # Filter out chunks that are already in the database
    new_chunks = [chunk for chunk, vector_id in zip(chunks, vector_ids) if vector_id not in existing_ids]
    new_vector_ids = [vector_id for vector_id in vector_ids if vector_id not in existing_ids]

    newchunkslen = len(new_chunks)

    if new_chunks:
        # Clear CUDA memory before adding documents
        clear_cuda_memory()
        
        # Add the new chunks to the vector store with their embeddings
        vectorstore.add_documents(new_chunks, ids=new_vector_ids)
        print(f"Collection count after adding: {vectorstore._collection.count()}")
        vectorstore.persist()
        print(f"#{docslen} files embedded via #{newchunkslen} chunks in vector database.")
        
        # Clear CUDA memory after adding documents
        clear_cuda_memory()
    else:
        # Already existing
        print(f"Chunks already available, no new chunks added to vector database.")

    return dirname, tenant_id

def similarity_search_for_tenant(tenant_id, embed_llm, persist_directory, similarity, normal, query, k=2, language="English", collection_name=None):
    """Perform similarity search for a tenant.
    
    Args:
        tenant_id: The tenant ID to search for
        embed_llm: The embedding model to use
        persist_directory: The directory where the vector database is stored
        similarity: The similarity metric to use (e.g., 'cosine')
        normal: Whether to normalize embeddings
        query: The query string to search for
        k: The number of results to return
        language: The language of the query
        collection_name: Optional specific collection name to use. If None, will generate from tenant_id
    """
    # Import necessary modules
    # Define clear_cuda_memory function locally since src.assistant.utils is not available
    def clear_cuda_memory():
        """Clear CUDA memory if available"""
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    import logging
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Debug information
    logger.info(f"Starting similarity search with: tenant_id={tenant_id}, collection_name={collection_name}, query={query}")

    
    # Clear CUDA memory before search
    clear_cuda_memory()
    
    # Get tenant-specific directory
    tenant_vdb_dir = os.path.join(persist_directory, tenant_id)
    logger.info(f"Tenant VDB directory: {tenant_vdb_dir}")
    
    # Check if directory exists
    if not os.path.exists(tenant_vdb_dir):
        error_msg = f"Vector database directory for tenant {tenant_id} does not exist at {tenant_vdb_dir}"
        logger.error(error_msg)
        raise Exception(error_msg)
    
    # Get collection name for tenant if not provided
    if collection_name is None:
        collection_name = get_tenant_collection_name(tenant_id)
    logger.info(f"Using collection name: {collection_name}")
    
    # Initialize vectorstore for search
    logger.info(f"Initializing Chroma with: dir={tenant_vdb_dir}, collection={collection_name}")
    
    try:
        # Try to use direct Chroma client first to validate collection exists
        from chromadb import PersistentClient
        client = PersistentClient(path=tenant_vdb_dir)
        collections = client.list_collections()
        logger.info(f"Available collections in {tenant_vdb_dir}: {collections}")
        
        # Extract collection names from Collection objects for comparison
        collection_names = [coll.name for coll in collections]
        
        if collection_name not in collection_names:
            logger.warning(f"Collection '{collection_name}' not found in available collections: {collections}")
            if collections:  # If there are any collections available
                logger.info(f"Trying with first available collection: {collections[0]}")
                collection_name = collections[0].name  # Extract name from Collection object
            else:
                logger.error(f"No collections found in {tenant_vdb_dir}")
                return []  # Return empty results if no collections available
    except Exception as e:
        logger.error(f"Error checking collections: {str(e)}")
    
    # Now initialize vectorstore with validated collection name
    vectorstore = Chroma(
        persist_directory=tenant_vdb_dir,
        collection_name=collection_name,
        embedding_function=embed_llm,
        collection_metadata={"hnsw:space": similarity, "normalize_embeddings": normal}
    )
    
    try:
        # Print language being used for retrieval
        logger.info(f"Using language for retrieval: {language}")
        
        # Perform similarity search
        logger.info(f"Executing similarity_search with query: '{query}' and k={k}")
        results = vectorstore.similarity_search(query, k=k)
        logger.info(f"Retrieved {len(results)} documents from search")
        
        if not results:
            logger.warning("No documents found in similarity search.")
        
        # Add language metadata to each document for downstream processing
        for doc in results:
            if "metadata" in doc.__dict__:
                doc.metadata["language"] = language
        
        # Clean up
        vectorstore._client = None
        del vectorstore
        
        # Clear CUDA memory after search
        clear_cuda_memory()
        
        return results
    except Exception as e:
        # Clean up in case of error
        if 'vectorstore' in locals():
            vectorstore._client = None
            del vectorstore
        
        # Clear CUDA memory in case of error
        clear_cuda_memory()
        
        # Re-raise the exception
        raise e


def transform_documents(documents: List[Document]) -> List[Dict[str, Any]]:
    """
    Transforms a list of Document objects into a specific dictionary format for the simplified workflow.
    
    Args:
        documents (list): List of Document objects with metadata and page_content
        
    Returns:
        list: List of dictionaries with content and metadata in the required format
    """
    transformed_docs = []
    
    for doc in documents:
        transformed_doc = {
            "content": doc.page_content,
            "metadata": {}
        }
        
        # Copy metadata if available
        if hasattr(doc, "metadata") and doc.metadata:
            for key, value in doc.metadata.items():
                transformed_doc["metadata"][key] = value
        
        transformed_docs.append(transformed_doc)
    
    return transformed_docs


def format_documents_as_plain_text(documents):
    """
    Format LangChain Documents into a plain text representation with ID, source, and content information.
    
    Args:
        documents (list): List of LangChain Document objects to format
        
    Returns:
        str: A formatted string with document information in plain text format
    """
    if not documents:
        return "No documents found."
    
    formatted_docs = []
    for i, doc in enumerate(documents):
        # Extract document ID, source, and content
        doc_id = doc.metadata.get('id', f'Unknown-ID-{i}')
        doc_source = doc.metadata.get('source', 'Unknown source')
        doc_content = doc.page_content
        
        # Format the document information
        formatted_doc = f"Document{i+1}:\nID is: {doc_id},\nSOURCE is: {doc_source},\nCONTENT is: {doc_content}\n"
        formatted_docs.append(formatted_doc)
    
    return "\n".join(formatted_docs)


def source_summarizer_ollama(user_query, context_documents, language, system_message, llm_model="deepseek-r1", human_feedback=""):
    # Make sure language is explicitly passed through the entire pipeline
    print(f"Generating summary using language: {language}")
    print(f"  [DEBUG] Actually using summarization model in source_summarizer_ollama: {llm_model}")
    
    # Robust language handling - ensure language is a string and has a valid value
    if not language or not isinstance(language, str):
        language = "English"
        print(f"  [WARNING] Invalid language parameter in source_summarizer_ollama, defaulting to {language}")
    
    # Override system_message to ensure language is set properly
    from src.prompts_v1_1 import SUMMARIZER_SYSTEM_PROMPT
    try:
        system_message = SUMMARIZER_SYSTEM_PROMPT.format(language=language)
        print(f"  [DEBUG] Successfully formatted system prompt with language: {language}")
    except Exception as e:
        print(f"  [ERROR] Error formatting system prompt with language '{language}': {str(e)}")
        # Try fallback to English if formatting fails
        language = "English"
        system_message = SUMMARIZER_SYSTEM_PROMPT.format(language=language)
        print(f"  [DEBUG] Using fallback language: {language}")

    # Check if context_documents is already a formatted string
    if isinstance(context_documents, str):
        formatted_context = context_documents
    else:
        # Handle the case where context_documents is a list of dictionary objects
        try:
            formatted_context = "\n".join(
                f"Content: {doc['content']}\nSource: {doc['metadata']['name']}\nPath: {doc['metadata']['path']}"
                for doc in context_documents
            )
        except (TypeError, KeyError):
            # Fallback: try to use the documents as they are
            formatted_context = str(context_documents)
    #formatted_context = "\n".join(
    #    f"{str(doc)}"
    #    for doc in context_documents
    #)
    prompt = SUMMARIZER_HUMAN_PROMPT.format(
        user_query=user_query,
        documents=formatted_context,
        human_feedback=human_feedback,
        language=language
    )
    
    # Initialize ChatOllama with the specified model and temperature
    llm = Ollama(model=llm_model, temperature=0.1, repeat_penalty=1.2) 
    # For RAG systems like your summarizer, consider:
    #    Using lower temperatures (0.1-0.3) for factual accuracy
    #   Combining with repeat_penalty=1.1-1.3 to avoid redundant content
    #   Monitoring token usage with num_ctx for long documents
    
    # Format messages for LangChain
    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=prompt)
    ]
    
    # Get response from the model
    response = llm.invoke(messages)
    
    # Extract content from response
    response_content = response
    
    # Clean markdown formatting if present
    try:
        final_content = re.sub(r"<think>.*?</think>", "", response_content, flags=re.DOTALL).strip()
    except:
        final_content = response_content.strip()

    # Extract metadata from all documents with added checks for structure
    document_names = []
    for doc in context_documents:
        if isinstance(doc, dict) and 'metadata' in doc and isinstance(doc['metadata'], dict):
            # Try to get name from metadata, with fallbacks to source or id if name doesn't exist
            if 'name' in doc['metadata']:
                document_names.append(doc['metadata']['name'])
            elif 'source' in doc['metadata']:
                document_names.append(doc['metadata']['source'])
            elif 'id' in doc['metadata']:
                # Extract filename from id if it contains a path
                doc_id = doc['metadata']['id']
                if ':' in doc_id:
                    doc_id = doc_id.split(':', 1)[0]  # Get the part before the first colon
                document_names.append(doc_id)
            else:
                # Use a default name if no identifiers are available
                document_names.append(f"Document-{len(document_names)+1}")
    
    document_paths = []
    for doc in context_documents:
        if isinstance(doc, dict) and 'metadata' in doc and isinstance(doc['metadata'], dict):
            # Try to get path from metadata, with fallback to source if path doesn't exist
            if 'path' in doc['metadata']:
                document_paths.append(doc['metadata']['path'])
            elif 'source' in doc['metadata']:
                document_paths.append(doc['metadata']['source'])
            else:
                # Use a default path if no path information is available
                document_paths.append("Unknown path")

    return {
        "content": final_content,
        "metadata": {
            "name": document_names,
            "path": document_paths
        }
    }   


def format_content_with_sources(content, source_filenames, source_paths):
    """
    Format content with source information in the format [Content][Source_filename][Source_path]
    
    Args:
        content (str): The main content text
        source_filenames (list or str): List of source filenames or a comma-separated string
        source_paths (list or str): List of source paths or a comma-separated string
        
    Returns:
        str: Formatted content in the format [Content][Source_filename][Source_path]
    """
    # Process source filenames
    if isinstance(source_filenames, list):
        source_filenames_str = ', '.join(source_filenames)
    else:
        source_filenames_str = source_filenames
        
    # Process source paths
    if isinstance(source_paths, list):
        source_paths_str = ', '.join(source_paths)
    else:
        source_paths_str = source_paths
    
    # Create the formatted string
    formatted_content = f"[{content}][{source_filenames_str}][{source_paths_str}]"
    
    return formatted_content


def load_models_from_file(file_path: str) -> List[str]:
    """
    Load model names from a markdown file.
    
    Args:
        file_path (str): Path to the markdown file containing model names
        
    Returns:
        List[str]: List of model names, one per line
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found: {file_path}")
    
    models = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):  # Skip empty lines and comments
                models.append(line)
    
    return models


def get_report_llm_models() -> List[str]:
    """
    Get the list of available report LLM models from the global configuration.
    
    Returns:
        List[str]: List of report LLM model names
    """
    # Get the path to the report_llms.md file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    report_llms_path = os.path.join(current_dir, 'report_llms.md')
    
    try:
        return load_models_from_file(report_llms_path)
    except FileNotFoundError:
        # Fallback to default models if file not found
        return [
            "deepseek-r1:latest",
            "deepseek-r1:70b", 
            "qwen3:latest",
            "mistral-small3.2:latest"
        ]


def get_summarization_llm_models() -> List[str]:
    """
    Get the list of available summarization LLM models from the global configuration.
    
    Returns:
        List[str]: List of summarization LLM model names
    """
    # Get the path to the summarization_llms.md file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    summarization_llms_path = os.path.join(current_dir, 'summarization_llms.md')
    
    try:
        return load_models_from_file(summarization_llms_path)
    except FileNotFoundError:
        # Fallback to default models if file not found
        return [
            "deepseek-r1:latest",
            "qwen3:latest",
            "mistral-small3.2:latest"
        ]


def get_all_available_models() -> List[str]:
    """
    Get a combined list of all available models from both report and summarization files.
    
    Returns:
        List[str]: Combined list of unique model names
    """
    report_models = get_report_llm_models()
    summarization_models = get_summarization_llm_models()
    
    # Combine and deduplicate while preserving order
    all_models = []
    seen = set()
    
    for model in report_models + summarization_models:
        if model not in seen:
            all_models.append(model)
            seen.add(model)
    
    return all_models


def get_license_content() -> str:
    """
    Get the content of the LICENSE file for display in the applications.
    
    Returns:
        str: The content of the LICENSE file
    """
    # Get the path to the LICENSE file (one level up from src)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    license_path = os.path.join(os.path.dirname(current_dir), 'LICENCE')
    
    try:
        with open(license_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        return "Apache License 2.0 - License file not found"


def parse_document_to_formatted_content(document_text):
    """
    Parse a document text that contains Content, Source_filename, and Source_path
    and format it as [Content][Source_filename][Source_path]
    
    Args:
        document_text (str): The document text containing Content, Source_filename, and Source_path sections
        
    Returns:
        str: Formatted content in the format [Content][Source_filename][Source_path]
    """
    content = ""
    source_filenames = ""
    source_paths = ""
    
    # Extract content
    content_start = document_text.find("Content:")
    if content_start != -1:
        content_start += len("Content:")
        source_filename_start = document_text.find("Source_filename:", content_start)
        if source_filename_start != -1:
            content = document_text[content_start:source_filename_start].strip()
        else:
            content = document_text[content_start:].strip()
    
    # Extract source filenames
    if "Source_filename:" in document_text:
        source_filename_start = document_text.find("Source_filename:")
        source_filename_start += len("Source_filename:")
        source_path_start = document_text.find("Source_path:", source_filename_start)
        if source_path_start != -1:
            source_filenames = document_text[source_filename_start:source_path_start].strip()
        else:
            source_filenames = document_text[source_filename_start:].strip()
    
    # Extract source paths
    if "Source_path:" in document_text:
        source_path_start = document_text.find("Source_path:")
        source_path_start += len("Source_path:")
        source_paths = document_text[source_path_start:].strip()
    
    # Format the content
    return format_content_with_sources(content, source_filenames, source_paths)

# Function to extract embedding model name from database directory
def extract_embedding_model(db_dir_name):
    """
    Extract the embedding model name from the database directory name.
    
    This function properly handles various database naming conventions, including:
    - Standard format: "organization/model_name"
    - Directory format with separators: "Qwen/StrlSch__Qwen--Qwen3-Embedding-0.6B--3000--600"
    - Legacy format: "model_name/chunk_size/overlap"
    
    Args:
        db_dir_name (str): The database directory name (e.g., "Qwen/StrlSch__Qwen--Qwen3-Embedding-0.6B--3000--600")
        
    Returns:
        str: The extracted embedding model name (e.g., "Qwen/Qwen3-Embedding-0.6B")
    """
    # Handle the specific case of database names with '__' and '--' separators
    # Example: "Qwen/StrlSch__Qwen--Qwen3-Embedding-0.6B--3000--600" -> "Qwen/Qwen3-Embedding-0.6B"
    
    # First handle __ separator if present
    if '__' in db_dir_name:
        # Split by __ to separate the path prefix from the model info
        # "StrlSch__Qwen--Qwen3-Embedding-0.6B--3000--600" -> ["StrlSch", "Qwen--Qwen3-Embedding-0.6B--3000--600"]
        parts = db_dir_name.split('__')
        if len(parts) >= 2:
            # The model info is in the second part after __
            model_info = parts[1]   # "Qwen--Qwen3-Embedding-0.6B--3000--600"
            
            # Now handle the model_info part with '--' separators
            if '--' in model_info:
                model_parts = model_info.split('--')
                if len(model_parts) >= 2:
                    # model_parts[0] is org ("Qwen"), model_parts[1] is model name ("Qwen3-Embedding-0.6B")
                    org = model_parts[0]  # "Qwen"
                    model_name = model_parts[1]  # "Qwen3-Embedding-0.6B"
                    result = f"{org}/{model_name}"  # "Qwen/Qwen3-Embedding-0.6B"
                    return result
            
            # Fallback: if no '--' in model_info, treat the whole model_info as model name
            # and try to extract org from the path prefix
            path_prefix = parts[0]  # "StrlSch"
            if '/' in path_prefix:
                org = path_prefix.split('/')[0]
            else:
                org = "unknown"
            return f"{org}/{model_info}"
    
    # Handle the case with only '--' separators (no '__')
    elif '--' in db_dir_name:
        parts = db_dir_name.split('--')
        
        if len(parts) >= 2:
            # The first part should contain the model organization and name
            first_part = parts[0]  # "Qwen"
            second_part = parts[1]  # "Qwen3-Embedding-0.6B"
            
            if '/' in first_part:
                # Extract organization from first part
                org = first_part.split('/')[0]  # "Qwen"
                result = f"{org}/{second_part}"  # "Qwen/Qwen3-Embedding-0.6B"
                return result
            else:
                # Fallback: use first part as org
                result = f"{first_part}/{second_part}"
                return result
    
    # Fallback to original logic if the new parsing fails
    model_name = db_dir_name.replace("vectordb_", "")
    model_name = model_name.replace("--", "/")

    return model_name


# Source handling functions for clickable PDF sources
def get_available_databases(database_path="./kb/database"):
    """
    Get list of available databases from the database directory.
    
    Args:
        database_path (str): Path to the database directory
        
    Returns:
        List of database directory names
    """
    from pathlib import Path
    
    database_path = Path(database_path)
    if not database_path.exists():
        return []
    
    return [d.name for d in database_path.iterdir() if d.is_dir()]


def extract_database_prefix(database_name: str) -> str:
    """
    Extract the prefix from a database name to find corresponding source directory.
    
    Args:
        database_name: Database name like "StrlSch__Qwen--Qwen3-Embedding-0.6B--3000--600"
        
    Returns:
        Prefix like "StrlSch"
        
    Examples:
        "StrlSch__Qwen--Qwen3-Embedding-0.6B--3000--600" -> "StrlSch"
        "NORM__Qwen--Qwen3-Embedding-0.6B--3000--600" -> "NORM"
    """
    # Handle None or empty database names
    if not database_name:
        return ""
    
    # Split by double underscore and take the first part
    parts = database_name.split("__")
    return parts[0] if parts else database_name


def resolve_source_directory(database_name: str, kb_path="./kb") -> Path:
    """
    Resolve database name to corresponding source directory.
    
    Args:
        database_name: Database name like "StrlSch__Qwen--Qwen3-Embedding-0.6B--3000--600"
        kb_path: Root KB directory path
        
    Returns:
        Path to source directory like "./kb/StrlSch__db_inserted"
    """
    from pathlib import Path
    
    prefix = extract_database_prefix(database_name)
    kb_path = Path(kb_path)
    
    # If no database name provided, search all subdirectories in kb_path
    if not prefix:
        # Return kb_path itself for broad search
        return kb_path
    
    # Try different possible patterns for source directories
    possible_patterns = [
        f"{prefix}__db_inserted",  # Standard pattern: StrlSch__db_inserted
        f"{prefix}_db_inserted",   # Alternative: StrlSch_db_inserted  
        f"{prefix}__inserted",     # Alternative: StrlSch__inserted
        f"{prefix}_inserted",      # Alternative: StrlSch_inserted
        prefix                     # Just the prefix: StrlSch
    ]
    
    for pattern in possible_patterns:
        source_path = kb_path / pattern
        if source_path.exists() and source_path.is_dir():
            return source_path
    
    # If no match found, return the most likely path (even if it doesn't exist)
    return kb_path / f"{prefix}__db_inserted"


def resolve_pdf_path(source_name: str, selected_database: str = None, kb_path="./kb") -> Path:
    """
    Resolve a source name to an actual PDF file path based on selected database.
    
    Args:
        source_name: Source reference like "StrlSchG--250508.pdf"
        selected_database: Database name like "StrlSch__Qwen--Qwen3-Embedding-0.6B--3000--600"
        kb_path: Root KB directory path
        
    Returns:
        Path to the actual PDF file
        
    Example:
        "StrlSchG--250508.pdf", "StrlSch__Qwen--Qwen3-Embedding-0.6B--3000--600" 
        -> "./kb/StrlSch__db_inserted/StrlSchG.pdf"
    """
    import re
    from pathlib import Path
    
    # Extract the base filename (remove timestamp/suffix if present)
    # Pattern: "StrlSchG--250508.pdf" -> "StrlSchG.pdf"
    base_name = re.sub(r'--\d+', '', source_name)
    
    # Determine the source directory based on selected database
    if selected_database:
        pdf_root = resolve_source_directory(selected_database, kb_path)
    else:
        # Fallback to default directory if no database selected
        pdf_root = Path(kb_path) / "StrlSch__db_inserted"
    
    # Try to find the file in the PDF directory
    pdf_path = (pdf_root / base_name).resolve()
    
    # If exact match doesn't exist, try to find similar files
    if not pdf_path.exists() and pdf_root.exists():
        # Look for files that start with the same base name (without extension)
        base_without_ext = Path(base_name).stem
        for pdf_file in pdf_root.glob("*.pdf"):
            if pdf_file.stem.lower() == base_without_ext.lower():
                return pdf_file.resolve()
    
    return pdf_path


def linkify_sources(markdown_text: str, selected_database: str = None, kb_path="./kb") -> str:
    """
    Convert source references in markdown to clickable links that open PDFs in new windows.
    
    Args:
        markdown_text: Text containing source references like [StrlSchG--250508.pdf] or [EPA_Kd-a]
        selected_database: Database name to determine source directory
        kb_path: Root KB directory path
        
    Returns:
        HTML text with clickable links that open in new windows
    """
    import base64
    import re
    from pathlib import Path
    
    # Pattern to match source references with BOTH square brackets and parentheses:
    # [filename.pdf], [filename--timestamp.pdf], (filename.pdf), (filename--timestamp.pdf)
    # CRITICAL: Only match references that look like filenames, not mathematical notation or other content
    # Matches:
    #   - Must start with a letter
    #   - Can contain letters, numbers, underscores, hyphens, dots
    #   - Minimum 3 characters total
    #   - Supports both [...] and (...) delimiters
    # Excludes: pure numbers, mathematical symbols (^, {, }, \, etc.), LaTeX notation
    source_pattern = re.compile(r'[\[\(]([A-Za-z][A-Za-z0-9_\-\.]{2,}(?:\.pdf)?)[\]\)]')
    
    def replace_with_link(match):
        source_ref = match.group(1)
        original_brackets = match.group(0)[0] + match.group(0)[-1]  # Store original brackets
        
        # Skip URL references - don't process links that contain http:// or https://
        if source_ref.startswith(('http://', 'https://', 'www.')):
            return match.group(0)  # Return original [URL] unchanged
        
        # Skip if it looks like a URL without protocol (contains common URL patterns)
        if any(pattern in source_ref.lower() for pattern in ['.com', '.org', '.net', '.edu', '.gov', '.de', '.uk']):
            return match.group(0)  # Return original [URL] unchanged
        
        # Skip if it's too short or looks like mathematical notation
        if len(source_ref) < 3 or source_ref.isdigit():
            return match.group(0)  # Return original unchanged
        
        # Handle all possible reference formats:
        # [StrlSchG], [StrlSchG.pdf], [StrlSchG--250508], [StrlSchG--250508.pdf]
        
        # First, try to resolve as a PDF reference (handles timestamps automatically)
        pdf_path = None
        
        # Case 1: Contains .pdf (e.g., [StrlSchG.pdf] or [StrlSchG--250508.pdf])
        if '.pdf' in source_ref:
            pdf_path = resolve_pdf_path(source_ref, selected_database, kb_path)
        
        # Case 2: No .pdf but might be a base filename (e.g., [StrlSchG] or [StrlSchG--250508])
        if not pdf_path or not pdf_path.exists():
            # Try adding .pdf to the reference
            pdf_ref_with_ext = source_ref + '.pdf' if not source_ref.endswith('.pdf') else source_ref
            pdf_path = resolve_pdf_path(pdf_ref_with_ext, selected_database, kb_path)
        
        # Case 3: Still not found, try fuzzy matching
        if not pdf_path or not pdf_path.exists():
            pdf_path = find_matching_pdf(source_ref, selected_database, kb_path)
        
        if pdf_path and pdf_path.exists():
            # Use base64 data URL for browser compatibility
            # Streamlit strips JavaScript, so we use a direct data URL in href
            try:
                import base64
                
                # Read and encode the PDF
                pdf_bytes = pdf_path.read_bytes()
                b64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
                
                # Create data URL with base64-encoded PDF
                data_url = f"data:application/pdf;base64,{b64_pdf}"
                
                # Return HTML link with target="_blank"
                # Note: This works in Streamlit with unsafe_allow_html=True
                display_name = pdf_path.name if source_ref.endswith('.pdf') else f"{source_ref}"
                
                return f'<a href="{data_url}" target="_blank" rel="noopener noreferrer" style="color: #1f77b4; text-decoration: underline; cursor: pointer;">📄 {display_name}</a>'
            except Exception as e:
                return f'<span style="color: red;">📄 {source_ref} (Error: {str(e)})</span>'
        else:
            source_dir = resolve_source_directory(selected_database, kb_path) if selected_database else "default directory"
            return f'<span style="color: orange;">📄 {source_ref} (Not found in {source_dir})</span>'
    
    return source_pattern.sub(replace_with_link, markdown_text)


def find_matching_pdf(short_ref: str, selected_database: str = None, kb_path="./kb") -> Path:
    """
    Find a PDF file that matches a reference like 'StrlSchG', 'EPA_Kd-a', etc.
    
    Args:
        short_ref: Reference like 'StrlSchG', 'StrlSchG--250508', 'EPA_Kd-a'
        selected_database: Database name to determine source directory
        kb_path: Root KB directory path
        
    Returns:
        Path to matching PDF file or None if not found
    """
    from pathlib import Path
    import re
    
    # Get the source directory
    source_dir = resolve_source_directory(selected_database, kb_path)
    
    if not source_dir.exists():
        return None
    
    # Extract base name (remove timestamp if present)
    # e.g., "StrlSchG--250508" -> "StrlSchG"
    base_ref = re.sub(r'--\d+', '', short_ref)
    
    # Try different matching strategies in order of preference
    patterns_to_try = [
        f"{short_ref}.pdf",           # Exact match with .pdf (e.g., StrlSchG.pdf)
        f"{base_ref}.pdf",            # Base name with .pdf (e.g., StrlSchG.pdf from StrlSchG--250508)
        f"{short_ref}*.pdf",          # Starts with short_ref (e.g., StrlSchG*.pdf)
        f"{base_ref}*.pdf",           # Starts with base_ref (e.g., StrlSchG*.pdf)
        f"*{short_ref}*.pdf",         # Contains short_ref
        f"*{base_ref}*.pdf",          # Contains base_ref
        f"*{short_ref.replace('_', '-')}*.pdf",  # Replace underscores with hyphens
        f"*{short_ref.replace('-', '_')}*.pdf",  # Replace hyphens with underscores
        f"*{base_ref.replace('_', '-')}*.pdf",   # Base ref with underscore/hyphen replacement
        f"*{base_ref.replace('-', '_')}*.pdf",   # Base ref with hyphen/underscore replacement
    ]
    
    for pattern in patterns_to_try:
        matches = list(source_dir.glob(pattern))
        if matches:
            # Return the first match (could be improved with better scoring)
            return matches[0]
    
    # If no direct matches, try case-insensitive search
    search_terms = [short_ref.lower(), base_ref.lower()]
    for pdf_file in source_dir.glob("*.pdf"):
        pdf_name_lower = pdf_file.name.lower()
        for term in search_terms:
            if term in pdf_name_lower:
                return pdf_file
    
    return None


def _deep_search_dict(data, target_keys):
    """
    Recursively search through nested dicts/lists to find values for target keys.
    Returns the deepest/most specific match found.
    
    Args:
        data: Dict, list, or other data structure to search
        target_keys: List of key names to search for (e.g., ['thinking', 'think'])
        
    Returns:
        Found value or None
    """
    if isinstance(data, dict):
        # First check current level
        for key in target_keys:
            if key in data and data[key]:
                # If the value itself is a dict/list with the same keys, recurse deeper
                value = data[key]
                if isinstance(value, (dict, list)):
                    deeper = _deep_search_dict(value, target_keys)
                    if deeper is not None:
                        return deeper
                return value
        
        # Then search all nested values
        for value in data.values():
            if isinstance(value, (dict, list)):
                result = _deep_search_dict(value, target_keys)
                if result is not None:
                    return result
    
    elif isinstance(data, list):
        # Search through list items
        for item in data:
            if isinstance(item, (dict, list)):
                result = _deep_search_dict(item, target_keys)
                if result is not None:
                    return result
    
    return None


def parse_structured_llm_output(final_answer):
    """
    Parse structured LLM output that contains thinking and final answer parts.
    Handles various formats including JSON strings, Python dict strings, and direct dicts.
    Enhanced to RECURSIVELY search through deeply nested structures to find thinking/final content.
    
    Args:
        final_answer: String, dict, or other format containing the structured output
        
    Returns:
        tuple: (final_content, thinking_content) where thinking_content can be None
    """
    if not final_answer:
        return "No final answer available.", None
    
    print(f"[DEBUG] Processing final_answer type: {type(final_answer)}")
    print(f"[DEBUG] Final answer content preview: {str(final_answer)[:200]}...")
    
    thinking_content = None
    final_content = final_answer
    
    # Handle string input that might be JSON or Python dict
    if isinstance(final_answer, str):
        # Check if it looks like a structured format
        if final_answer.strip().startswith('{') and ('}' in final_answer):
            try:
                import json
                import re
                
                # Clean up the JSON string - handle potential formatting issues
                cleaned_json = final_answer.strip()
                
                # Handle cases where there might be multiple JSON objects or nested structures
                # Try to extract the main JSON object
                if cleaned_json.count('{') > 1:
                    # Find the main JSON structure
                    brace_count = 0
                    start_idx = cleaned_json.find('{')
                    end_idx = start_idx
                    
                    for i, char in enumerate(cleaned_json[start_idx:], start_idx):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_idx = i + 1
                                break
                    
                    if end_idx > start_idx:
                        cleaned_json = cleaned_json[start_idx:end_idx]
                
                # Try JSON parsing first
                parsed = json.loads(cleaned_json)
                if isinstance(parsed, dict):
                    final_answer = parsed
                    print(f"[DEBUG] Successfully parsed JSON dict with keys: {list(parsed.keys())}")
                    
            except (json.JSONDecodeError, ValueError) as json_error:
                print(f"[DEBUG] JSON parsing failed: {json_error}")
                try:
                    # Try ast.literal_eval for Python dict strings
                    import ast
                    parsed = ast.literal_eval(final_answer)
                    if isinstance(parsed, dict):
                        final_answer = parsed
                        print(f"[DEBUG] Successfully parsed Python dict string")
                except (ValueError, SyntaxError) as e:
                    print(f"[DEBUG] Failed to parse dict string: {e}")
                    # Try to extract JSON from within the string using regex
                    json_match = re.search(r'\{[^{}]*"thinking"[^{}]*"final"[^{}]*\}', final_answer, re.DOTALL)
                    if json_match:
                        try:
                            parsed = json.loads(json_match.group())
                            if isinstance(parsed, dict):
                                final_answer = parsed
                                print(f"[DEBUG] Successfully extracted and parsed JSON from text")
                        except:
                            print(f"[DEBUG] Failed to parse extracted JSON, using original string")
                            final_content = final_answer
                    else:
                        # Return original string if all parsing fails
                        final_content = final_answer
        else:
            # Regular string, apply cleanup and return
            final_content = final_answer
    
    # Handle dictionary input - use RECURSIVE deep search
    if isinstance(final_answer, dict):
        print(f"[DEBUG] Processing dictionary with keys: {list(final_answer.keys())}")
        
        # Define possible key patterns for thinking and final content
        thinking_keys = ['thinking', 'think', 'thought', 'reasoning', 'analysis', 'process']
        final_keys = ['final', 'answer', 'content', 'response', 'result', 'report']
        
        # RECURSIVELY search for thinking content at any depth
        thinking_content = _deep_search_dict(final_answer, thinking_keys)
        if thinking_content:
            print(f"[DEBUG] Found thinking content through deep search")
        
        # RECURSIVELY search for final content at any depth
        final_content_from_search = _deep_search_dict(final_answer, final_keys)
        if final_content_from_search:
            final_content = final_content_from_search
            print(f"[DEBUG] Found final content through deep search")
        else:
            # If no specific keys found, use the whole dict as final content
            final_content = str(final_answer)
            print(f"[DEBUG] Using entire dict as final content")
    
    # Clean up thinking content - don't show if it's too short or just placeholder text
    if isinstance(thinking_content, str):
        thinking_content = thinking_content.strip()
        if len(thinking_content) < 10 or thinking_content.lower() in ["none", "null", "n/a", ""]:
            thinking_content = None
    
    # Clean up final content - remove any remaining <think> blocks
    if isinstance(final_content, str):
        import re
        # Remove <think>...</think> blocks from final content
        final_content = re.sub(r'<think>.*?</think>', '', final_content, flags=re.DOTALL)
        # Remove malformed <think> tags
        final_content = re.sub(r'<think>.*', '', final_content, flags=re.DOTALL)
        final_content = final_content.strip()
    
    print(f"[DEBUG] Final result - thinking: {thinking_content is not None}, content length: {len(str(final_content))}")
    return final_content, thinking_content