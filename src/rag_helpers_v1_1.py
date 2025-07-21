import os, re
from datetime import datetime
from typing import List
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


def source_summarizer_ollama(query, context_documents, language, system_message, llm_model="deepseek-r1"):
    # Make sure language is explicitly passed through the entire pipeline
    print(f"Generating summary using language: {language}")
    print(f"  [DEBUG] Actually using summarization model in source_summarizer_ollama: {llm_model}")
    # Override system_message to ensure language is set properly
    from src.prompts_v1_1 import SUMMARIZER_SYSTEM_PROMPT
    system_message = SUMMARIZER_SYSTEM_PROMPT.format(language=language)
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
    prompt = SUMMARIZER_HUMAN_PROMPT.format(query=query, documents=formatted_context, language=language)
    
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