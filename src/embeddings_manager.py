from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from typing import List
from langchain_core.documents import Document
import shutil


def create_embeddings_function():
    """
    Create and return an OpenAI embeddings function.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return embeddings


def create_vectorstore(chunks: List[Document], db_name: str = "vector_db", force_recreate: bool = False):
    """
    Create a Chroma vectorstore from documents.
    
    Args:
        chunks: List of Document objects to embed and store
        db_name: Name/path of the vector database
        force_recreate: If True, delete existing vectorstore before creating new one
    
    Returns:
        Chroma vectorstore instance
    """
    # Get embeddings function
    embeddings = create_embeddings_function()
    
    # Get the project root directory
    current_dir = Path(__file__).parent.parent
    db_path = current_dir / db_name
    
    # Check if vectorstore exists and handle accordingly
    if db_path.exists():
        if force_recreate:
            print(f"🗑️  Deleting existing vectorstore at {db_path}...")
            shutil.rmtree(db_path)
            print("  ✓ Deleted existing vectorstore")
        else:
            print(f"⚠️  Vectorstore already exists at {db_path}")
            print("  Loading existing vectorstore...")
            vectorstore = Chroma(
                persist_directory=str(db_path),
                embedding_function=embeddings
            )
            print(f"  ✓ Loaded vectorstore with {vectorstore._collection.count()} documents")
            return vectorstore
    
    # Create new vectorstore
    print(f"📦 Creating new vectorstore at {db_path}...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(db_path)
    )
    print(f"  ✓ Vectorstore created with {vectorstore._collection.count()} documents")
    
    return vectorstore


def load_vectorstore(db_name: str = "vector_db"):
    """
    Load an existing Chroma vectorstore.
    
    Args:
        db_name: Name/path of the vector database
    
    Returns:
        Chroma vectorstore instance
    """
    embeddings = create_embeddings_function()
    
    # Get the project root directory
    current_dir = Path(__file__).parent.parent
    db_path = current_dir / db_name
    
    if not db_path.exists():
        raise FileNotFoundError(f"Vectorstore not found at {db_path}")
    
    print(f"📂 Loading vectorstore from {db_path}...")
    vectorstore = Chroma(
        persist_directory=str(db_path),
        embedding_function=embeddings
    )
    print(f"  ✓ Loaded vectorstore with {vectorstore._collection.count()} documents")
    
    return vectorstore
