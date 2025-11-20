from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from typing import List
from langchain_core.documents import Document


def add_metadata(doc: Document, doc_type: str, file_name: str) -> Document:
    """Add metadata to document."""
    doc.metadata["doc_type"] = doc_type
    doc.metadata["file_name"] = file_name
    return doc


def document_loader() -> List[Document]:
    """
    Load all documents from the data folder.
    Supports PDF files from data/pdf/ and Markdown files from data/markdown/
    """
    documents = []
    
    # Get the project root directory (parent of src/)
    current_dir = Path(__file__).parent.parent
    data_dir = current_dir / "data"
    
    # Load PDF files
    pdf_dir = data_dir / "pdf"
    if pdf_dir.exists():
        print(f"Loading PDF files from {pdf_dir}...")
        for pdf_file in pdf_dir.glob("*.pdf"):
            try:
                loader = PyPDFLoader(str(pdf_file))
                pdf_docs = loader.load()
                for doc in pdf_docs:
                    doc = add_metadata(doc, "pdf", pdf_file.name)
                    documents.append(doc)
                print(f"  ✓ Loaded {len(pdf_docs)} pages from {pdf_file.name}")
            except Exception as e:
                print(f"  ✗ Error loading {pdf_file.name}: {e}")
    
    # Load Markdown files
    markdown_dir = data_dir / "markdown"
    if markdown_dir.exists():
        print(f"Loading Markdown files from {markdown_dir}...")
        text_loader_kwargs = {"encoding": "utf-8"}
        try:
            loader = DirectoryLoader(
                str(markdown_dir),
                glob="**/*.md",
                loader_cls=TextLoader,
                loader_kwargs=text_loader_kwargs,
            )
            md_docs = loader.load()
            for doc in md_docs:
                file_name = Path(doc.metadata.get("source", "unknown")).name
                doc = add_metadata(doc, "markdown", file_name)
                documents.append(doc)
            print(f"  ✓ Loaded {len(md_docs)} markdown files")
        except Exception as e:
            print(f"  ✗ Error loading markdown files: {e}")
    
    # Summary
    print(f"\n📚 Total documents loaded: {len(documents)}")
    doc_types = {}
    for doc in documents:
        doc_type = doc.metadata.get('doc_type', 'unknown')
        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
    print(f"Document types: {dict(doc_types)}")
    
    return documents
