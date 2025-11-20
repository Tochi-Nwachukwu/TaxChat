from src.document_loader import document_loader
from src.agentic_chunker import AgenticChunker
from src.embeddings_manager import create_vectorstore
from src.retrieval_system import create_rag_chatbot, interactive_chat
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from constants import db_name


def extract_propositions_from_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Split documents into smaller propositions/sentences for agentic chunking.
    """
    print("\n📝 Extracting propositions from documents...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    propositions = []
    for doc in documents:
        # Split the document into smaller chunks/propositions
        splits = text_splitter.split_text(doc.page_content)
        
        # Add metadata to track source
        for split in splits:
            if split.strip():  # Only add non-empty splits
                propositions.append({
                    'text': split.strip(),
                    'source': doc.metadata.get('file_name', 'unknown'),
                    'doc_type': doc.metadata.get('doc_type', 'unknown')
                })
    
    print(f"  ✓ Extracted {len(propositions)} propositions")
    return propositions


if __name__ == "__main__":
    print("🚀 Starting RAG Pipeline...\n")
    
    # Step 1: Load documents
    print("=" * 60)
    print("STEP 1: Loading Documents")
    print("=" * 60)
    documents = document_loader()
    
    if not documents:
        print("\n⚠️  No documents were loaded!")
        exit(1)
    
    print(f"\n✅ Successfully loaded {len(documents)} documents!")
    
    # Step 2: Extract propositions
    print("\n" + "=" * 60)
    print("STEP 2: Extracting Propositions")
    print("=" * 60)
    propositions = extract_propositions_from_documents(documents)
    
    # Step 3: Agentic Chunking (using first 10 propositions as demo)
    print("\n" + "=" * 60)
    print("STEP 3: Agentic Chunking")
    print("=" * 60)
    print(f"⚠️  Processing first 10 propositions as demo (full dataset has {len(propositions)} propositions)")
    print("💡 Tip: Remove the limit to process all propositions\n")
    
    ac = AgenticChunker()
    
    # Process propositions (limited to 10 for demo - remove [:10] to process all)
    proposition_texts = [p['text'] for p in propositions]
    ac.add_propositions(proposition_texts)
    
    # Step 4: Display results
    print("\n" + "=" * 60)
    print("STEP 4: Chunk Results")
    print("=" * 60)
    ac.pretty_print_chunks()
    
    # Step 5: Convert chunks to Documents for embedding
    print("\n" + "=" * 60)
    print("STEP 5: Creating Vector Embeddings")
    print("=" * 60)
    
    # Get chunks as list of strings and convert to Document objects
    chunk_strings = ac.get_chunks(get_type='list_of_strings')
    chunk_documents = []
    
    for chunk_id, chunk_data in ac.chunks.items():
        doc = Document(
            page_content=" ".join(chunk_data['propositions']),
            metadata={
                'chunk_id': chunk_id,
                'title': chunk_data['title'],
                'summary': chunk_data['summary'],
                'chunk_index': chunk_data['chunk_index']
            }
        )
        chunk_documents.append(doc)
    
    print(f"  Converting {len(chunk_documents)} chunks to embeddings...")
    
    # Create vectorstore
    vectorstore = create_vectorstore(chunk_documents, db_name=db_name, force_recreate=True)
    
    print("\n🎉 Pipeline completed successfully!")
    print(f"\n📊 Summary:")
    print(f"  - Documents loaded: {len(documents)}")
    print(f"  - Propositions extracted: {len(propositions)}")
    print(f"  - Chunks created: {len(chunk_documents)}")
    print(f"  - Vectorstore location: {db_name}")
    
    # Step 6: Test Retrieval System
    print("\n" + "=" * 60)
    print("STEP 6: Testing RAG Retrieval System")
    print("=" * 60)
    
    # Create chatbot
    chatbot = create_rag_chatbot(vectorstore)
    
    # Test with a sample question
    test_question = "If I earn 60,000 naira monthly, how much Personal income tax would I have to pay?"
    print(f"\n🤔 Test Question: {test_question}")
    print("\n🤖 Answer:")
    result = chatbot.ask_with_sources(test_question)
    print(f"\n{result['answer']}")
    
    # Optionally start interactive chat
    print("\n" + "=" * 60)
    start_chat = input("\n💬 Start interactive chat session? (y/n): ").strip().lower()
    if start_chat == 'y':
        interactive_chat(chatbot)
    else:
        print("\n✅ All done! You can test queries programmatically or start interactive chat later.")