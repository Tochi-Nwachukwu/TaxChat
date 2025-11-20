from constants import MODEL
from langchain_openai import ChatOpenAI
from typing import Dict, Any, List, Tuple
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate


class RAGChatbot:
    """
    A conversational RAG chatbot that retrieves information from a vectorstore
    and provides contextual answers using an LLM.
    """
    
    def __init__(self, vectorstore: Chroma, model_name: str = MODEL, temperature: float = 0.7):
        """
        Initialize the RAG chatbot.
        
        Args:
            vectorstore: The Chroma vectorstore to retrieve from
            model_name: The OpenAI model to use (default from constants)
            temperature: Temperature for response generation (0-1)
        """
        self.llm = ChatOpenAI(temperature=temperature, model_name=model_name)
        self.vectorstore = vectorstore
        self.retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}  # Retrieve top 5 most relevant chunks
        )
        self.chat_history: List[Tuple[str, str]] = []
        
        # System prompt
        self.system_prompt = (
            "You are a helpful AI assistant specializing in Nigerian tax law. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer based on the context, say that you don't know. "
            "Keep the answer concise and accurate."
        )
    
    def ask(self, question: str) -> Dict[str, Any]:
        """
        Ask a question and get an answer from the RAG system.
        
        Args:
            question: The question to ask
            
        Returns:
            Dictionary containing 'answer' and 'context' (source documents)
        """
        # Retrieve relevant documents
        docs = self.retriever.invoke(question)
        
        # Format context from retrieved documents
        context = "\n\n".join([
            f"Document {i+1} (Title: {doc.metadata.get('title', 'N/A')}):\n{doc.page_content}"
            for i, doc in enumerate(docs)
        ])
        
        # Build messages for the LLM
        messages = [
            SystemMessage(content=self.system_prompt)
        ]
        
        # Add chat history
        for q, a in self.chat_history[-3:]:  # Keep last 3 exchanges for context
            messages.append(HumanMessage(content=q))
            messages.append(AIMessage(content=a))
        
        # Add current question with context
        user_message = f"Context:\n{context}\n\nQuestion: {question}"
        messages.append(HumanMessage(content=user_message))
        
        # Get response from LLM
        response = self.llm.invoke(messages)
        answer = response.content
        
        # Update chat history
        self.chat_history.append((question, answer))
        
        return {
            "answer": answer,
            "context": docs,
            "question": question
        }
    
    def get_answer(self, question: str) -> str:
        """
        Ask a question and return just the answer text.
        
        Args:
            question: The question to ask
            
        Returns:
            The answer as a string
        """
        result = self.ask(question)
        return result["answer"]
    
    def clear_history(self):
        """Clear the conversation history."""
        self.chat_history = []
        print("💭 Conversation history cleared")
    
    def get_chat_history(self) -> List[Tuple[str, str]]:
        """Get the current chat history."""
        return self.chat_history
    
    def ask_with_sources(self, question: str, show_sources: bool = True) -> Dict[str, Any]:
        """
        Ask a question and display sources used for the answer.
        
        Args:
            question: The question to ask
            show_sources: Whether to print the sources
            
        Returns:
            Dictionary with answer and source information
        """
        result = self.ask(question)
        
        if show_sources and result.get("context"):
            print("\n" + "=" * 60)
            print("📚 Sources Used:")
            print("=" * 60)
            for i, doc in enumerate(result["context"], 1):
                print(f"\n[Source {i}]")
                print(f"  Title: {doc.metadata.get('title', 'N/A')}")
                summary = doc.metadata.get('summary', 'N/A')
                if len(summary) > 150:
                    summary = summary[:150] + "..."
                print(f"  Summary: {summary}")
                content_preview = doc.page_content[:200]
                if len(doc.page_content) > 200:
                    content_preview += "..."
                print(f"  Content: {content_preview}")
        
        return result


def create_rag_chatbot(vectorstore: Chroma, model_name: str = MODEL, temperature: float = 0.7) -> RAGChatbot:
    """
    Create a RAG chatbot instance.
    
    Args:
        vectorstore: The Chroma vectorstore to use for retrieval
        model_name: The OpenAI model name
        temperature: Temperature for response generation
        
    Returns:
        RAGChatbot instance
    """
    return RAGChatbot(vectorstore, model_name, temperature)


def interactive_chat(chatbot: RAGChatbot):
    """
    Start an interactive chat session with the RAG chatbot.
    
    Args:
        chatbot: The RAGChatbot instance
    """
    print("\n" + "=" * 60)
    print("💬 Interactive Chat Session Started")
    print("=" * 60)
    print("Commands:")
    print("  - Type your question to ask")
    print("  - Type 'clear' to clear conversation history")
    print("  - Type 'quit' or 'exit' to end the session")
    print("=" * 60 + "\n")
    
    while True:
        try:
            question = input("\n🤔 You: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye! Chat session ended.")
                break
            
            if question.lower() == 'clear':
                chatbot.clear_history()
                continue
            
            print("\n🤖 Assistant: ", end="", flush=True)
            result = chatbot.ask_with_sources(question, show_sources=False)
            print(result["answer"])
            
            # Optionally show sources
            if result.get("context"):
                show = input("\n📚 Show sources? (y/n): ").strip().lower()
                if show == 'y':
                    print("\n" + "=" * 60)
                    print("Sources Used:")
                    print("=" * 60)
                    for i, doc in enumerate(result["context"], 1):
                        print(f"\n[{i}] {doc.metadata.get('title', 'Untitled')}")
                        content = doc.page_content[:150]
                        if len(doc.page_content) > 150:
                            content += "..."
                        print(f"    {content}")
        
        except KeyboardInterrupt:
            print("\n\n👋 Chat session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again.")
