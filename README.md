# TaxChat - Nigerian Tax Law AI Assistant 💼

An advanced RAG (Retrieval-Augmented Generation) system for querying Nigerian Tax Law documents, powered by LangChain, OpenAI, and Streamlit.

## Features

- 📚 **Document Loading**: Loads PDF and Markdown files from tax law documents
- ✂️ **Agentic Chunking**: Intelligently groups related content using AI
- 🔍 **Vector Search**: Fast semantic search using ChromaDB and OpenAI embeddings
- 💬 **Interactive Chat**: Beautiful Streamlit UI for natural conversations
- 📖 **Source Citations**: View source documents for every answer
- 🧠 **Context-Aware**: Maintains conversation history for coherent responses

## Installation

### Prerequisites

- Python 3.11+
- OpenAI API key
- `uv` package manager (recommended) or `pip`

### Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd ADVANCED_PYTHON_RAG
```

2. Create a `.env` file with your OpenAI API key:
```bash
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

3. Install dependencies:
```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

## Usage

### Step 1: Build the Knowledge Base

Run the main pipeline to process documents and create the vector database:

```bash
python main.py
```

This will:
1. Load all PDF and Markdown files from the `data/` folder
2. Extract propositions from the documents
3. Create intelligent chunks using agentic chunking
4. Generate embeddings and store them in a vector database
5. Test the retrieval system

**Note**: The demo processes only 10 propositions. To process all documents, edit `main.py` line 70:
```python
# Change from:
proposition_texts = [p['text'] for p in propositions[:10]]

# To:
proposition_texts = [p['text'] for p in propositions]
```

### Step 2: Launch the Chat Interface

Start the Streamlit web interface:

```bash
# Using the run script
./run_chat.sh

# Or directly
uv run streamlit run src/chat_interface.py
```

The chat interface will open in your browser at `http://localhost:8501`

### Command-Line Chat (Optional)

For a terminal-based chat experience, use:

```python
from src.embeddings_manager import load_vectorstore
from src.retrieval_system import create_rag_chatbot, interactive_chat
from constants import db_name

# Load the system
vectorstore = load_vectorstore(db_name)
chatbot = create_rag_chatbot(vectorstore)

# Start interactive chat
interactive_chat(chatbot)
```

## Project Structure

```
ADVANCED_PYTHON_RAG/
├── data/
│   ├── pdf/                    # PDF tax documents
│   │   └── Nigeria-Tax-Act-2025.pdf
│   └── markdown/               # Markdown tax documents
│       ├── tax_addendum1.md
│       └── ...
├── src/
│   ├── document_loader.py     # Load documents from data folder
│   ├── agentic_chunker.py     # AI-powered intelligent chunking
│   ├── embeddings_manager.py  # Create and manage vector embeddings
│   ├── retrieval_system.py    # RAG chatbot logic
│   └── chat_interface.py      # Streamlit web UI
├── vector_db/                  # ChromaDB vector store (created on first run)
├── main.py                     # Main pipeline script
├── constants.py                # Configuration constants
├── requirements.txt            # Python dependencies
├── pyproject.toml             # Project metadata
└── README.md                   # This file
```

## Components

### 1. Document Loader
Loads PDF and Markdown files from the `data/` directory with metadata tracking.

### 2. Agentic Chunker
Uses GPT-4o-mini to intelligently group related propositions:
- Analyzes content semantically
- Creates dynamic summaries and titles
- Groups similar concepts together

### 3. Embeddings Manager
- Uses OpenAI's `text-embedding-3-small` model
- Stores embeddings in ChromaDB
- Supports persistence and reloading

### 4. Retrieval System
- Semantic search over document chunks
- Conversational memory (keeps last 3 exchanges)
- Source document tracking
- Context-aware responses

### 5. Chat Interface (Streamlit)
- Modern, responsive UI
- Real-time chat
- Source citation display
- Example questions
- Chat history management

## Configuration

Edit `constants.py` to change settings:

```python
MODEL = "gpt-4o-mini"          # OpenAI model
db_name = "vector_db"           # Vector database name
```

## Example Questions

- "What is the tax rate for companies?"
- "How is personal income tax calculated?"
- "If I earn 60,000 naira monthly, how much tax would I pay?"
- "What are the deductions allowed?"
- "Explain petroleum profits tax"

## Cost Considerations

- Document processing uses OpenAI API (agentic chunking)
- Each query uses embeddings + GPT-4o-mini
- Process all documents: ~$1-5 depending on size
- Per query: ~$0.001-0.01

## Troubleshooting

### "Vectorstore not found"
Run `python main.py` first to create the knowledge base.

### "ModuleNotFoundError"
Ensure all dependencies are installed: `uv sync` or `pip install -r requirements.txt`

### "API Key Error"
Check that your `.env` file contains a valid `OPENAI_API_KEY`

## Technologies Used

- **LangChain**: RAG framework
- **OpenAI**: GPT-4o-mini and embeddings
- **ChromaDB**: Vector database
- **Streamlit**: Web interface
- **PyPDF**: PDF processing
- **Python 3.11+**

## License

Open Source License feel free to contribute

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## Acknowledgments

Built with LangChain, OpenAI, and the power of RAG! 🚀

