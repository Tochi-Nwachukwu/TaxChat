# TaxChat - Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Setup Environment

```bash
# Create .env file with your OpenAI API key
echo "OPENAI_API_KEY=your_key_here" > .env

# Install dependencies
uv sync
```

### Step 2: Build Knowledge Base

```bash
# Run the pipeline to process documents
python main.py
```

This creates the vector database from your tax documents.

### Step 3: Launch Chat Interface

```bash
# Start the Streamlit app
./run_chat.sh

# Or manually:
uv run streamlit run src/chat_interface.py
```

The app will open in your browser at http://localhost:8501

## 🎯 Using the Chat Interface

### Features

1. **Chat with AI**: Ask questions about Nigerian Tax Law in natural language
2. **View Sources**: See which documents were used to answer your question
3. **Example Questions**: Click pre-made questions to get started quickly
4. **Clear History**: Reset the conversation at any time
5. **Context Awareness**: The bot remembers your conversation

### Example Queries

- "What is the tax rate for companies in Nigeria?"
- "If I earn 60,000 naira monthly, how much tax do I pay?"
- "Explain petroleum profits tax"
- "What deductions are allowed for income tax?"
- "How is capital gains tax calculated?"

### Tips for Best Results

✅ **Do:**
- Ask specific questions about tax law
- Reference amounts, rates, or specific scenarios
- Follow up with related questions
- Use the example questions as templates

❌ **Don't:**
- Ask questions unrelated to Nigerian tax law
- Expect financial advice (this is informational only)
- Ask about very recent changes not in the documents

## 📊 Interface Overview

### Sidebar
- **Load Knowledge Base**: Initialize the RAG system
- **Show Sources**: Toggle source display on/off
- **Clear Chat**: Start a fresh conversation
- **Example Questions**: Quick start templates

### Main Area
- **Chat History**: See all messages
- **Input Box**: Type your questions
- **Sources**: Expandable sections showing references

## 🛠️ Troubleshooting

### "Knowledge base not found"
**Solution**: Run `python main.py` first to create the vector database

### "API Key Error"
**Solution**: Check your `.env` file has `OPENAI_API_KEY=your_key`

### App won't start
**Solution**: 
```bash
# Reinstall dependencies
uv sync

# Try running directly
streamlit run src/chat_interface.py
```

### Slow responses
**Cause**: First query loads the model and database (10-15 seconds)
**Normal**: Subsequent queries are faster (2-5 seconds)

## 📁 File Structure

```
ADVANCED_PYTHON_RAG/
├── data/              # Your tax documents (PDF & Markdown)
├── vector_db/         # Generated vector database
├── src/
│   └── chat_interface.py  # Streamlit app
├── main.py            # Pipeline to build knowledge base
├── run_chat.sh        # Quick start script
└── .env              # Your API key (create this)
```

## 🎨 Customization

### Change Model
Edit `constants.py`:
```python
MODEL = "gpt-4o-mini"  # or "gpt-4o", "gpt-3.5-turbo"
```

### Adjust Temperature
In `chat_interface.py` line 29:
```python
chatbot = create_rag_chatbot(vectorstore, temperature=0.7)
```
- Lower (0.0-0.3): More focused, factual
- Higher (0.7-1.0): More creative, varied

### Change Number of Sources
In `retrieval_system.py` line 25:
```python
search_kwargs={"k": 5}  # Number of chunks to retrieve
```

## 💡 Pro Tips

1. **Process All Documents**: Edit `main.py` line 70 to remove `[:10]` limit
2. **Keyboard Shortcuts**: Press `/` to focus the chat input
3. **Copy Answers**: Right-click any answer to copy text
4. **Mobile Friendly**: The interface works on tablets and phones

## 📞 Need Help?

- Check the main README.md for detailed documentation
- Review error messages carefully
- Ensure .env file is in the project root
- Verify OpenAI API key has credits

---

Enjoy using TaxChat! 🎉

