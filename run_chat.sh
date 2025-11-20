#!/bin/bash
# Run the Streamlit chat interface

echo "🚀 Starting TaxChat Streamlit Interface..."
echo ""

# Check if vector database exists
if [ ! -d "vector_db" ]; then
    echo "⚠️  Vector database not found!"
    echo "Please run 'python main.py' first to create the knowledge base."
    echo ""
    exit 1
fi

# Run Streamlit
uv run streamlit run src/chat_interface.py

