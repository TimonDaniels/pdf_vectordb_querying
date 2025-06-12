# PDF Vector Search System

A tool for getting familiar with vector databases using different embedding models. You can add your own PDF documents, extract text, create embeddings, and search through them using a web interface or command line.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Place PDF files in `data/` directory and run:
```bash
python example_usage.py
```

For web interface:
```bash
cd web-app && python app.py
```

4. **Add your PDF files**
   ```bash
   mkdir -p data
   # Copy your PDF files to the data/ directory
   ```

## 🚦 Quick Start

### Command Line Usage

```bash
python example_usage.py
# Select option 1-4 from the menu
```

### Web Interface

```bash
# Start the web server
cd web-app
python app.py

# Open browser to http://localhost:5000
```

## 📖 Usage Guide

### 1. Adding PDF Documents

Simply place your PDF files in the `data/` directory. The system will:
- Extract text from all PDFs
- Create document chunks for better search
- Build vector embeddings using your chosen model
- Store everything in ChromaDB for fast retrieval

### 2. Supported Embedding Models

| Model | Description | Use Case |
|-------|-------------|----------|
| `huggingface_minilm` | Fast, lightweight model | Quick searches, general use |
| `huggingface_mpnet` | Higher quality embeddings | Better accuracy, slower |
| `sentence_transformer` | SentenceTransformers library | Balanced performance |
| `openai` | OpenAI text-embedding-ada-002 | Highest quality (requires API key) |

### 3. Database Status Indicators

- 🚀 **Loaded**: Database is cached in memory (fastest searches)
- ✓ **Available**: Database exists on disk, ready to load
- ⏳ **Creating**: Database is being built in background
- ✗ **Failed**: Database creation encountered an error
- **Not Created**: Database needs to be built

## 🔧 Configuration

### PDF Processing Settings

Edit `pdf_processor.py` to adjust:

```python
# Text chunking parameters
chunk_size=2000        # Characters per chunk
chunk_overlap=400      # Overlap between chunks

# Default directories
pdf_directory="data"           # Where to find PDFs
base_db_directory="chroma_db"  # Where to store databases
```

### Adding New Embedding Models

```python
SUPPORTED_EMBEDDINGS = {
    "your_model_name": {
        "class": YourEmbeddingClass,
        "kwargs": {"model_name": "your-model-identifier"}
    }
}
```