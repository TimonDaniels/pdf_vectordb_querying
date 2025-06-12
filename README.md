# PDF Vector Search System

A tool for testing vector databases and embedding models on PDF files.

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

### 4. Performance Benefits

The caching system provides significant performance improvements:

- **First search**: Database loads from disk (~2-5 seconds)
- **Subsequent searches**: Uses cached database (~0.1-0.5 seconds)
- **Model switching**: Previously loaded models remain cached
- **Memory efficient**: Only loads databases when needed

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

## 📊 Examples

### Basic Search
```python
from pdf_processor import PDFProcessor

processor = PDFProcessor()
results = processor.search_with_model(
    query="artificial intelligence",
    model_name="huggingface_minilm",
    k=5
)

for result in results:
    print(f"Document: {result['document']}")
    print(f"Score: {result['similarity_score']:.3f}")
    print(f"Content: {result['content'][:200]}...")
```

### Model Comparison
```python
models = ["huggingface_minilm", "sentence_transformer"]
query = "machine learning"

for model in models:
    results = processor.search_with_model(query, model, k=3)
    print(f"\n{model} results:")
    for r in results:
        print(f"  {r['similarity_score']:.3f}: {r['content'][:100]}")
```

### Cache Management
```python
# Check cache status
stats = processor.get_cache_stats()
print(f"Cached models: {stats['cached_vectorstores']}")
print(f"Cache hits: {stats['hits']}")
print(f"Cache misses: {stats['misses']}")

# Clear specific model cache
processor.clear_cache("huggingface_minilm")

# Clear all caches
processor.clear_cache()
```

## 🌐 Web Interface Features

- **Real-time Search**: Type and search instantly
- **Model Selection**: Switch between embedding models
- **Status Updates**: Live database creation progress
- **Responsive Design**: Works on desktop and mobile
- **Result Details**: Full metadata and content display
- **Cache Status**: See which models are loaded

## 🔍 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main search interface |
| `/search` | POST | Perform search query |
| `/models` | GET | Get all models and status |
| `/models/<name>/status` | GET | Get specific model status |
| `/models/<name>/create` | POST | Trigger database creation |
| `/cache/status` | GET | Get cache statistics |
| `/cache/clear` | POST | Clear cache |

## 🧪 Performance Tips

1. **Start with smaller models** (huggingface_minilm) for testing
2. **Use OpenAI embeddings** for highest quality results
3. **Keep frequently used models loaded** in cache
4. **Monitor cache hit ratio** for optimal performance
5. **Adjust chunk size** based on your document types

## 🐛 Troubleshooting

### Common Issues

**Database creation fails**
- Check PDF files are readable
- Ensure sufficient disk space
- Verify model dependencies are installed

**Search returns no results**
- Try different embedding models
- Check if database was created successfully
- Verify PDF text extraction worked

**Memory issues**
- Clear unused caches: `processor.clear_cache()`
- Use lighter models for large document collections
- Reduce batch size in `pdf_processor.py`

## 📝 Requirements

- Python 3.8+
- See `requirements.txt` for package dependencies
- Optional: OpenAI API key for OpenAI embeddings

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

[Add your license information here]

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) for document processing
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [HuggingFace](https://huggingface.co/) for embedding models

---

For detailed web interface documentation, see [`web-app/README.md`](web-app/README.md).
