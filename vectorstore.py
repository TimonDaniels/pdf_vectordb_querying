"""
PDF Text Extraction and Vector Database System with Caching
Extract text from PDFs, chunk them, and store in ChromaDB for semantic search.
Supports multiple embedding models with separate databases and caching for better performance.
"""

import os
import glob
import json
from typing import List, Dict, Optional
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings, OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv

from pdf_parser import PDFParser
from chunker import TextChunker

# Load environment variables from .env.local file
load_dotenv('.env.local')


class VectorStore:
    """
    PDF processor with support for multiple embedding models and databases.
    Includes caching for better performance when switching between models.
    """
    
    # Supported embedding models
    SUPPORTED_EMBEDDINGS = {
        "huggingface_minilm": {
            "class": HuggingFaceEmbeddings,
            "kwargs": {"model_name": "all-MiniLM-L6-v2"}
        },
        "huggingface_mpnet": {
            "class": HuggingFaceEmbeddings,
            "kwargs": {"model_name": "all-mpnet-base-v2"}
        },
        "openai": {
            "class": OpenAIEmbeddings,
            "kwargs": {}
        },
        "sentence_transformer": {
            "class": SentenceTransformerEmbeddings,
            "kwargs": {"model_name": "all-MiniLM-L6-v2"}
        }
    }

    def __init__(self, pdf_directory: str = "data", base_db_directory: str = "chroma_db"):
        """
        Initialize the PDF processor.
        
        Args:
            pdf_directory: Directory containing PDF files to process
            base_db_directory: Base directory to store multiple ChromaDB databases
        """
        self.pdf_directory = pdf_directory
        self.base_db_directory = base_db_directory
        self.current_embedding_model = None
        self.current_embeddings = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=400,
            length_function=len
        )
        
        # Cache for loaded databases and embedding models
        self._vectorstore_cache = {}  # model_name -> Chroma vectorstore
        self._embedding_cache = {}    # model_name -> embedding instance
        self._cache_stats = {         # Track cache usage
            'hits': 0,
            'misses': 0,
            'loaded_models': []
        }
        
    def set_embedding_model(self, model_name: str):
        """
        Set the current embedding model.
        
        Args:
            model_name: Name of the embedding model from SUPPORTED_EMBEDDINGS
        """
        if model_name not in self.SUPPORTED_EMBEDDINGS:
            available_models = list(self.SUPPORTED_EMBEDDINGS.keys())
            raise ValueError(f"Unsupported embedding model: {model_name}. Available models: {available_models}")
        
        self.current_embedding_model = model_name
        self.current_embeddings = self.get_cached_embedding(model_name)
        print(f"Set embedding model to: {model_name}")
    
    def get_cached_embedding(self, model_name: str):
        """
        Get cached embedding model or create and cache it.
        
        Args:
            model_name: Name of the embedding model
            
        Returns:
            Embedding instance
        """
        if model_name in self._embedding_cache:
            print(f"Using cached embedding for {model_name}")
            self._cache_stats['hits'] += 1
            return self._embedding_cache[model_name]
        
        if model_name not in self.SUPPORTED_EMBEDDINGS:
            available_models = list(self.SUPPORTED_EMBEDDINGS.keys())
            raise ValueError(f"Unsupported embedding model: {model_name}. Available models: {available_models}")
        
        print(f"Loading and caching embedding model: {model_name}")
        self._cache_stats['misses'] += 1
        model_config = self.SUPPORTED_EMBEDDINGS[model_name]
        embedding = model_config["class"](**model_config["kwargs"])
        
        # Cache the embedding
        self._embedding_cache[model_name] = embedding
        if model_name not in self._cache_stats['loaded_models']:
            self._cache_stats['loaded_models'].append(model_name)
        
        return embedding
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            **self._cache_stats,
            'cached_vectorstores': list(self._vectorstore_cache.keys()),
            'cached_embeddings': list(self._embedding_cache.keys())
        }
    
    def clear_cache(self, model_name: str = None):
        """
        Clear cache for specific model or all models.
        
        Args:
            model_name: Specific model to clear, or None to clear all
        """
        if model_name:
            if model_name in self._vectorstore_cache:
                del self._vectorstore_cache[model_name]
                print(f"Cleared vectorstore cache for {model_name}")
            if model_name in self._embedding_cache:
                del self._embedding_cache[model_name]
                print(f"Cleared embedding cache for {model_name}")
            if model_name in self._cache_stats['loaded_models']:
                self._cache_stats['loaded_models'].remove(model_name)
        else:
            self._vectorstore_cache.clear()
            self._embedding_cache.clear()
            self._cache_stats['loaded_models'].clear()
            print("Cleared all caches")
    
    def get_database_directory(self, model_name: str) -> str:
        """Get the database directory for a specific embedding model."""
        return os.path.join(self.base_db_directory, model_name)
    
    def get_available_databases(self) -> List[str]:
        """
        Get list of available databases (by embedding model).
        
        Returns:
            List of embedding model names that have databases
        """
        available = []
        for model_name in self.SUPPORTED_EMBEDDINGS.keys():
            db_dir = self.get_database_directory(model_name)
            if os.path.exists(db_dir) and os.path.exists(os.path.join(db_dir, "chroma.sqlite3")):
                available.append(model_name)
        
        return available
    
    def get_database_info(self, model_name: str) -> Optional[Dict]:
        """
        Get metadata information about a specific database.
        
        Args:
            model_name: Name of the embedding model
            
        Returns:
            Dictionary with database metadata or None if not found
        """
        db_dir = self.get_database_directory(model_name)
        metadata_file = os.path.join(db_dir, "metadata.json")
        
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error reading metadata for {model_name}: {e}")
        
        return None
    
    def save_database_metadata(self, model_name: str, document_count: int):
        """Save metadata about the database."""
        db_dir = self.get_database_directory(model_name)
        metadata = {
            "embedding_model": model_name,
            "created_at": datetime.now().isoformat(),
            "document_count": document_count,
            "pdf_directory": self.pdf_directory
        }
        
        metadata_file = os.path.join(db_dir, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved metadata for {model_name} database")


    def create_database_with_model(self, model_name: str, parser: PDFParser, chunker: TextChunker, force_recreate: bool = False) -> Optional[Chroma]:
        """
        Create ChromaDB vector database with a specific embedding model.
        
        Args:
            model_name: Name of the embedding model to use
            force_recreate: Whether to recreate if database already exists
            
        Returns:
            ChromaDB vector store or None if creation fails
        """
        if model_name not in self.SUPPORTED_EMBEDDINGS:
            available_models = list(self.SUPPORTED_EMBEDDINGS.keys())
            print(f"Unsupported embedding model: {model_name}. Available: {available_models}")
            return None
        
        # Set the embedding model
        self.set_embedding_model(model_name)
        
        db_dir = self.get_database_directory(model_name)
        
        # Check if database already exists
        if os.path.exists(db_dir) and not force_recreate:
            print(f"Database for {model_name} already exists. Use force_recreate=True to overwrite.")
            return self.load_database_by_model(model_name)
        
        print(f"Creating database with {model_name} embedding model...")
        
        try:
            # Step 1: Extract text from all PDFs
            pdf_files = glob.glob(os.path.join(self.pdf_directory, "*.pdf"))
            documents = []
            for pdf_path in pdf_files:
                print(f"Processing PDF: {os.path.basename(pdf_path)}")
                document = parser.extract_text(pdf_path)
                if document:
                    documents.append(Document(page_content=document))
                else:
                    print(f"Warning: No text extracted from {pdf_path}")

            if not documents:
                print("No documents were processed. Check your PDF directory.")
                return None
            
            # Step 2: Chunk the documents
            chunked_docs = chunker.chunk_text(documents)
            
            # Step 3: Create vector database
            print(f"Creating vector database in {db_dir}...")
            
            # Remove existing database if force_recreate
            if os.path.exists(db_dir) and force_recreate:
                import shutil
                print("Removing existing database to recreate...")
                shutil.rmtree(db_dir)
            
            # Create the database directory
            os.makedirs(db_dir, exist_ok=True)
            
            # Process documents in batches
            batch_size = 500
            vectorstore = None

            for i in range(0, len(chunked_docs), batch_size):
                batch = chunked_docs[i:i + batch_size]
                print(f"Processing batch {i//batch_size + 1}/{(len(chunked_docs)-1)//batch_size + 1} ({len(batch)} documents)...")
                
                if vectorstore is None:
                    vectorstore = Chroma.from_documents(
                        documents=batch,
                        embedding=self.current_embeddings,
                        persist_directory=db_dir,
                        collection_name="political_programs"
                    )
                    print(f"Initial vector store created with {len(batch)} documents")
                else:
                    vectorstore.add_documents(batch)
                    print(f"Added {len(batch)} documents to vector store")

            # Save metadata
            self.save_database_metadata(model_name, len(chunked_docs))
            
            # Cache the vectorstore for future use
            self._vectorstore_cache[model_name] = vectorstore
            
            print(f"Database created successfully for {model_name} with {len(chunked_docs)} total documents")
            return vectorstore
            
        except Exception as e:
            print(f"Error creating database for {model_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    def load_database_by_model(self, model_name: str) -> Optional[Chroma]:
        """
        Load existing ChromaDB database for a specific embedding model.
        Uses caching to avoid reloading the same database multiple times.
        
        Args:
            model_name: Name of the embedding model
            
        Returns:
            ChromaDB vector store or None if loading fails
        """
        if model_name not in self.SUPPORTED_EMBEDDINGS:
            available_models = list(self.SUPPORTED_EMBEDDINGS.keys())
            print(f"Unsupported embedding model: {model_name}. Available: {available_models}")
            return None
        
        # Check cache first
        if model_name in self._vectorstore_cache:
            print(f"Using cached vectorstore for {model_name}")
            self._cache_stats['hits'] += 1
            return self._vectorstore_cache[model_name]
        
        self._cache_stats['misses'] += 1
        
        db_dir = self.get_database_directory(model_name)
        
        if not os.path.exists(db_dir):
            print(f"Database directory for {model_name} does not exist: {db_dir}")
            return None
        
        # Get the embedding model (cached)
        embedding = self.get_cached_embedding(model_name)
        
        try:
            print(f"Loading database for {model_name}...")
            vectorstore = Chroma(
                persist_directory=db_dir,
                embedding_function=embedding,
                collection_name="political_programs"
            )
            
            # Test the connection
            vectorstore.similarity_search("test", k=1)
            
            # Cache the vectorstore for future use
            self._vectorstore_cache[model_name] = vectorstore
            
            print(f"Successfully loaded and cached database for {model_name}")
            return vectorstore
            
        except Exception as e:
            print(f"Error loading database for {model_name}: {e}")
            return None
    
    
    def search_with_model(self, query: str, model_name: str, k: int = 5, filter: Optional[Dict[str, str]] = None) -> List[Dict]:
        """
        Search using a specific embedding model's database.
        Uses cached vectorstore for better performance.
        
        Args:
            query: Search query
            model_name: Name of the embedding model to use
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of search results
        """
        try:
            vectorstore = self.load_database_by_model(model_name)
            if vectorstore is None:
                print(f"Could not load database for {model_name}")
                return []
                
            results = vectorstore.similarity_search_with_score(query, k=k, filter=filter)
            
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "document": doc.metadata.get("document", "Unknown"),
                    "filename": doc.metadata.get("filename", "Unknown"),
                    "source": doc.metadata.get("source", "Unknown"),
                    "chunk_id": doc.metadata.get("chunk_id", "Unknown"),
                    "similarity_score": score,
                    "embedding_model": model_name
                })
            
            return formatted_results
        
        except Exception as e:
            print(f"Error searching with {model_name}: {e}")
            return []

    def list_databases_info(self):
        """Print information about all available databases."""
        available = self.get_available_databases()
        
        if not available:
            print("No databases found.")
            return
        
        print(f"Available databases ({len(available)}):")
        print("-" * 50)
        
        for model_name in available:
            info = self.get_database_info(model_name)
            if info:
                print(f"Model: {model_name}")
                print(f"  Created: {info.get('created_at', 'Unknown')}")
                print(f"  Documents: {info.get('document_count', 'Unknown')}")
                print(f"  PDF Directory: {info.get('pdf_directory', 'Unknown')}")
                print()
            else:
                print(f"Model: {model_name} (no metadata available)")
                print()

    def reset_database(self, model_name: Optional[str] = None):
        """
        Reset database(s) by removing the directory.
        Also clears cache for the reset databases.
        
        Args:
            model_name: Specific model to reset, or None to reset all
        """
        import shutil
        
        if model_name:
            # Reset specific model database
            db_dir = self.get_database_directory(model_name)
            if os.path.exists(db_dir):
                print(f"Removing database for {model_name}: {db_dir}")
                shutil.rmtree(db_dir)
                print(f"Database for {model_name} reset complete.")
            else:
                print(f"No database found for {model_name}")
            
            # Clear cache for this model
            self.clear_cache(model_name)
        else:
            # Reset all databases
            if os.path.exists(self.base_db_directory):
                print(f"Removing all databases: {self.base_db_directory}")
                shutil.rmtree(self.base_db_directory)
                print("All databases reset complete.")
            else:
                print("No databases found to reset.")
            
            # Clear all caches
            self.clear_cache()
