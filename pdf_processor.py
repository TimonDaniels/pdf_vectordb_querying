"""
PDF Text Extraction and Vector Database System
Extract text from PDFs, chunk them, and store in ChromaDB for semantic search.
"""

import os
import glob
from typing import List, Dict, Optional
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document


class PDFProcessor:
    def __init__(self, pdf_directory: str = "programma-2023", db_directory: str = "chroma_db"):
        """
        Initialize the PDF processor.
        
        Args:
            pdf_directory: Directory containing PDF files
            db_directory: Directory to store ChromaDB database
        """
        self.pdf_directory = pdf_directory
        self.db_directory = db_directory
        self.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a single PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as string
        """
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
        
        return text
    
    def process_all_pdfs(self) -> List[Document]:
        """
        Extract text from all PDFs in the directory and create documents.
        
        Returns:
            List of LangChain Document objects
        """
        documents = []
        pdf_files = glob.glob(os.path.join(self.pdf_directory, "*.pdf"))
        
        print(f"Found {len(pdf_files)} PDF files to process...")
        
        for pdf_path in pdf_files:
            print(f"Processing: {os.path.basename(pdf_path)}")
            
            # Extract text
            text = self.extract_text_from_pdf(pdf_path)
            
            if text.strip():
                # Get party name from filename (remove .pdf extension)
                party_name = os.path.splitext(os.path.basename(pdf_path))[0]
                
                # Create document with metadata
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": pdf_path,
                        "party": party_name,
                        "filename": os.path.basename(pdf_path)
                    }
                )
                documents.append(doc)
            else:
                print(f"Warning: No text extracted from {pdf_path}")
        
        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks while preserving metadata.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of chunked Document objects with preserved metadata
        """
        print("Splitting documents into chunks...")
        chunked_docs = self.text_splitter.split_documents(documents)
        
        # Verify and enhance metadata for each chunk
        for i, chunk in enumerate(chunked_docs):
            # Add chunk index to metadata
            chunk.metadata["chunk_id"] = i
            # Ensure all required metadata is present
            if "party" not in chunk.metadata:
                chunk.metadata["party"] = "Unknown"
            if "filename" not in chunk.metadata:
                chunk.metadata["filename"] = "Unknown"
            if "source" not in chunk.metadata:
                chunk.metadata["source"] = "Unknown"
            print(f"Created {len(chunked_docs)} chunks from {len(documents)} documents")
        return chunked_docs

    def create_vector_database(self, documents: List[Document]) -> Chroma:
        """
        Create ChromaDB vector database from documents.
        
        Args:
            documents: List of Document objects
            
        Returns:
            ChromaDB vector store
        """
        print("Creating vector database...")
        print(f"Processing {len(documents)} documents...")
        
        # Create the database directory if it doesn't exist
        os.makedirs(self.db_directory, exist_ok=True)
        
        try:
            # Reset database if it exists to avoid conflicts
            if os.path.exists(self.db_directory):
                import shutil
                print("Removing existing database directory to avoid conflicts...")
                shutil.rmtree(self.db_directory)
                os.makedirs(self.db_directory, exist_ok=True)
            
            print("Initializing ChromaDB...")
            # Process documents in smaller batches to avoid memory issues
            batch_size = 50
            vectorstore = None

            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                print(f"Processing batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1} ({len(batch)} documents)...")
                
                if vectorstore is None:
                    # Create initial vector store with first batch
                    print("Creating initial vector store...")
                    vectorstore = Chroma.from_documents(
                        documents=batch,
                        embedding=self.embeddings,
                        persist_directory=self.db_directory,
                        collection_name="political_programs"
                    )
                    print(f"Initial vector store created with {len(batch)} documents")
                else:
                    # Add subsequent batches to existing store
                    print(f"Adding batch to existing vector store...")
                    vectorstore.add_documents(batch)
                    print(f"Added {len(batch)} documents to vector store")

            print(f"Vector database created successfully with {len(documents)} total documents")
            return vectorstore
            
        except Exception as e:
            print(f"Error creating vector database: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            raise

    def load_existing_database(self) -> Optional[Chroma]:
        """
        Load existing ChromaDB database.
        
        Returns:
            ChromaDB vector store or None if loading fails
        """
        try:
            # Check if database directory exists
            if not os.path.exists(self.db_directory):
                print(f"Database directory '{self.db_directory}' does not exist.")
                return None
            
            vectorstore = Chroma(
                persist_directory=self.db_directory,
                embedding_function=self.embeddings,
                collection_name="political_programs"
            )
            
            # Test the connection by trying a simple operation
            try:
                vectorstore.similarity_search("test", k=1)
            except Exception as test_e:
                print(f"Database connection test failed: {test_e}")
                return None
                
            return vectorstore
            
        except Exception as e:
            print(f"Error loading existing database: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            return None

    def process_pdfs_and_create_db(self):
        """
        Complete pipeline: extract, chunk, and store PDFs in vector database.
        """
        print("Starting PDF processing pipeline...")
        
        try:
            # Step 1: Extract text from all PDFs
            documents = self.process_all_pdfs()
            
            if not documents:
                print("No documents were processed. Check your PDF directory.")
                return None
            
            # Step 2: Chunk the documents
            chunked_docs = self.chunk_documents(documents)
            
            # Step 3: Create vector database
            vectorstore = self.create_vector_database(chunked_docs)
            
            print("Pipeline completed successfully!")
            return vectorstore
        except Exception as e:
            print(f"Error in pipeline: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            return None

    def search_similar_content(self, query: str, k: int = 5, filter: Optional[Dict[str, str]] = None) -> List[Dict]:
        try:
            vectorstore = self.load_existing_database()
            if vectorstore is None:
                print("Could not load database. Try recreating it by running the full pipeline.")
                return []
                
            results = vectorstore.similarity_search_with_score(query, k=k, filter=filter)
            
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "party": doc.metadata.get("party", "Unknown"),
                    "filename": doc.metadata.get("filename", "Unknown"),
                    "source": doc.metadata.get("source", "Unknown"),
                    "chunk_id": doc.metadata.get("chunk_id", "Unknown"),
                    "similarity_score": score
                })
            
            return formatted_results
        
        except Exception as e:
            print(f"Error searching database: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            return []
    
    def reset_database(self):
        """
        Reset the database by removing the existing directory and recreating it.
        """
        import shutil
        
        if os.path.exists(self.db_directory):
            print(f"Removing existing database directory: {self.db_directory}")
            shutil.rmtree(self.db_directory)
        
        print("Database reset complete. Run process_pdfs_and_create_db() to recreate.")