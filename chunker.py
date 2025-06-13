from typing import List, Protocol

from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextChunker(Protocol):
    """
    Protocol for text chunking.
    Classes implementing this protocol should provide a method to chunk text into smaller segments.
    """
    def __init__(*args, **kwargs):
        pass

    def chunk_text(documents: List[Document]) -> List[Document]:
        pass


class RecursiveTextChunker:

    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 400):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, documents: List[Document]) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )

        chunked_docs = text_splitter.split_documents(documents)
        # Verify and enhance metadata for each chunk
        for i, chunk in enumerate(chunked_docs):
            # Add chunk index to metadata
            chunk.metadata["chunk_id"] = i
            # Ensure all required metadata is present
            if "document" not in chunk.metadata:
                chunk.metadata["document"] = "Unknown"
            if "filename" not in chunk.metadata:
                chunk.metadata["filename"] = "Unknown"
            if "source" not in chunk.metadata:
                chunk.metadata["source"] = "Unknown"        
        print(f"Created {len(chunked_docs)} chunks from {len(documents)} documents")

        return chunked_docs 

