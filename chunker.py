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

class LayoutTextChunker:

    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 400):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    
    def chunk_text(self, documents: List[Document]) -> List[Document]:
        assert hasattr(documents[0], "chunks"), "Document does not have chunks attribute, make sure to use a LayoutPDFReader or similar that supports chunking."
        documents = []
        
        for doc in documents:
            try:
                chunk_id = 0
                for chunk in doc.chunks():
                    chunk_text = chunk.to_context_text().strip()
                    
                    if chunk_text:  # Only create document if chunk has content
                        document = Document(
                            page_content=chunk_text,
                            metadata={
                                "chunk_id": chunk_id,
                                "chunk_type": "paragraph",
                                "parser": "layout_pdf_reader",
                                **doc.metadata  # Copy existing metadata
                            }
                        )
                        documents.append(document)
                        chunk_id += 1
                
                if documents:
                    print(f"Successfully extracted {len(documents)} paragraph chunks from {doc.metadata.get('source', 'unknown source')}")
                else:
                    print(f"Warning: No paragraph chunks extracted from {doc.metadata.get('source', 'unknown source')}")
                    
            except Exception as e:
                print(f"Error extracting paragraphs from {doc.metadata.get('source', 'unknown source')}: {e}")

        return documents