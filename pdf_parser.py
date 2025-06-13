import os
import ssl
import urllib3
from typing import Protocol, List

from langchain.schema import Document
import PyPDF2
from llmsherpa.readers import LayoutPDFReader

# Disable SSL warnings for development
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class PDFParser(Protocol):
    """
    Protocol for PDF text extraction.
    Classes implementing this protocol should provide a method to extract text from PDFs.
    """
    def extract_text(pdf_path: str) -> Document:
        pass

class PyPDFParser:

    @staticmethod
    def extract_text(pdf_path: str) -> Document:
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")

        if text.strip():
            # Get document name from filename (remove .pdf extension)
            document_name = os.path.splitext(os.path.basename(pdf_path))[0]
            doc = Document(
                page_content=text,
                metadata={
                    "source": pdf_path,
                    "document": document_name,
                    "filename": os.path.basename(pdf_path)
                }
            )
            return doc
        else:
            print(f"Warning: No text extracted from {pdf_path}")
            return None


class LayoutPDFParser:
    """
    PDF parser using LLMSherpa's LayoutPDFReader to extract text and chunk into paragraphs.
    This parser provides better structure-aware text extraction with paragraph-level chunking.
    """
    
    def __init__(self, llmsherpa_api_url: str = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"):
        """
        Initialize the LayoutPDFParser.
        
        Args:
            llmsherpa_api_url: URL for the LLMSherpa API service
        """
        self.llmsherpa_api_url = llmsherpa_api_url
        self.reader = LayoutPDFReader(llmsherpa_api_url)
    
    def extract_text(self, pdf_path: str) -> Document:
        try:
            # Parse the PDF document
            parsed_doc = self.reader.read_pdf(pdf_path)
            
            # Extract all text content
            text = ""
            for chunk in parsed_doc.chunks():
                text += chunk.to_context_text() + "\n\n"
            
            if text.strip():
                # Get document name from filename (remove .pdf extension)
                document_name = os.path.splitext(os.path.basename(pdf_path))[0]
                document = Document(
                    page_content=text,
                    metadata={
                        "source": pdf_path,
                        "document": document_name,
                        "filename": os.path.basename(pdf_path)
                    }
                )
                return document
            else:
                print(f"Warning: No text extracted from {pdf_path}")
                return None
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return None

    def extract_documents_with_paragraphs(self, pdf_path: str) -> List[Document]:
        """
        Extract text from PDF and create Document objects for each paragraph chunk.
        This method provides more granular document chunking at the paragraph level.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of Document objects, one per paragraph/chunk
        """
        documents = []
        
        if self.reader is None:
            print(f"Error: LayoutPDFReader not initialized. Cannot extract paragraphs from {pdf_path}")
            return documents
        
        try:
            # Parse the PDF document
            doc = self.reader.read_pdf(pdf_path)
            
            # Get document name from filename (remove .pdf extension)
            document_name = os.path.splitext(os.path.basename(pdf_path))[0]
            
            # Extract chunks as separate documents
            chunk_id = 0
            for chunk in doc.chunks():
                chunk_text = chunk.to_context_text().strip()
                
                if chunk_text:  # Only create document if chunk has content
                    document = Document(
                        page_content=chunk_text,
                        metadata={
                            "source": pdf_path,
                            "document": document_name,
                            "filename": os.path.basename(pdf_path),
                            "chunk_id": chunk_id,
                            "chunk_type": "paragraph",
                            "parser": "layout_pdf_reader"
                        }
                    )
                    documents.append(document)
                    chunk_id += 1
            
            if documents:
                print(f"Successfully extracted {len(documents)} paragraph chunks from {pdf_path}")
            else:
                print(f"Warning: No paragraph chunks extracted from {pdf_path}")
                
        except ssl.SSLError as ssl_err:
            print(f"SSL Error extracting paragraphs from {pdf_path}: {ssl_err}")
            print("Try using a local PDF parser or check your network connection.")
        except Exception as e:
            print(f"Error extracting paragraphs from {pdf_path}: {e}")
        
        return documents
    