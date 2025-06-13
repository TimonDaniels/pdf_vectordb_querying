import os
from typing import Protocol

from langchain.schema import Document
import PyPDF2


class PDFParser(Protocol):
    """
    Protocol for PDF text extraction.
    Classes implementing this protocol should provide a method to extract text from PDFs.
    """
    def extract_text(pdf_path: str) -> str:
        pass

class PyPDFParser:

    @staticmethod
    def extract_text(pdf_path: str) -> str:
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")

        documents = []
        if text.strip():
            # Get document name from filename (remove .pdf extension)
            document_name = os.path.splitext(os.path.basename(pdf_path))[0]
            
            # Create document with metadata
            doc = Document(
                page_content=text,
                metadata={
                    "source": pdf_path,
                    "document": document_name,
                    "filename": os.path.basename(pdf_path)
                }
            )
            documents.append(doc)
        else:
            print(f"Warning: No text extracted from {pdf_path}")
        return text

