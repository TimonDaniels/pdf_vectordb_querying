import os
import ssl
import urllib3
from typing import Protocol, List

from langchain.schema import Document
import PyPDF2
from llmsherpa.readers import LayoutPDFReader

# Import PyMuPDF with error handling
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("Warning: PyMuPDF (fitz) not found. PyMuPDFParser will not be available.")

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


class PyMuPDFParser:
    """
    PDF parser using PyMuPDF (fitz) to extract text while preserving document structure.
    This parser provides excellent structure preservation including:
    - Text blocks and paragraphs
    - Font information
    - Text formatting (bold, italic)
    - Headers and sections
    - Tables and lists
    - Page layout information
    """
    
    def __init__(self):
        """Initialize the PyMuPDFParser and check if PyMuPDF is available."""
        if not PYMUPDF_AVAILABLE:
            raise ImportError("PyMuPDF (fitz) is required but not installed. Install it with: pip install PyMuPDF")
    
    def extract_text(self, pdf_path: str) -> Document:
        """
        Extract all text from PDF as a single document with preserved structure.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Document object with structured text content
        """
        try:
            # Open the PDF document
            doc = fitz.open(pdf_path)
            structured_text = ""
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Add page header
                structured_text += f"\n{'='*50}\nPage {page_num + 1}\n{'='*50}\n\n"
                
                # Get text blocks with structure information
                blocks = page.get_text("dict")
                
                for block in blocks["blocks"]:
                    if "lines" in block:  # Text block
                        block_text = self._extract_block_text(block)
                        if block_text.strip():
                            structured_text += block_text + "\n\n"
            
            doc.close()
            
            if structured_text.strip():
                # Get document name from filename (remove .pdf extension)
                document_name = os.path.splitext(os.path.basename(pdf_path))[0]
                document = Document(
                    page_content=structured_text,
                    metadata={
                        "source": pdf_path,
                        "document": document_name,
                        "filename": os.path.basename(pdf_path),
                        "parser": "pymupdf",
                        "structure_preserved": True
                    }
                )
                return document
            else:
                print(f"Warning: No text extracted from {pdf_path}")
                return None
                
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return None
    
    def extract_documents_with_structure(self, pdf_path: str) -> List[Document]:
        """
        Extract text from PDF and create Document objects for each structural element.
        This method provides granular document chunking based on text blocks and paragraphs.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of Document objects, one per text block/paragraph
        """
        documents = []
        
        try:
            # Open the PDF document
            doc = fitz.open(pdf_path)
            document_name = os.path.splitext(os.path.basename(pdf_path))[0]
            
            block_id = 0
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                blocks = page.get_text("dict")
                
                for block in blocks["blocks"]:
                    if "lines" in block:  # Text block
                        block_text = self._extract_block_text(block)
                        
                        if block_text.strip():  # Only create document if block has content
                            # Determine block type based on formatting
                            block_type = self._classify_block_type(block)
                            
                            document = Document(
                                page_content=block_text.strip(),
                                metadata={
                                    "source": pdf_path,
                                    "document": document_name,
                                    "filename": os.path.basename(pdf_path),
                                    "page_number": page_num + 1,
                                    "block_id": block_id,
                                    "block_type": block_type,
                                    "parser": "pymupdf",
                                    "structure_preserved": True,
                                    "bbox": block.get("bbox", [])  # Bounding box coordinates
                                }
                            )
                            documents.append(document)
                            block_id += 1
            
            doc.close()
            
            if documents:
                print(f"Successfully extracted {len(documents)} structured blocks from {pdf_path}")
            else:
                print(f"Warning: No structured blocks extracted from {pdf_path}")
                
        except Exception as e:
            print(f"Error extracting structured text from {pdf_path}: {e}")
        
        return documents
    
    def _extract_block_text(self, block) -> str:
        """
        Extract text from a block while preserving formatting information.
        
        Args:
            block: PyMuPDF text block dictionary
            
        Returns:
            Formatted text string
        """
        block_text = ""
        
        for line in block["lines"]:
            line_text = ""
            current_font_size = None
            
            for span in line["spans"]:
                text = span["text"]
                font_size = span["size"]
                font_flags = span["flags"]
                
                # Add formatting markers based on font properties
                if font_flags & 16:  # Bold
                    text = f"**{text}**"
                if font_flags & 2:  # Italic
                    text = f"*{text}*"
                
                # Check for potential headers (larger font size)
                if current_font_size is None:
                    current_font_size = font_size
                elif font_size > current_font_size + 2:  # Significantly larger font
                    text = f"# {text}"
                
                line_text += text
            
            if line_text.strip():
                block_text += line_text + "\n"
        
        return block_text
    
    def _classify_block_type(self, block) -> str:
        """
        Classify the type of text block based on formatting and content.
        
        Args:
            block: PyMuPDF text block dictionary
            
        Returns:
            Block type classification
        """
        if not block.get("lines"):
            return "unknown"
        
        # Analyze the first line to determine block type
        first_line = block["lines"][0]
        if not first_line.get("spans"):
            return "text"
        
        first_span = first_line["spans"][0]
        font_size = first_span["size"]
        font_flags = first_span["flags"]
        text = first_span["text"].strip()
        
        # Header detection
        if font_flags & 16 or font_size > 14:  # Bold or large font
            if len(text) < 100 and not text.endswith('.'):
                return "header"
        
        # List detection
        if text.startswith(('•', '-', '*', '1.', '2.', '3.', '4.', '5.')):
            return "list_item"
        
        # Table detection (basic heuristic)
        if len(block["lines"]) == 1 and len(first_line["spans"]) > 3:
            return "table_row"
        
        # Default to paragraph
        return "paragraph"
    
    def extract_with_metadata(self, pdf_path: str) -> Document:
        """
        Extract text with comprehensive metadata about document structure.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Document with detailed structural metadata
        """
        try:
            doc = fitz.open(pdf_path)
            
            # Document-level metadata
            doc_metadata = doc.metadata
            page_count = len(doc)
            
            # Extract structured content
            structured_content = {
                "pages": [],
                "headers": [],
                "paragraphs": [],
                "lists": [],
                "tables": []
            }
            
            full_text = ""
            
            for page_num in range(page_count):
                page = doc[page_num]
                page_text = ""
                page_blocks = []
                
                blocks = page.get_text("dict")
                
                for block in blocks["blocks"]:
                    if "lines" in block:
                        block_text = self._extract_block_text(block)
                        block_type = self._classify_block_type(block)
                        
                        if block_text.strip():
                            page_text += block_text + "\n"
                            page_blocks.append({
                                "text": block_text.strip(),
                                "type": block_type,
                                "bbox": block.get("bbox", [])
                            })
                            
                            # Categorize content
                            if block_type == "header":
                                structured_content["headers"].append(block_text.strip())
                            elif block_type == "paragraph":
                                structured_content["paragraphs"].append(block_text.strip())
                            elif block_type == "list_item":
                                structured_content["lists"].append(block_text.strip())
                            elif block_type == "table_row":
                                structured_content["tables"].append(block_text.strip())
                
                structured_content["pages"].append({
                    "page_number": page_num + 1,
                    "text": page_text,
                    "blocks": page_blocks
                })
                
                full_text += f"\n--- Page {page_num + 1} ---\n" + page_text
            
            doc.close()
            
            if full_text.strip():
                document_name = os.path.splitext(os.path.basename(pdf_path))[0]
                document = Document(
                    page_content=full_text,
                    metadata={
                        "source": pdf_path,
                        "document": document_name,
                        "filename": os.path.basename(pdf_path),
                        "parser": "pymupdf_enhanced",
                        "page_count": page_count,
                        "doc_metadata": doc_metadata,
                        "structure": structured_content,
                        "headers_count": len(structured_content["headers"]),
                        "paragraphs_count": len(structured_content["paragraphs"]),
                        "lists_count": len(structured_content["lists"]),
                        "tables_count": len(structured_content["tables"])
                    }
                )
                return document
            else:
                print(f"Warning: No text extracted from {pdf_path}")
                return None
                
        except Exception as e:
            print(f"Error extracting text with metadata from {pdf_path}: {e}")
            return None
    
    def extract_header_aware_chunks(self, pdf_path: str, num_chunks: int = 10) -> List[Document]:
        """
        Split document into roughly equal chunks while respecting header boundaries.
        Chunks are created to end at header lines when possible, with preference for larger headers.
        
        Args:
            pdf_path: Path to the PDF file
            num_chunks: Number of chunks to create (default: 10)
            
        Returns:
            List of Document objects representing the chunks
        """
        try:
            doc = fitz.open(pdf_path)
            document_name = os.path.splitext(os.path.basename(pdf_path))[0]
            
            # First pass: extract all blocks with their positions and classify them
            all_blocks = []
            total_chars = 0
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                blocks = page.get_text("dict")
                
                for block_idx, block in enumerate(blocks["blocks"]):
                    if "lines" in block:  # Text block
                        block_text = self._extract_block_text(block)
                        block_type = self._classify_block_type(block)
                        
                        if block_text.strip():
                            # Calculate header priority (higher for more important headers)
                            header_priority = self._get_header_priority(block)
                            
                            block_info = {
                                "text": block_text.strip(),
                                "type": block_type,
                                "page": page_num + 1,
                                "position": len(all_blocks),
                                "char_count": len(block_text.strip()),
                                "header_priority": header_priority,
                                "is_header": block_type == "header",
                                "bbox": block.get("bbox", [])
                            }
                            all_blocks.append(block_info)
                            total_chars += len(block_text.strip())
            
            doc.close()
            
            if not all_blocks:
                print(f"Warning: No text blocks found in {pdf_path}")
                return []
            
            # Calculate target chunk size
            target_chunk_size = total_chars // num_chunks
            
            # Second pass: create chunks respecting header boundaries
            chunks = []
            current_chunk_blocks = []
            current_chunk_chars = 0
            
            for i, block in enumerate(all_blocks):
                current_chunk_blocks.append(block)
                current_chunk_chars += block["char_count"]
                
                # Check if we should end the chunk here
                should_end_chunk = False
                
                # Always end at the last block
                if i == len(all_blocks) - 1:
                    should_end_chunk = True
                
                # End chunk if we've reached target size and next block is a good breaking point
                elif current_chunk_chars >= target_chunk_size:
                    # Look ahead to find the best breaking point
                    best_break_point = self._find_best_break_point(all_blocks, i, target_chunk_size * 0.3)
                    
                    if best_break_point is not None and best_break_point <= i + 5:  # Don't look too far ahead
                        # Add blocks up to the break point
                        while i < best_break_point:
                            i += 1
                            if i < len(all_blocks):
                                current_chunk_blocks.append(all_blocks[i])
                                current_chunk_chars += all_blocks[i]["char_count"]
                        should_end_chunk = True
                    elif current_chunk_chars >= target_chunk_size * 1.5:  # Force break if chunk is too large
                        should_end_chunk = True
                
                # Create chunk if we should end here
                if should_end_chunk and current_chunk_blocks:
                    chunk_text = self._create_chunk_text(current_chunk_blocks)
                    
                    if chunk_text.strip():
                        chunk_doc = Document(
                            page_content=chunk_text,
                            metadata={
                                "source": pdf_path,
                                "document": document_name,
                                "filename": os.path.basename(pdf_path),
                                "chunk_id": len(chunks),
                                "chunk_type": "header_aware",
                                "parser": "pymupdf_header_aware",
                                "total_chunks": num_chunks,
                                "char_count": current_chunk_chars,
                                "block_count": len(current_chunk_blocks),
                                "start_page": current_chunk_blocks[0]["page"],
                                "end_page": current_chunk_blocks[-1]["page"],
                                "has_headers": any(b["is_header"] for b in current_chunk_blocks),
                                "header_count": sum(1 for b in current_chunk_blocks if b["is_header"])
                            }
                        )
                        chunks.append(chunk_doc)
                    
                    # Reset for next chunk
                    current_chunk_blocks = []
                    current_chunk_chars = 0
            
            # If we have fewer chunks than requested, it means the document is small
            if len(chunks) < num_chunks:
                print(f"Note: Created {len(chunks)} chunks instead of {num_chunks} (document may be smaller than expected)")
            
            print(f"Successfully created {len(chunks)} header-aware chunks from {pdf_path}")
            return chunks
            
        except Exception as e:
            print(f"Error creating header-aware chunks from {pdf_path}: {e}")
            return []
    
    def _get_header_priority(self, block) -> int:
        """
        Calculate header priority based on font size and formatting.
        Higher numbers indicate more important headers (better break points).
        
        Args:
            block: PyMuPDF text block dictionary
            
        Returns:
            Header priority (0 = not a header, higher = more important header)
        """
        if not block.get("lines") or not block["lines"]:
            return 0
        
        first_line = block["lines"][0]
        if not first_line.get("spans") or not first_line["spans"]:
            return 0
        
        first_span = first_line["spans"][0]
        font_size = first_span["size"]
        font_flags = first_span["flags"]
        text = first_span["text"].strip()
        
        # Not a header if text is too long or ends with period
        if len(text) > 100 or text.endswith('.'):
            return 0
        
        priority = 0
        
        # Font size contribution
        if font_size > 18:
            priority += 10
        elif font_size > 16:
            priority += 8
        elif font_size > 14:
            priority += 6
        elif font_size > 12:
            priority += 4
        
        # Font formatting contribution
        if font_flags & 16:  # Bold
            priority += 5
        if font_flags & 2:   # Italic
            priority += 2
        
        # Content-based hints
        if text.isupper():   # ALL CAPS
            priority += 3
        if text.startswith(('Chapter', 'Section', 'Part', 'Introduction', 'Conclusion')):
            priority += 4
        if any(char.isdigit() for char in text[:10]):  # Numbered headers
            priority += 2
        
        return priority
    
    def _find_best_break_point(self, blocks, current_index, max_lookahead_chars) -> int:
        """
        Find the best point to break a chunk within a reasonable distance.
        
        Args:
            blocks: List of all text blocks
            current_index: Current position in blocks
            max_lookahead_chars: Maximum characters to look ahead
            
        Returns:
            Index of best break point, or None if no good break point found
        """
        best_break = None
        best_priority = 0
        chars_looked_ahead = 0
        
        # Look ahead for headers
        for i in range(current_index + 1, min(len(blocks), current_index + 20)):
            chars_looked_ahead += blocks[i]["char_count"]
            
            # Don't look too far ahead in terms of characters
            if chars_looked_ahead > max_lookahead_chars:
                break
            
            if blocks[i]["is_header"] and blocks[i]["header_priority"] > best_priority:
                best_priority = blocks[i]["header_priority"]
                best_break = i
        
        return best_break
    
    def _create_chunk_text(self, blocks) -> str:
        """
        Create formatted text from a list of blocks.
        
        Args:
            blocks: List of block dictionaries
            
        Returns:
            Formatted chunk text
        """
        chunk_text = ""
        current_page = None
        
        for block in blocks:
            # Add page separator if we're on a new page
            if current_page is not None and block["page"] != current_page:
                chunk_text += f"\n--- Page {block['page']} ---\n"
            current_page = block["page"]
            
            # Add the block text with appropriate spacing
            if block["is_header"]:
                chunk_text += f"\n{block['text']}\n"
            else:
                chunk_text += f"{block['text']}\n\n"
        
        return chunk_text.strip()
