import asyncio
import glob
import json
import os
import time
from typing import Any, Dict, List, Protocol, Tuple

from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import AsyncOpenAI

from src.utils import parse_json_text


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


class SemanticChunker(Protocol):

    def chucking_prompt(self, chunk_content: str):
        return f"""You are an expert at intelligently chunking text for document processing and retrieval systems. The input will be a text in dutch on the vision and plans of political parties in the Netherlands.

    Your task is to divide the following text into semantically coherent chunks of approximately 300 characters each and analyse weather the chunk is related to the vision of the party (what they want to achieve) or if the chunk is more related to the strategy (how they want to achieve their goals). Follow these guidelines:

    1. **Chunk Size**: Aim for chunks of ~300 characters 
    2. **Semantic Coherence**: Each chunk should contain related information that makes sense together
    3. **Paragraph Preservation**: Try to keep complete paragraphs together when possible
    4. **Natural Breaks**: Split at natural boundaries like:
    - End of paragraphs
    - Before new topics or sections
    - After complete thoughts or concepts
    - Before headers or subheadings
    5. **Content analysis**: Is the chunk related to the vision of the party (what they want to achieve) or is it more related to the strategy (how they want to achieve their goals). 
    6. **Output Format**: 
    - Make a JSON format with each chunk as a separate entry and contains the following information:
            - `chunk_id`: Unique identifier for the chunk
            - `content`: The text content of the chunk
            - 'type': 'vision' or 'strategy' depending on the content
            - `page_number`: The page number from which the chunk was extracted (if applicable)
            - `header`: The header or title of the section, else put "No header" 
    - Do not add any commentary or explanations
    7. **Quality**: Ensure each chunk can stand alone and be meaningful for information retrieval. 
    8. **Relevance**: Don't return chunks that are not relevant for capturing the vision or plans of the political party. Think about things like, dankwoord, inhoudsopgave, voorwoord, etc.

    Here is the text to chunk:

    {chunk_content}

    Now provide the semantically chunked version using the specified format:"""


    async def create_semantic_chunk_async(self, client: AsyncOpenAI, chunk_content: str, chunk_file: str, semaphore: asyncio.Semaphore) -> Tuple[str, bool, Any]:
        """
        Async function to create semantic chunks for a single chunk file.
        
        Args:
            client: Async OpenAI client
            chunk_content: Text content to chunk
            chunk_file: Path to the chunk file
            semaphore: Semaphore to limit concurrent requests
            
        Returns:
            Tuple of (chunk_file, success, result)
        """
        async with semaphore:  # Limit concurrent API calls
            try:
                prompt = self.chucking_prompt(chunk_content)

                response = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an expert at intelligent text chunking for document processing systems. You create semantically coherent chunks that preserve meaning and context."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=16000
                )

                result = parse_json_text(response.choices[0].message.content.strip(), "semantic_chunks")
                return chunk_file, True, result
                
            except Exception as e:
                return chunk_file, False, str(e)


    async def chunk_text_from_files(self, chunk_file_patterns: List[str], max_concurrent: int = 25) -> Dict[str, Any]:
        """
        Async function to process multiple chunk files for semantic chunking.
        
        Args:
            chunk_file_patterns: List of glob patterns for chunk files
            max_concurrent: Maximum number of concurrent API calls
            
        Returns:
            Dictionary with processing results
        """
        print("\n" + "=" * 60)
        print("ASYNC SEMANTIC CHUNKING")
        print("=" * 60)
        
        # Set up OpenAI client
        try:
            client = AsyncOpenAI()
            print("✓ Async OpenAI client initialized")
        except Exception as e:
            print(f"✗ Error initializing OpenAI client: {e}")
            return {"error": str(e)}
        
        # Find all chunk files
        chunk_files = []
        for pattern in chunk_file_patterns:
            chunk_files.extend(glob.glob(pattern))
        
        if not chunk_files:
            print("✗ No chunk files found.")
            return {"error": "No chunk files found"}
        
        print(f"Found {len(chunk_files)} chunk files to process")
        print(f"Max concurrent API calls: {max_concurrent}")
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        # Read all chunk files
        chunk_data = []
        for chunk_file in sorted(chunk_files):
            try:
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                if content:
                    chunk_data.append((chunk_file, content))
                else:
                    print(f"   ⚠ Empty file: {chunk_file}")
            except Exception as e:
                print(f"   ✗ Error reading {chunk_file}: {e}")
        
        if not chunk_data:
            print("✗ No valid chunk files to process")
            return {"error": "No valid chunk files"}
        
        print(f"Processing {len(chunk_data)} valid chunk files...")
        start_time = time.time()
        
        try:
            # Create tasks for all chunk files
            tasks = [
                self.create_semantic_chunk_async(client, content, chunk_file, semaphore)
                for chunk_file, content in chunk_data
            ]
            
            # Process all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            successful = 0
            failed = 0
            all_results = {}
            
            for result in results:
                if isinstance(result, Exception):
                    print(f"   ✗ Task failed with exception: {result}")
                    failed += 1
                    continue
                
                chunk_file, success, data = result
                
                if success:
                    successful += 1
                    print(f"   ✓ {chunk_file}: Semantic chunking completed")
                    
                    # Save semantic chunks
                    file_name = os.path.basename(chunk_file.replace('.txt', '_semantic.json'))
                    with open(os.path.join("data/semantic_chunks", file_name), 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    
                    # Analyze the chunks
                    try:
                        self.analyze_semantic_chunks(data)
                    except Exception as e:
                        print(f"   ⚠ Analysis failed for {chunk_file}: {e}")
                    
                    all_results[chunk_file] = {
                        "success": True,
                        "output_file": file_name,
                        "data": data
                    }
                else:
                    failed += 1
                    print(f"   ✗ {chunk_file}: {data}")
                    all_results[chunk_file] = {
                        "success": False,
                        "error": data
                    }
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            print(f"\n📊 Async Processing Summary:")
            print(f"  - Processed: {successful}/{len(chunk_data)} files successfully")
            print(f"  - Failed: {failed} files")
            print(f"  - Total processing time: {processing_time:.2f} seconds")
            print(f"  - Average time per file: {processing_time/len(chunk_data):.2f} seconds")
            
            return {
                "successful": successful,
                "failed": failed,
                "processing_time": processing_time,
                "results": all_results
            }
            
        except Exception as e:
            print(f"✗ Error in async processing: {e}")
            return {"error": str(e)}
        
        finally:
            await client.close()

    def analyze_semantic_chunks(self, chunks):
        # Calculate statistics
        chunk_lengths = []
        for chunk in chunks:
            if isinstance(chunk, dict) and 'content' in chunk:
                chunk_lengths.append(len(chunk['content']))
            elif isinstance(chunk, str):
                chunk_lengths.append(len(chunk))
            else:
                print(f"   ⚠ Unknown chunk format: {type(chunk)}")
        
        if not chunk_lengths:
            print("   ✗ No valid chunks found")
            return
        
        total_chars = sum(chunk_lengths)
        avg_length = total_chars / len(chunk_lengths)
        min_length = min(chunk_lengths)
        max_length = max(chunk_lengths)
        
        print(f"   ✓ Analysis Results:")
        print(f"     - Total chunks: {len(chunk_lengths)}")
        print(f"     - Average length: {avg_length:.0f} characters")
        print(f"     - Length range: {min_length} - {max_length} characters")
        
        # Show first chunk preview
        if chunks:
            first_chunk = chunks[0]
            if isinstance(first_chunk, dict) and 'content' in first_chunk:
                preview = first_chunk['content'][:150]
                header = first_chunk.get('header', 'No header')
                print(f"     - First chunk header: {header}")
                print(f"     - First chunk preview: {preview}...")
            else:
                print(f"     - First chunk preview: {str(first_chunk)[:150]}...")

