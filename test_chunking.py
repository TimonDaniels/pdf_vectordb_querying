#!/usr/bin/env python3
"""
Test script for the PyMuPDF parser.
This script demonstrates how to use the new PyMuPDFParser class.
"""

import os
import glob
import sys
import asyncio
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
import openai
from openai import AsyncOpenAI
from dotenv import load_dotenv
from typing import List, Tuple, Dict, Any
import json
from pdf_parser import PyMuPDFParser


def chucking_prompt(chunk_content: str):
    return f"""You are an expert at intelligently chunking text for document processing and retrieval systems. The input will be a text in dutch on the vision and plans of political parties in the Netherlands.

Your task is to divide the following text into semantically coherent chunks of approximately 1000 characters each and analyse weather the chunk is related to the vision of the party (what they want to achieve) or if the chunk is more related to the strategy (how they want to achieve their goals). Follow these guidelines:

1. **Chunk Size**: Aim for chunks of ~1000 characters (can range from 800-1200 characters)
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


def process_single_pdf(pdf_file: str, token_limit: int = 15000) -> Tuple[str, bool, Dict[str, Any]]:
    """
    Process a single PDF file with header-aware chunking.
    This function will be called by multiprocessing workers.
    
    Args:
        pdf_file: Path to the PDF file
        token_limit: Maximum tokens per chunk
        
    Returns:
        Tuple of (pdf_file, success, results_dict)
    """
    try:
        # Initialize parser in the worker process
        parser = PyMuPDFParser()
        
        if not os.path.exists(pdf_file):
            return pdf_file, False, {"error": f"PDF file not found: {pdf_file}"}
        
        # Extract text and calculate chunks needed
        doc = parser.extract_text(pdf_file)
        token_length = len(doc.page_content) / 4
        number_of_chunks = int(token_length / token_limit) + 1
        
        # Create header-aware chunks
        chunks = parser.extract_header_aware_chunks(pdf_file, number_of_chunks)
        
        if not chunks:
            return pdf_file, False, {"error": "No chunks created"}
        
        # Save chunks to files
        chunk_files = []
        for chunk_num, chunk in enumerate(chunks):
            chunk_file = pdf_file.replace('.pdf', f'_chunk_{chunk_num}.txt')
            with open(chunk_file, 'w', encoding='utf-8') as f:
                f.write(chunk.page_content + "\n\n")
            chunk_files.append(chunk_file)
        
        # Calculate statistics
        total_chars = sum(len(chunk.page_content) for chunk in chunks)
        avg_chunk_size = total_chars / len(chunks) if chunks else 0
        chunk_sizes = [len(chunk.page_content) for chunk in chunks]
        min_size, max_size = min(chunk_sizes), max(chunk_sizes)
        headers_in_chunks = sum(1 for chunk in chunks if chunk.metadata.get('has_headers', False))
        
        results = {
            "num_chunks": len(chunks),
            "chunk_files": chunk_files,
            "total_chars": total_chars,
            "avg_chunk_size": avg_chunk_size,
            "size_range": (min_size, max_size),
            "headers_in_chunks": headers_in_chunks,
            "token_estimates": [len(chunk.page_content) / 4 for chunk in chunks]
        }
        
        return pdf_file, True, results
        
    except Exception as e:
        return pdf_file, False, {"error": str(e)}


def header_aware_chunking_parallel(pdf_files: List[str], token_limit: int = 15000, max_workers: int = None) -> Dict[str, Any]:
    """
    Process multiple PDF files in parallel using multiprocessing.
    
    Args:
        pdf_files: List of PDF file paths
        token_limit: Maximum tokens per chunk
        max_workers: Maximum number of worker processes (default: CPU count)
        
    Returns:
        Dictionary with processing results
    """
    print("=" * 60)
    print("PARALLEL PDF PROCESSING")
    print("=" * 60)
    
    if not pdf_files:
        print("✗ No PDF files provided")
        return {}
    
    print(f"Processing {len(pdf_files)} PDF files in parallel...")
    
    if max_workers is None:
        max_workers = min(len(pdf_files), multiprocessing.cpu_count())
    
    start_time = time.time()
    
    try:
        # Use ProcessPoolExecutor for better resource management
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_pdf = {
                executor.submit(process_single_pdf, pdf_file, token_limit): pdf_file 
                for pdf_file in pdf_files
            }
            
            results = {}
            successful = 0
            failed = 0
            
            # Collect results as they complete
            for future in future_to_pdf:
                pdf_file, success, result_data = future.result()
                results[pdf_file] = result_data
                
                if success:
                    successful += 1
                    print(f"✓ {pdf_file}: {result_data['num_chunks']} chunks created")
                    print(f"  - Total chars: {result_data['total_chars']:,}")
                    print(f"  - Avg chunk size: {result_data['avg_chunk_size']:.0f} chars")
                    print(f"  - Size range: {result_data['size_range'][0]}-{result_data['size_range'][1]} chars")
                    print(f"  - Chunks with headers: {result_data['headers_in_chunks']}/{result_data['num_chunks']}")
                    print(f"  - Token estimates: {[f'{t:.0f}' for t in result_data['token_estimates']]}")
                else:
                    failed += 1
                    print(f"✗ {pdf_file}: {result_data.get('error', 'Unknown error')}")
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"\n📊 Summary:")
        print(f"  - Processed: {successful}/{len(pdf_files)} files successfully")
        print(f"  - Failed: {failed} files")
        print(f"  - Processing time: {processing_time:.2f} seconds")
        print(f"  - Average time per file: {processing_time/len(pdf_files):.2f} seconds")
        
        return {
            "successful": successful,
            "failed": failed,
            "processing_time": processing_time,
            "results": results
        }
        
    except Exception as e:
        print(f"✗ Error in parallel processing: {e}")
        return {"error": str(e)}


async def create_semantic_chunk_async(client: AsyncOpenAI, chunk_content: str, chunk_file: str, semaphore: asyncio.Semaphore) -> Tuple[str, bool, Any]:
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
            prompt = chucking_prompt(chunk_content)

            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at intelligent text chunking for document processing systems. You create semantically coherent chunks that preserve meaning and context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=16000
            )
            
            result = response.choices[0].message.content.strip()
            return chunk_file, True, result
            
        except Exception as e:
            return chunk_file, False, str(e)


async def semantic_chunking_async(chunk_file_patterns: List[str], max_concurrent: int = 5) -> Dict[str, Any]:
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
        load_dotenv('.env.local')
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
            create_semantic_chunk_async(client, content, chunk_file, semaphore)
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
                base_name = chunk_file.replace('.txt', '_semantic.txt')
                with open(base_name, 'w', encoding='utf-8') as f:
                    f.write(data)
                
                # Analyze the chunks
                try:
                    analyze_semantic_chunks(data)
                except Exception as e:
                    print(f"   ⚠ Analysis failed for {chunk_file}: {e}")
                
                all_results[chunk_file] = {
                    "success": True,
                    "output_file": base_name,
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

# ====================================================================================
# ========================== Synchronous Semantic Chunking ===========================
# ====================================================================================

def semantic_chunking_sync(chunk_file_regex: List[str]):
    """
    Synchronous semantic chunking using OpenAI GPT-4o-mini to create semantically coherent chunks
    of approximately 1000 characters while maintaining paragraph structure.
    """
    print("\n" + "=" * 60)
    print("SEMANTIC CHUNKING TEST (SYNC)")
    print("=" * 60)
    
    # Set up OpenAI client (assumes OPENAI_API_KEY is set in environment)
    try:
        load_dotenv('.env.local')
        client = openai.OpenAI()  # Uses OPENAI_API_KEY environment variable
        print("✓ OpenAI client initialized")
    except Exception as e:
        print(f"✗ Error initializing OpenAI client: {e}")
        print("Make sure OPENAI_API_KEY environment variable is set")
        return
    
    # Find all chunk files
    chunk_files = []
    for file_regex in chunk_file_regex:
        chunk_files += glob.glob(file_regex)
    
    if not chunk_files:
        print("✗ No chunk files found. Run header_aware_chunking() first to generate chunks.")
        return
    
    print(f"Found {len(chunk_files)} chunk files to process")
    
    # Process each chunk file
    for chunk_file in sorted(chunk_files):
        print(f"\n--- Processing {chunk_file} ---")
        
        try:
            # Read the chunk content
            with open(chunk_file, 'r', encoding='utf-8') as f:
                chunk_content = f.read().strip()
            
            if not chunk_content:
                print(f"   ✗ Empty file: {chunk_file}")
                continue
            
            print(f"   ✓ Loaded {len(chunk_content)} characters")
            
            # Create semantic chunks using OpenAI
            semantic_chunks = create_semantic_chunks(client, chunk_content)
            
            if semantic_chunks:
                print(f"   ✓ Created semantic chunks")
                
                # Save semantic chunks
                base_name = chunk_file.replace('.txt', '_semantic.txt')
                with open(base_name, 'w', encoding='utf-8') as f:
                    f.write(semantic_chunks)
                
                print(f"   ✓ Saved semantic chunks to {base_name}")
                
                # Analyze the chunks
                analyze_semantic_chunks(semantic_chunks)
            else:
                print(f"   ✗ Failed to create semantic chunks")
                
        except Exception as e:
            print(f"   ✗ Error processing {chunk_file}: {e}")


def create_semantic_chunks(client, text_content):
    """
    Use OpenAI to create semantic chunks from text content.
    
    Args:
        client: OpenAI client
        text_content: Text to be chunked
        
    Returns:
        String containing all chunks separated by special delimiters
    """
    
    # Create the prompt for semantic chunking
    prompt = chucking_prompt(text_content)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert at intelligent text chunking for document processing systems. You create semantically coherent chunks that preserve meaning and context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Low temperature for consistent chunking
            max_tokens=16000   # Adjust based on input size
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"   ✗ OpenAI API error: {e}")
        return None


def analyze_semantic_chunks(chunked_text):
    """
    Analyze the semantic chunks to provide statistics.
    
    Args:
        chunked_text: JSON string with chunks or text with chunks separated by delimiters
    """
    
    chunks = None
    try:
        # Try to parse as JSON first
        if chunked_text.strip().startswith('[') or chunked_text.strip().startswith('{'):
            chunks = json.loads(chunked_text)
            if isinstance(chunks, dict):
                chunks = [chunks]  # Convert single chunk to list
        else:
            # Try to extract JSON from the response (sometimes wrapped in ```json```)
            import re
            json_match = re.search(r'```json\s*(\[.*?\])\s*```', chunked_text, re.DOTALL)
            if json_match:
                chunks = json.loads(json_match.group(1))
            else:
                # Fallback to old eval method (deprecated)
                chunks = eval(chunked_text[7:-3])
    except Exception as e:
        print(f"   ✗ Error parsing semantic chunks: {e}")
        print(f"   Raw output (first 200 chars): {chunked_text[:200]}...")
        return
    
    if not chunks:
        print("   ✗ No chunks found in output")
        return
    
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
    
    # Count chunks in target range (800-1200 characters)
    target_range_count = sum(1 for length in chunk_lengths if 800 <= length <= 1200)
    target_percentage = (target_range_count / len(chunk_lengths)) * 100
    
    print(f"   ✓ Analysis Results:")
    print(f"     - Total chunks: {len(chunk_lengths)}")
    print(f"     - Average length: {avg_length:.0f} characters")
    print(f"     - Length range: {min_length} - {max_length} characters")
    print(f"     - Chunks in target range (800-1200): {target_range_count}/{len(chunk_lengths)} ({target_percentage:.1f}%)")
    
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



if __name__ == "__main__":
    # Check for command line arguments

    pdf_files = [
        "data/CDA.pdf",
        "data/CU.pdf",
    ]
    chunk_file_regex = [
        "data/BBB_chunk_0.txt",
        "data/BBB_chunk_1.txt",
        "data/BBB_chunk_2.txt",
        "data/BBB_chunk_5.txt",
        "data/CDA_chunk_*.txt",
        "data/CU_chunk_*.txt",
    ]
    if len(sys.argv) > 1:
        if "--semantic-only" in sys.argv:
            print("Running Semantic Chunking Only")
            print("=" * 50)
            semantic_chunking_sync(pdf_files)
        elif "--async-only" in sys.argv:
            print("Running Async Semantic Chunking Only")
            print("=" * 50)
            asyncio.run(semantic_chunking_async(chunk_file_regex, max_concurrent=5))
    else:
        print("Testing PyMuPDF Parser with Parallel & Async Processing")
        print("=" * 60)
        
        # Test parallel PDF processing
        header_aware_chunking_parallel(pdf_files)
        
        # Test async semantic chunking
        asyncio.run(semantic_chunking_async(chunk_file_regex, max_concurrent=5))
        
        print("\n" + "=" * 60)
        print("All tests completed!")
        print("\nAvailable options:")
        print("  --semantic-only    : Run only semantic chunking")
        print("  --async-only      : Run only async semantic chunking")
