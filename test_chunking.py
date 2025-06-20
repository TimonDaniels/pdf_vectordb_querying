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
from typing import List, Tuple, Dict, Any

from dotenv import load_dotenv
from src.chunker import SemanticChunker
from src.pdf_parser import PyMuPDFParser
from src.utils import find_project_root

load_dotenv(os.path.join(find_project_root(), ".env.local"))


def process_single_pdf(pdf_file: str, token_limit) -> Tuple[str, bool, Dict[str, Any]]:
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


def header_aware_chunking_parallel(pdf_regex_files: List[str], token_limit: int = 3000, max_workers: int = None) -> Dict[str, Any]:
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
    
    pdf_files = [] 
    for pattern in pdf_regex_files:
        pdf_files.extend(glob.glob(pattern))
    
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



if __name__ == "__main__":
    # Check for command line arguments
    pdf_files = [
        "data/*.pdf",
    ]
    chunk_file_regex = [
        "data/VVD_chunk_23.txt",
    ]

    if len(sys.argv) > 1:
        if "--semantic-only" in sys.argv:
            print("Running Async Semantic Chunking Only")
            print("=" * 50)
            asyncio.run(SemanticChunker.chunk_text_from_files(chunk_file_regex))
    else:
        print("Testing PyMuPDF Parser with Parallel & Async Processing")
        print("=" * 60)
        
        # Test parallel PDF processing
        # header_aware_chunking_parallel(pdf_files)
        
        # Test async semantic chunking
        asyncio.run(SemanticChunker.chunk_text_from_files(chunk_file_regex))

        print("\n" + "=" * 60)
        print("All tests completed!")
        print("\nAvailable options:")
        print("  --semantic-only    : Run only semantic chunking")
        print("  --async-only      : Run only async semantic chunking")
