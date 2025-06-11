"""
Example usage of the PDF processor system.
This script shows how to use the PDFProcessor class for different tasks.
"""

from pdf_processor import PDFProcessor


def example_usage():
    """
    Example of how to use the PDF processing system.
    """
    # Initialize the processor
    processor = PDFProcessor()
    
    # Process all PDFs and create the database (only needs to be done once)
    print("Processing PDFs and creating vector database...")
    vectorstore = processor.process_pdfs_and_create_db()
    
    # Example queries
    example_queries = [
        "climate change and environment protection",
        "immigration and integration policies", 
        "healthcare and social security",
        "education and youth development",
        "economic growth and job creation"
    ]
    
    print("\n" + "="*60)
    print("EXAMPLE SEARCHES")
    print("="*60)
    
    for query in example_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 40)
        
        results = processor.search_similar_content(query, k=2)
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['party']}: {result['content'][:150]}...")
            print(f"   (Score: {result['similarity_score']:.3f}, Chunk: {result['chunk_id']})")
        
        print()


def search_interactive():
    """
    Interactive search function.
    """
    processor = PDFProcessor()
    
    print("Interactive Search Mode")
    print("Enter your questions or opinions to find relevant political positions.")
    print("Type 'quit' to exit.\n")
    
    while True:
        query = "Ik wil graag dat de markten zo vrij mogelijk zijn. Maar ook dat er geen misbruik van gemaakt wordt. Wat is de mening van de partijen hierover?"

        results = processor.search_similar_content(query, k=3)
        print(f"\nResults for: '{query}'")
        print("-" * 50)
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"\n{i}. Party: {result['party']}")
                print(f"   File: {result['filename']}")
                print(f"   Chunk ID: {result['chunk_id']}")
                print(f"   Relevance: {result['similarity_score']:.3f}")
                print(f"   Content: {result['content'][:250]}...")
        else:
            print("No relevant content found.")
        
        print("\n" + "="*50)


if __name__ == "__main__":
    # You can uncomment the function you want to run:
    
    # Run example searches
    # example_usage()
    
    # Run interactive search
    search_interactive()
