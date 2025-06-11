"""
Test script to diagnose and fix database connection issues.
"""

from pdf_processor import PDFProcessor


def test_database_connection():
    """Test if the database can be loaded and queried."""
    print("Testing database connection...")
    
    processor = PDFProcessor()
    
    # Try to load existing database
    vectorstore = processor.load_existing_database()
    
    if vectorstore is None:
        print("Failed to load existing database. Let's try to recreate it.")
        
        # Reset the database
        processor.reset_database()
        
        # Recreate the database
        print("Recreating database from PDFs...")
        vectorstore = processor.process_pdfs_and_create_db()
        
        if vectorstore is None:
            print("Failed to create database. Check your PDF files and dependencies.")
            return False
    
    # Test with a simple query
    print("Testing search functionality...")
    try:
        results = processor.search_similar_content("test query", k=1)
        if results:
            print(f"✓ Database connection successful! Found {len(results)} results.")
            print(f"Sample result: {results[0]['party']}")
            return True
        else:
            print("✓ Database connection successful but no results found.")
            return True
    except Exception as e:
        print(f"✗ Database search failed: {e}")
        return False


def interactive_search():
    """Run interactive search if database is working."""
    processor = PDFProcessor()
    
    print("\nInteractive Search Mode")
    print("Enter your questions to find relevant political positions.")
    print("Type 'quit' to exit.\n")
    
    while True:
        query = input("Your question: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not query:
            continue
            
        print(f"\nSearching for: '{query}'")
        print("-" * 50)
        
        results = processor.search_similar_content(query, k=3)
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"\n{i}. Party: {result['party']}")
                print(f"   Relevance: {result['similarity_score']:.3f}")
                print(f"   Content: {result['content'][:200]}...")
        else:
            print("No relevant content found.")
        
        print("\n" + "="*50)


if __name__ == "__main__":
    # Test database connection first
    if test_database_connection():
        print("\nDatabase is working! Starting interactive search...")
        interactive_search()
    else:
        print("\nDatabase connection failed. Please check the error messages above.")
