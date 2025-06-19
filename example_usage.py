"""
Example usage of the PDF processor system with multiple embedding models.
This script shows how to use the PDFProcessor class for different tasks.
"""

from chunker import RecursiveTextChunker
from pdf_parser import LayoutPDFParser, PyPDFParser
from vectorstore import VectorStore


def example_usage():
    """
    Example of how to use the PDF processing system with different embedding models.
    """
    # Initialize the processor
    processor = VectorStore()
    
    # Show available models
    print("Supported embedding models:")
    for model_name in processor.SUPPORTED_EMBEDDINGS.keys():
        print(f"  - {model_name}")
    print()
    
    # Check existing databases
    available_dbs = processor.get_available_databases()
    if available_dbs:
        print(f"Available databases: {available_dbs}")
        processor.list_databases_info()
    else:
        print("No databases found. Creating database with HuggingFace MiniLM model...")
        # Create database with a specific embedding model
        vectorstore = processor.create_database_with_model("huggingface_minilm")
        if vectorstore is None:
            print("Failed to create database. Check your PDF directory.")
            return
      # Example queries
    example_queries = [
        "climate change and environment",
        "technology and innovation", 
        "healthcare and medicine",
        "education and learning",
        "economic growth and development"
    ]
    
    # Use the first available model for searches
    model_to_use = available_dbs[0] if available_dbs else "huggingface_minilm"
    
    print("\n" + "="*60)
    print(f"EXAMPLE SEARCHES using {model_to_use}")
    print("="*60)
    
    for query in example_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 40)
        results = processor.search_with_model(query, model_to_use, k=2)
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['document']}: {result['content'][:100]}...")
            print(f"   (Score: {result['similarity_score']:.3f}, Model: {result['embedding_model']})")
        
        print()


def search_interactive():
    """
    Interactive search function with model selection.
    """
    processor = VectorStore()
    
    # Check available databases
    available_dbs = processor.get_available_databases()
    
    if not available_dbs:
        print("No databases found. Creating one with HuggingFace MiniLM...")
        vectorstore = processor.create_database_with_model("huggingface_minilm")
        if vectorstore is None:
            print("Failed to create database. Check your PDF directory.")
            return
        available_dbs = ["huggingface_minilm"]
    
    print("Interactive Search Mode")
    print("Available embedding models:")
    for i, model in enumerate(available_dbs, 1):
        print(f"  {i}. {model}")
    
    # Let user choose model
    while True:
        try:
            choice = input(f"\nSelect model (1-{len(available_dbs)}) or press Enter for {available_dbs[0]}: ").strip()
            if not choice:
                selected_model = available_dbs[0]
                break
            else:
                model_idx = int(choice) - 1
                if 0 <= model_idx < len(available_dbs):
                    selected_model = available_dbs[model_idx]
                    break
                else:
                    print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    print(f"\nUsing model: {selected_model}")
    print("Enter your search queries to find relevant content in the documents.")
    print("Type 'quit' to exit.\n")
    while True:
        query = input("Your query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not query:
            continue

        results = processor.search_with_model(query, selected_model, k=3)
        print(f"\nResults for: '{query}' (using {selected_model})")
        print("-" * 50)
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"\n{i}. Document: {result['document']}")
                print(f"   File: {result['filename']}")
                print(f"   Chunk ID: {result['chunk_id']}")
                print(f"   Relevance: {result['similarity_score']:.3f}")
                print(f"   Model: {result['embedding_model']}")
                print(f"   Content: {result['content'][:200]}...")
        else:
            print("No relevant content found.")
        
        print("\n" + "="*50)


def database_management_demo():
    """
    Demonstrate database management features.
    """
    processor = VectorStore()
    
    print("Database Management Demo")
    print("========================")
    
    # Show current databases
    existing_dbs = processor.get_available_databases()
    if existing_dbs:
        print("Existing databases:")
        processor.list_databases_info()
    else:
        print("No existing databases found.")
    
    # Find models without databases
    all_models = list(processor.SUPPORTED_EMBEDDINGS.keys())
    available_models = [model for model in all_models if model not in existing_dbs]
    
    if not available_models:
        print("\nAll supported embedding models already have databases.")
        print("You can reset a database using processor.reset_database(model_name) if needed.")
        return
    
    # Show available models for database creation
    print(f"\nEmbedding models available for database creation:")
    for i, model_name in enumerate(available_models, 1):
        config = processor.SUPPORTED_EMBEDDINGS[model_name]
        print(f"  {i}. {model_name}: {config['kwargs']}")
    
    # Let user choose which model to create database for
    while True:
        try:
            choice = input(f"\nSelect model to create database (1-{len(available_models)}) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                print("Cancelled database creation.")
                return
            
            model_idx = int(choice) - 1
            if 0 <= model_idx < len(available_models):
                selected_model = available_models[model_idx]
                break
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(available_models)}.")
        except ValueError:
            print("Invalid input. Please enter a number or 'q' to quit.")
    
    # Create database with selected model
    print(f"\nCreating database with {selected_model}...")
    
    # Special note for OpenAI model
    if selected_model == "openai":
        print("Note: This requires OPENAI_API_KEY in your .env.local file")
        confirm = input("Continue? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Database creation cancelled.")
            return
    
    # Create the database
    # parser = PyPDFParser()
    parser = LayoutPDFParser()
    chunker = RecursiveTextChunker()
    vectorstore = processor.create_database_with_model(selected_model, parser=parser, chunker=chunker)
    
    if vectorstore:
        print(f"\n✓ Successfully created database with {selected_model}!")
        print("\nUpdated database list:")
        processor.list_databases_info()
          # Optional: Test the new database
        test_choice = input(f"\nTest the new {selected_model} database with a sample query? (y/n): ").strip().lower()
        if test_choice == 'y':
            test_query = "technology innovation"
            print(f"\nTesting with query: '{test_query}'")
            results = processor.search_with_model(test_query, selected_model, k=2)
            
            if results:
                print(f"Found {len(results)} results:")
                for i, result in enumerate(results, 1):
                    print(f"  {i}. {result['document']} (score: {result['similarity_score']:.3f})")
                    print(f"     {result['content'][:100]}...")
            else:
                print("No results found for test query.")
    else:
        print(f"\n✗ Failed to create database with {selected_model}.")
        print("Check the error messages above for details.")


if __name__ == "__main__":
    print("PDF Processor Demo Options:")
    print("1. Basic example usage")
    print("2. Interactive search")
    print("4. Database management demo")
    
    choice = input("\nSelect option (1-3) or press Enter for interactive search: ").strip()
    
    if choice == "1":
        example_usage()
    elif choice == "3":
        database_management_demo()
    else:
        # Default to interactive search
        search_interactive()
