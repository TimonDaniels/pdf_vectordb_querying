"""
Flask web application for PDF vector search functionality.
This provides a web interface for searching through PDF documents using multiple embedding models.
"""

from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
import sys
import os
import threading

# Add parent directory to path to import pdf_processor
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from src.vectorstore import VectorStore
from src.pdf_parser import PDFParser, PyPDFParser
from src.chunker import TextChunker, RecursiveTextChunker
from src.query_expander import QueryExpander
from src.synthesis import synthesize_opinion_fit
from src.utils import find_project_root


app = Flask(__name__)

# Initialize the processor globally with correct paths
load_dotenv(os.path.join(find_project_root(), ".env.local"))
data_dir = os.path.join(parent_dir, "data")
chroma_db_dir = os.path.join(parent_dir, "chroma_db")
processor = VectorStore(pdf_directory=data_dir, base_db_directory=chroma_db_dir)
creation_lock = threading.Lock()
query_expander = QueryExpander()
database_creation_status = {}


def create_database_background(model_name: str, parser: PDFParser, chunker: TextChunker):
    """Create database in background thread."""
    def create_db():
        with creation_lock:
            database_creation_status[model_name] = 'creating'
        
        try:
            print(f"Starting background creation of database for {model_name}...")
            vectorstore = processor.create_database_with_model(model_name, parser, chunker)

            with creation_lock:
                if vectorstore:
                    database_creation_status[model_name] = 'available'
                    print(f"✓ Successfully created database for {model_name}")
                else:
                    database_creation_status[model_name] = 'failed'
                    print(f"✗ Failed to create database for {model_name}")
        except Exception as e:
            with creation_lock:
                database_creation_status[model_name] = 'failed'
            print(f"✗ Error creating database for {model_name}: {str(e)}")
    
    # Start creation in background thread
    thread = threading.Thread(target=create_db)
    thread.daemon = True
    thread.start()


def get_model_status_with_cache(model_name):
    """
    Get comprehensive status of a model including cache information.
    
    Returns:
        'not_created': Database doesn't exist
        'available': Database exists but not loaded in cache
        'loaded': Database exists and is loaded in cache
        'creating': Database is being created
        'failed': Database creation failed
    """
    # Check if it's being created or failed
    if model_name in database_creation_status:
        return database_creation_status[model_name]
    
    available_dbs = processor.get_available_databases()
    if model_name not in available_dbs:
        return 'not_created'
    
    # Check if it's in cache (loaded)
    cache_stats = processor.get_cache_stats()
    if model_name in cache_stats['cached_vectorstores']:
        return 'loaded'
    
    return 'available'


@app.route('/')
def index():
    """Main page with search interface."""
    # Get all supported models
    all_models = list(processor.SUPPORTED_EMBEDDINGS.keys())
    
    # Create model info with status
    model_info = []
    for model in all_models:
        status = get_model_status_with_cache(model)
        
        model_info.append({
            'name': model,
            'status': status
        })
    
    return render_template('index.html', model_info=model_info)


@app.route('/search', methods=['POST'])
def search():
    """Handle search requests."""
    data = request.get_json()
    print(f"Received request data: {data}")  # Debug logging
    
    if not data:
        print("Error: No JSON data received")  # Debug logging
        return jsonify({'error': 'No data received'}), 400
        
    query = data.get('query', '').strip()
    model = data.get('model', '')
    use_expansion = data.get('use_expansion', False)
    
    print(f"Parsed - Query: '{query}', Model: '{model}', Expansion: {use_expansion}")  # Debug logging
    
    if not query:
        print("Error: Query is empty")  # Debug logging
        return jsonify({'error': 'Query cannot be empty'}), 400
    
    if not model:
        print("Error: Model not selected")  # Debug logging
        return jsonify({'error': 'Model must be selected'}), 400
    
    # Check if model exists in supported models
    if model not in processor.SUPPORTED_EMBEDDINGS:
        print(f"Error: Invalid model '{model}'")  # Debug logging
        return jsonify({'error': 'Invalid model selected'}), 400
        
    # Handle query expansion
    expansion_info = None
    final_query = query
    
    if use_expansion:
        try:
            expansion_result = query_expander.expand_query(query)
            expansion_info = expansion_result
            if expansion_result['expansion_used']:
                final_query = expansion_result['expanded_query']
                print(f"Original query: {query}")
                print(f"Expanded query: {final_query}")
        except Exception as e:
            print(f"Query expansion failed: {e}")
            # Continue with original query if expansion fails
    
    # Check if database exists for this model
    available_dbs = processor.get_available_databases()
    if model not in available_dbs:
        # Check if database is being created
        if model in database_creation_status and database_creation_status[model] == 'creating':
            return jsonify({'status': 'in_progress', 'message': f'Database for {model} is being created. Please wait and try again.'}), 202
        
        # Start creating database in background
        parser = PyPDFParser()
        chunker = RecursiveTextChunker()
        create_database_background(model, parser, chunker)
        return jsonify({'status': 'in_progress', 'message': f'Database for {model} is being created. Please try again in a few moments.'}), 202
    
    try:
        # Perform search with the final query (original or expanded)
        results = processor.search_with_model(final_query, model, k=3)
        
        # Format results for JSON response
        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_results.append({
                'rank': i,
                'document': result['document'],
                'filename': result['filename'],
                'chunk_id': result['chunk_id'],
                'type': result.get('type'),
                'page_number': result.get('page_number'),
                'header': result.get('header'),
                'similarity_score': round(result['similarity_score'], 3),
                'embedding_model': result['embedding_model'],
                'content': result['content']
            })
        
        # Generate synthesis if results are available
        synthesis = None
        if formatted_results:
            try:
                # Convert formatted results to the format expected by synthesis module
                synthesis_input = []
                for result in formatted_results:
                    synthesis_input.append({
                        'content': result['content'],
                        'metadata': {
                            'source': result['filename'],
                            'page': result.get('page_number'),
                            'chunk': result['chunk_id']
                        }
                    })
                
                synthesis = synthesize_opinion_fit(query, synthesis_input)
            except Exception as e:
                print(f"Synthesis generation failed: {e}")
                # Continue without synthesis if it fails
        
        response_data = {
            'query': query,
            'model': model,
            'results': formatted_results,
            'synthesis': synthesis
        }
        
        # Add expansion info if expansion was used
        if expansion_info:
            response_data['expansion_info'] = expansion_info
            if expansion_info['expansion_used']:
                response_data['expanded_query'] = final_query
        
        return jsonify(response_data)
    
    except Exception as e:
        return jsonify({'error': f'Search failed: {str(e)}'}), 500



@app.route('/models')
def get_models():
    """Get all available models with their status."""
    all_models = list(processor.SUPPORTED_EMBEDDINGS.keys())
    
    model_info = []
    for model in all_models:
        status = get_model_status_with_cache(model)
        
        model_info.append({
            'name': model,
            'status': status
        })
    
    return jsonify({'models': model_info})


@app.route('/models/<model_name>/status')
def get_model_status(model_name):
    """Get status of a specific model."""
    if model_name not in processor.SUPPORTED_EMBEDDINGS:
        return jsonify({'error': 'Invalid model'}), 400
    
    status = get_model_status_with_cache(model_name)
    
    return jsonify({
        'model': model_name,
        'status': status
    })


@app.route('/models/<model_name>/create', methods=['POST'])
def create_model_database(model_name):
    """Trigger database creation for a specific model."""
    if model_name not in processor.SUPPORTED_EMBEDDINGS:
        return jsonify({'error': 'Invalid model'}), 400
    
    available_dbs = processor.get_available_databases()
    if model_name in available_dbs:
        return jsonify({'message': f'Database for {model_name} already exists'}), 200
    
    if model_name in database_creation_status and database_creation_status[model_name] == 'creating':
        return jsonify({'message': f'Database for {model_name} is already being created'}), 200
    
    # Start creating database in background
    create_database_background(model_name)
    return jsonify({'message': f'Started creating database for {model_name}'}), 202


@app.route('/cache/status')
def get_cache_status():
    """Get cache statistics."""
    stats = processor.get_cache_stats()
    return jsonify(stats)


@app.route('/cache/clear', methods=['POST'])
def clear_cache():
    """Clear cache for all models or specific model."""
    data = request.get_json() or {}
    model_name = data.get('model_name')
    
    processor.clear_cache(model_name)
    
    if model_name:
        return jsonify({'message': f'Cache cleared for {model_name}'})
    else:
        return jsonify({'message': 'All caches cleared'})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
