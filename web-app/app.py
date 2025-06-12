"""
Flask web application for PDF vector search functionality.
This provides a web interface for searching through PDF documents using multiple embedding models.
"""

from flask import Flask, render_template, request, jsonify
import sys
import os
import threading
import time

# Add parent directory to path to import pdf_processor
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from pdf_processor import PDFProcessor

app = Flask(__name__)

# Initialize the processor globally with correct paths
data_dir = os.path.join(parent_dir, "data")
chroma_db_dir = os.path.join(parent_dir, "chroma_db")
processor = PDFProcessor(pdf_directory=data_dir, base_db_directory=chroma_db_dir)

# Track database creation status
database_creation_status = {}
creation_lock = threading.Lock()


def create_database_background(model_name):
    """Create database in background thread."""
    def create_db():
        with creation_lock:
            database_creation_status[model_name] = 'creating'
        
        try:
            print(f"Starting background creation of database for {model_name}...")
            vectorstore = processor.create_database_with_model(model_name)
            
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
    query = data.get('query', '').strip()
    model = data.get('model', '')
    
    if not query:
        return jsonify({'error': 'Query cannot be empty'}), 400
    
    if not model:
        return jsonify({'error': 'Model must be selected'}), 400
    
    # Check if model exists in supported models
    if model not in processor.SUPPORTED_EMBEDDINGS:
        return jsonify({'error': 'Invalid model selected'}), 400
    
    # Check if database exists for this model
    available_dbs = processor.get_available_databases()
    if model not in available_dbs:
        # Check if database is being created
        if model in database_creation_status and database_creation_status[model] == 'creating':
            return jsonify({'status': 'in_progress', 'message': f'Database for {model} is being created. Please wait and try again.'}), 202
        
        # Start creating database in background
        create_database_background(model)
        return jsonify({'status': 'in_progress', 'message': f'Database for {model} is being created. Please try again in a few moments.'}), 202
    
    try:
        # Perform search with the selected model
        results = processor.search_with_model(query, model, k=3)
        
        # Format results for JSON response
        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_results.append({
                'rank': i,
                'document': result['document'],
                'filename': result['filename'],
                'chunk_id': result['chunk_id'],
                'similarity_score': round(result['similarity_score'], 3),
                'embedding_model': result['embedding_model'],
                'content': result['content']
            })
        
        return jsonify({
            'query': query,
            'model': model,
            'results': formatted_results
        })
    
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
    app.run(debug=True, host='0.0.0.0', port=5000)
