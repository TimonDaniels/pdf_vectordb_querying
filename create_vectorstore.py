import os 
from test_chunking import parse_json_text
import json


def main():
    
    from vectorstore import VectorStore

    vector_store = VectorStore()
    vector_store.set_embedding_model("openai")
    vector_store.from_chunk_files("data/semantic_chunks", ["*_chunk_*_semantic.json"], batch_size=500)

    
if __name__ == "__main__":
    main()