import json
import os 


def find_project_root():
    project_name = "pdf_vectordb_querying"
    current_path = os.path.abspath(__file__)
    path_parts = current_path.split(os.sep)
    # Reverse search for the project folder name
    for i in range(len(path_parts) - 1, -1, -1):
        if path_parts[i] == project_name:
            return os.sep.join(path_parts[:i+1])
    raise ValueError(f"Project folder '{project_name}' not found in the directory path")


def parse_json_text(chunked_text: str, chunk_name: str):
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
            return chunks

    except Exception as e:
        print(f"   ✗ Error parsing semantic chunks with name {chunk_name}: {e}")
        print(f"   Raw output (first 200 chars): {chunked_text[:200]}...")
        return

