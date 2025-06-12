"""
Query Expansion Service using OpenAI GPT-3.5-turbo
Expands search queries with synonyms, related terms, and alternative phrasings.
"""

import json
import hashlib
import os
import time
from typing import Dict, Optional
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv('../.env.local')


class QueryExpander:
    """
    Query expansion service that uses OpenAI GPT-3.5-turbo to enhance search queries
    with synonyms, related terms, and alternative phrasings.
    """
    
    def __init__(self, cache_file: str = "query_cache.json", cache_ttl: int = 86400):
        """
        Initialize the query expander.
        
        Args:
            cache_file: File to store cached expansions
            cache_ttl: Cache time-to-live in seconds (default: 24 hours)
        """
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.cache_file = cache_file
        self.cache_ttl = cache_ttl
        self._cache = self._load_cache()
        
    def _load_cache(self) -> Dict:
        """Load cached expansions from file."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load cache file: {e}")
        return {}
    
    def _save_cache(self):
        """Save cache to file."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self._cache, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save cache file: {e}")
    
    def _get_cache_key(self, query: str) -> str:
        """Generate cache key for a query."""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()
    
    def _is_cache_valid(self, cache_entry: Dict) -> bool:
        """Check if cache entry is still valid."""
        if 'timestamp' not in cache_entry:
            return False
        return time.time() - cache_entry['timestamp'] < self.cache_ttl
    
    def _create_expansion_prompt(self, query: str) -> str:
        """Create prompt for GPT-3.5-turbo to expand the query."""
        return f"""You are helping to expand a search query for better document retrieval. 

Original query: "{query}"

Please expand this query by adding relevant synonyms, related terms, and alternative phrasings that would help find related content in political documents. The expanded query should:
1. Include the original terms
2. Add synonyms and related concepts
3. Include common alternative phrasings
4. Stay focused on the core topic
5. Be suitable for semantic search

Respond with just the expanded query text, nothing else. Keep it under 200 words."""

    def expand_query(self, original_query: str) -> Dict:
        """
        Expand a query using OpenAI GPT-3.5-turbo.
        
        Args:
            original_query: The original search query
            
        Returns:
            Dictionary with expansion results:
            {
                "original_query": str,
                "expanded_query": str,
                "expansion_used": bool,
                "cached": bool,
                "error": str or None
            }
        """
        if not original_query or not original_query.strip():
            return {
                "original_query": original_query,
                "expanded_query": original_query,
                "expansion_used": False,
                "cached": False,
                "error": "Empty query"
            }
        
        # Check cache first
        cache_key = self._get_cache_key(original_query)
        if cache_key in self._cache and self._is_cache_valid(self._cache[cache_key]):
            cached_result = self._cache[cache_key]
            return {
                "original_query": original_query,
                "expanded_query": cached_result["expanded_query"],
                "expansion_used": True,
                "cached": True,
                "error": None
            }
        
        # Try to expand using OpenAI
        try:
            print(f"Expanding query: {original_query}")
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that expands search queries for better document retrieval."},
                    {"role": "user", "content": self._create_expansion_prompt(original_query)}
                ],
                max_tokens=300,
                temperature=0.3
            )
            
            expanded_query = response.choices[0].message.content.strip()
            
            # Cache the result
            cache_entry = {
                "expanded_query": expanded_query,
                "timestamp": time.time()
            }
            self._cache[cache_key] = cache_entry
            self._save_cache()
            
            print(f"Query expanded successfully (cached)")
            
            return {
                "original_query": original_query,
                "expanded_query": expanded_query,
                "expansion_used": True,
                "cached": False,
                "error": None
            }
            
        except Exception as e:
            print(f"Error expanding query: {e}")
            # Return original query on error
            return {
                "original_query": original_query,
                "expanded_query": original_query,
                "expansion_used": False,
                "cached": False,
                "error": str(e)
            }
    
    def clear_cache(self):
        """Clear the expansion cache."""
        self._cache = {}
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
        print("Query expansion cache cleared")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        valid_entries = sum(1 for entry in self._cache.values() if self._is_cache_valid(entry))
        return {
            "total_entries": len(self._cache),
            "valid_entries": valid_entries,
            "expired_entries": len(self._cache) - valid_entries,
            "cache_file": self.cache_file,
            "cache_ttl_hours": self.cache_ttl / 3600
        }
