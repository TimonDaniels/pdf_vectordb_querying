import os
from typing import Dict
from openai import OpenAI


class QueryExpander:
    
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
   
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
