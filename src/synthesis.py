"""
Synthesis Module for PDF Vector Database Querying
Provides OpenAI-powered synthesis of explanations based on opinions and query results.
"""

import os
from typing import List, Dict, Optional, Any
from openai import OpenAI


class OpinionSynthesizer:
    """
    Synthesizes explanations about how well parties/entities fit given opinions
    using OpenAI models and vector database query results.
    """
    
    def __init__(self, model: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
        """
        Initialize the synthesizer with OpenAI configuration.
        
        Args:
            model: OpenAI model to use (default: gpt-3.5-turbo)
            api_key: OpenAI API key (if not provided, will use environment variable)
        """
        self.model = model
        self.client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
    
    def synthesize_party_opinion_fit(
        self, 
        opinion: str, 
        query_results: List[Dict[str, Any]], 
        max_context_length: int = 3000
    ) -> str:
        """
        Synthesize an explanation of how well parties fit a given opinion based on query results.
        
        Args:
            opinion: The opinion/stance to evaluate against
            query_results: List of relevant documents from vector database query
            max_context_length: Maximum length of context to include in prompt
            
        Returns:
            Synthesized explanation as a string
        """
        # Extract party names from query results
        parties = self._extract_parties_from_results(query_results)
        
        # Prepare context from query results
        context = self._prepare_context(query_results, max_context_length)
        
        # Create the synthesis prompt
        prompt = self._create_synthesis_prompt(opinion, parties, context)
        
        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=800
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error generating synthesis: {str(e)}"
    
    def _prepare_context(self, query_results: List[Dict[str, Any]], max_length: int) -> str:
        """
        Prepare context from query results, ensuring it doesn't exceed max_length.
        
        Args:
            query_results: List of documents with content and metadata
            max_length: Maximum character length for context
            
        Returns:
            Formatted context string
        """
        context_parts = []
        current_length = 0
        
        for result in query_results:
            # Extract content and metadata
            content = result.get('content', result.get('page_content', ''))
            metadata = result.get('metadata', {})
            
            # Format the context entry
            source_info = ""
            if metadata:
                source_parts = []
                if 'source' in metadata:
                    source_parts.append(f"Source: {metadata['source']}")
                if 'page' in metadata:
                    source_parts.append(f"Page: {metadata['page']}")
                if 'chunk' in metadata:
                    source_parts.append(f"Chunk: {metadata['chunk']}")
                source_info = f" [{', '.join(source_parts)}]" if source_parts else ""
            
            entry = f"Document{source_info}:\n{content}\n\n"
            
            # Check if adding this entry would exceed max_length
            if current_length + len(entry) > max_length:
                # Truncate if necessary
                remaining_space = max_length - current_length
                if remaining_space > 100:  # Only add if there's meaningful space
                    truncated_content = content[:remaining_space-50] + "..."
                    entry = f"Document{source_info}:\n{truncated_content}\n\n"
                    context_parts.append(entry)
                break
            
            context_parts.append(entry)
            current_length += len(entry)
        
        return "".join(context_parts)
    
    def _extract_parties_from_results(self, query_results: List[Dict[str, Any]]) -> List[str]:
        """
        Extract party names from query results metadata.
        
        Args:
            query_results: List of documents with content and metadata
            
        Returns:
            List of unique party names found in the results
        """
        parties = set()
        
        for result in query_results:
            metadata = result.get('metadata', {})
            source = metadata.get('source', '')
            
            # Extract party names from source filenames (e.g., "D66_chunk_1.txt" -> "D66")
            if source:
                # Look for common patterns like "PARTY_chunk_X.txt" or "PARTY.pdf"
                import re
                party_match = re.match(r'^([A-Z]+(?:[0-9]+)?)', source.split('/')[-1])
                if party_match:
                    party_name = party_match.group(1)
                    # Handle common Dutch party names
                    if party_name in ['D66', 'PVV', 'VVD', 'CDA', 'GL', 'PvdA', 'SP', 'CU', 'SGP', 'DENK', 'FVD', 'JA21', 'BBB']:
                        parties.add(party_name)
        
        return sorted(list(parties))
    
    def _create_synthesis_prompt(self, opinion: str, parties: List[str], context: str) -> str:
        """
        Create the synthesis prompt for the OpenAI model.
        
        Args:
            opinion: The opinion to evaluate against
            parties: List of party names found in the query results
            context: Prepared context from query results
            
        Returns:
            Formatted prompt string
        """
        if not parties:
            parties_text = "the political parties/entities represented in the documents"
        elif len(parties) == 1:
            parties_text = f'"{parties[0]}"'
        else:
            parties_text = f'"{", ".join(parties[:-1])}" and "{parties[-1]}"'
        
        prompt = f"""
Based on the provided documents, analyze how well {parties_text} align with or fit the following opinion:

OPINION: "{opinion}"

Please provide a comprehensive analysis that:
1. Explains how each party's positions, actions, or statements relate to this opinion using the provided context and quote the specific documents
2. Provides a balanced assessment of the degree of fit for each party and give it a number ranging from 1 to 10, where 1 is no fit and 10 is perfect fit
3. Keep the response concise but informative, focusing on key points and evidence and use a quote if possible
4. Compares the parties' positions if multiple parties are present

RELEVANT DOCUMENTS:
{context}

Please structure your response as JSON with the following format:
```json
{{
    {{
        party_analysis: [
            {{"party_name": party, "score": 1-10, "explanation": "example explanation", "evidence": "specific quotes or references from documents"}},
            {{"party_name": party, "score": 1-10, "explanation": "example explanation", "evidence": "specific quotes or references from documents"}}
        ],
        "conclusion": "Overall summary of party alignments",
    }}
}}
```         
"""
        return prompt
    
    def _get_system_prompt(self) -> str:
        """
        Get the system prompt that defines the AI's role and behavior.
        
        Returns:
            System prompt string
        """
        return """You are an expert political analyst and researcher. Your task is to objectively analyze how well political parties or entities align with given opinions based on provided documentary evidence.

Key guidelines:
- Be objective and evidence-based in your analysis
- Cite specific examples and quotes from the provided documents
- Acknowledge when evidence is limited or contradictory
- Provide nuanced assessments rather than simple yes/no answers
- Consider both explicit statements and implicit positions
- Distinguish between party positions and individual member views when relevant
- Be clear about the strength of the evidence for your conclusions
- Answer in Dutch
"""


def synthesize_opinion_fit(
    opinion: str,
    query_results: List[Dict[str, Any]],
    model: str = "gpt-3.5-turbo",
    api_key: Optional[str] = None
) -> str:
    """
    Convenience function to synthesize party-opinion fit explanation.
    
    Args:
        opinion: The opinion/stance to evaluate against
        query_results: List of relevant documents from vector database query
        model: OpenAI model to use
        api_key: OpenAI API key (optional)
        
    Returns:
        Synthesized explanation as a string
    """
    synthesizer = OpinionSynthesizer(model=model, api_key=api_key)
    return synthesizer.synthesize_party_opinion_fit(opinion, query_results)


# Example usage
if __name__ == "__main__":
    # Example query results structure
    example_results = [
        {
            "content": "The party supports renewable energy initiatives and has voted for climate action bills.",
            "metadata": {"source": "D66_chunk_1.txt", "page": 1}
        },
        {
            "content": "Leadership has made statements supporting environmental protection measures.",
            "metadata": {"source": "PVV_chunk_5.txt", "page": 3}
        }
    ]
    
    # Example usage
    opinion = "Climate change requires immediate government action"
    
    explanation = synthesize_opinion_fit(opinion, example_results)
    print(f"Analysis of how the parties fit the opinion '{opinion}':")
    print(explanation)