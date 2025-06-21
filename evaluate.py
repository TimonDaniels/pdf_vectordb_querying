"""
Simple TruLens Evaluation Script for PDF Vector Database RAG System
Evaluates retrieval quality and answer synthesis quality with persistent storage.
"""

import json
import time
import numpy as np
from typing import List, Dict, Any
from dotenv import load_dotenv

from trulens.core import TruSession, Feedback, Select
from trulens.apps.app import TruApp, instrument
from trulens.providers.openai import OpenAI as TruOpenAI
from trulens.providers.huggingface import Huggingface

from src.vectorstore import VectorStore
from src.synthesis import OpinionSynthesizer

load_dotenv('.env.local')

class RAGEvaluator:
    """Simple RAG system wrapper for TruLens evaluation."""
    
    def __init__(self, embedding_model: str = "openai"):
        self.vector_store = VectorStore()
        self.vector_store.set_embedding_model(embedding_model)
        self.vector_store.load_database_by_model(embedding_model)
        self.synthesizer = OpinionSynthesizer()
        self.embedding_model = embedding_model
    
    @instrument
    def retrieve(self, query: str, k: int = 5) -> List[str]:
        """Retrieve relevant documents from vector store."""
        results = self.vector_store.search_with_model(
            query=query, 
            model_name=self.embedding_model, 
            k=k
        )
        return results
    
    @instrument
    def synthesize(self, opinion: str, context_docs: List[str]) -> str:
        """Generate synthesis from retrieved context."""
        # Use stored full results with metadata
        return self.synthesizer.synthesize_party_opinion_fit(
            opinion=opinion,
            query_results=context_docs,
        )
    
    @instrument
    def query(self, opinion: str, k: int = 5) -> str:
        """Complete RAG pipeline: retrieve + synthesize."""
        context_docs = self.retrieve(opinion, k=k)
        response = self.synthesize(opinion, context_docs)
        return response


def setup_trulens_session():
    """Initialize TruLens session with persistent storage."""
    session = TruSession()
    # Don't reset database to maintain persistent storage
    print("TruLens session initialized")
    return session


def setup_feedback_functions():
    """Setup TruLens feedback functions for evaluation."""
    oai_provider = TruOpenAI()
    # hf_provider = Huggingface()
    
    # # Groundedness: How well is the answer supported by retrieved context
    # # Use standard TruLens approach - it will automatically extract content from strings
    # f_groundedness = (
    #     Feedback(hf_provider.groundedness_measure_with_nli, name="Groundedness")
    #     .on(Select.RecordCalls.retrieve.rets[:].content)  # Gets the list of strings returned by retrieve
    #     .on_output()
    # )
    
    # Answer Relevance: How well does the answer address the opinion/question  
    f_answer_relevance = (
        Feedback(oai_provider.relevance_with_cot_reasons, name="Answer Relevance")
        .on_input()
        .on_output()
    )
    
    # Context Relevance: How relevant are the retrieved docs to the opinion
    f_context_relevance = (
        Feedback(oai_provider.context_relevance_with_cot_reasons, name="Context Relevance")
        .on_input()
        .on(Select.RecordCalls.retrieve.rets[:].content)  # Gets each string in the list
        .aggregate(np.mean)
    )
    
    return [f_answer_relevance, f_context_relevance]


def load_test_questions(file_path: str = "test_questions.json") -> List[Dict]:
    """Load test questions from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def run_evaluation(embedding_model: str = "openai"):
    """Run the evaluation with TruLens tracking."""
    print("🚀 Starting TruLens evaluation...")
    
    # Setup
    session = setup_trulens_session()
    rag_evaluator = RAGEvaluator(embedding_model=embedding_model)
    feedback_functions = setup_feedback_functions()
    
    # Create TruApp wrapper
    tru_rag = TruApp(
        rag_evaluator,
        app_name="PDF_RAG",
        app_version="openai-v1.1",
        feedbacks=feedback_functions
    )
    
    # Load test questions
    test_questions = load_test_questions()
    # take 2 questions for quick testing
    test_questions = test_questions[:2]
    print(f"📝 Loaded {len(test_questions)} test questions")
    
    # Run evaluation
    print("🔍 Running evaluation...")
    with tru_rag as recording:
        for i, question_data in enumerate(test_questions):
            opinion = question_data["opinion"]
            print(f"  Question {i+1}/{len(test_questions)}: {opinion[:50]}...")
            
            try:
                response = rag_evaluator.query(opinion)
                print(f"  ✅ Response generated successfully")
            except Exception as e:
                print(f"  ❌ Error: {e}")
    

    # Display results
    print("\n📊 Evaluation Results:")
    leaderboard = session.get_leaderboard()
    print(leaderboard)
    
    return session


if __name__ == "__main__":
    # Run evaluation
    session = run_evaluation()
