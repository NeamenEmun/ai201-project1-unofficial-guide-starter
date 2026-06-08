"""
LLM Generation with Grounding

Uses Groq API to generate grounded responses from retrieved context.
Enforces that responses are drawn only from the retrieved chunks, not from model knowledge.
"""

from groq import Groq
from typing import List, Dict
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()


class GroundedGenerator:
    def __init__(self, api_key: str = None):
        """
        Initialize Groq client for generation.
        
        Args:
            api_key: Groq API key (falls back to GROQ_API_KEY env var)
        """
        if api_key is None:
            api_key = os.getenv("GROQ_API_KEY")
        
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables. Set it in .env file.")
        
        self.client = Groq(api_key=api_key)
        print("Initialized Groq LLM client")
    
    def generate_answer(
        self,
        query: str,
        retrieved_chunks: List[Dict],
        max_tokens: int = 500
    ) -> Dict:
        """
        Generate a grounded answer using retrieved context.
        
        Args:
            query: User's question
            retrieved_chunks: List of retrieved chunks with keys {text, source, distance}
            max_tokens: Max tokens in response
        
        Returns:
            Dict with keys {answer, sources}
        """
        
        # Format context from retrieved chunks
        context = "\n\n".join([
            f"[Source: {chunk['source']}]\n{chunk['text']}"
            for chunk in retrieved_chunks
        ])
        
        # Extract unique sources
        sources = list(set(chunk["source"] for chunk in retrieved_chunks))
        
        # System prompt that enforces grounding
        system_prompt = """You are a helpful assistant answering questions about CS/Engineering courses and professors.
        
CRITICAL RULES:
1. Answer ONLY using the provided context. Do not use your general knowledge.
2. If the context doesn't contain enough information to answer the question, say: "I don't have enough information in the available reviews to answer that question."
3. Always cite which document(s) you're drawing from.
4. Be specific and factual - don't generalize beyond what's in the context.
5. If reviews contradict each other, mention both perspectives.

Format your answer clearly and include specific details (numbers, names, course codes, etc.) from the context."""

        user_prompt = f"""Context from student reviews:

{context}

Question: {query}

Answer based ONLY on the context above. If the context doesn't address the question, say so explicitly."""

        # Call Groq API
        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.3  # Lower temp for more grounded, factual responses
        )
        
        answer_text = response.choices[0].message.content
        
        return {
            "answer": answer_text,
            "sources": sources,
            "num_chunks_retrieved": len(retrieved_chunks)
        }


def generate_grounded_response(
    query: str,
    retrieved_chunks: List[Dict]
) -> Dict:
    """
    Convenience function to generate a response.
    """
    generator = GroundedGenerator()
    return generator.generate_answer(query, retrieved_chunks)


if __name__ == "__main__":
    # Test generation
    from embedder import load_vector_store
    
    # Load the vector store
    embedder = load_vector_store()
    
    # Test queries
    test_queries = [
        "How hard is Data Structures with Smith?",
        "What should I expect in Algorithms with Johnson?",
        "Are there weeder courses I should know about?",
    ]
    
    print("=== TESTING GROUNDED GENERATION ===\n")
    
    for query in test_queries:
        print(f"Question: {query}")
        print("-" * 80)
        
        # Retrieve chunks
        chunks = embedder.retrieve(query, top_k=5)
        
        # Generate answer
        response = generate_grounded_response(query, chunks)
        
        print(f"Answer:\n{response['answer']}")
        print(f"\nSources used: {', '.join(response['sources'])}")
        print(f"Chunks retrieved: {response['num_chunks_retrieved']}")
        print("\n" + "="*80 + "\n")
