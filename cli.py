"""
Command-Line Interface for The Unofficial Guide RAG System

Simple CLI for querying the professor/course review system.
"""

from embedder import load_vector_store
from generator import GroundedGenerator
import sys


def main():
    """Main CLI loop."""
    print("=" * 80)
    print("🎓 The Unofficial Guide: Professor & Course Reviews")
    print("=" * 80)
    print("Ask questions about CS/Engineering courses and professors.\n")
    
    # Load vector store
    print("Loading knowledge base...")
    embedder = load_vector_store()
    
    # Initialize generator
    print("Initializing LLM...\n")
    generator = GroundedGenerator()
    
    print("Ready! Type your question below (or 'quit' to exit).\n")
    
    while True:
        try:
            query = input("Your question: ").strip()
            
            if not query:
                print("Please enter a question.\n")
                continue
            
            if query.lower() in ["quit", "exit", "q"]:
                print("Thanks for using The Unofficial Guide!")
                break
            
            # Retrieve chunks
            print("\n🔍 Searching knowledge base...")
            chunks = embedder.retrieve(query, top_k=5)
            
            print(f"   Found {len(chunks)} relevant chunks\n")
            
            # Generate answer
            print("💡 Generating answer...\n")
            response = generator.generate_answer(query, chunks)
            
            # Display answer
            print("=" * 80)
            print("ANSWER:")
            print("=" * 80)
            print(f"\n{response['answer']}\n")
            
            print("=" * 80)
            print(f"Sources: {', '.join(response['sources'])}")
            print(f"Chunks retrieved: {response['num_chunks_retrieved']}")
            print("=" * 80 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again.\n")


if __name__ == "__main__":
    main()
