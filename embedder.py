"""
Embedding and Vector Store Setup

Embeds chunks using sentence-transformers (all-MiniLM-L6-v2) and stores them in ChromaDB.
"""

import chromadb
from sentence_transformers import SentenceTransformer
import json
from pathlib import Path
from typing import List, Dict
from ingestion import process_documents


class RAGEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", db_path: str = "chroma_db"):
        """
        Initialize embedder with sentence-transformers and ChromaDB.
        
        Args:
            model_name: HuggingFace model name for embeddings
            db_path: Path to store ChromaDB
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.db_path = db_path
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_or_create_collection(
            name="professor_reviews",
            metadata={"hnsw:space": "cosine"}  # Cosine similarity for text
        )
        print(f"Initialized ChromaDB at {db_path}")
    
    def embed_and_store(self, chunks: List[Dict]):
        """
        Embed all chunks and store in ChromaDB.
        
        Args:
            chunks: List of dicts with keys {text, source, chunk_id}
        """
        print(f"Embedding {len(chunks)} chunks...")
        
        # Extract texts for embedding
        texts = [chunk["text"] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        print(f"Storing embeddings in ChromaDB...")
        
        # Prepare data for ChromaDB
        ids = [f"{chunk['source']}_{chunk['chunk_id']}" for chunk in chunks]
        documents = texts
        metadatas = [
            {
                "source": chunk["source"],
                "chunk_id": str(chunk["chunk_id"])
            }
            for chunk in chunks
        ]
        
        # Add to collection
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )
        
        print(f"Successfully stored {len(chunks)} chunks in ChromaDB")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve top-k most similar chunks for a query.
        
        Args:
            query: User query string
            top_k: Number of chunks to retrieve
        
        Returns:
            List of dicts with keys {text, source, distance}
        """
        # Embed the query
        query_embedding = self.model.encode(query)
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        retrieved = []
        for i in range(len(results["documents"][0])):
            retrieved.append({
                "text": results["documents"][0][i],
                "source": results["metadatas"][0][i]["source"],
                "distance": results["distances"][0][i]
            })
        
        return retrieved


def build_vector_store(documents_dir: str = "documents", db_path: str = "chroma_db"):
    """
    Build the complete vector store from scratch.
    """
    # Process documents
    print("=== DOCUMENT PROCESSING ===")
    chunks = process_documents(documents_dir)
    
    # Embed and store
    print("\n=== EMBEDDING & STORAGE ===")
    embedder = RAGEmbedder(db_path=db_path)
    embedder.embed_and_store(chunks)
    
    return embedder


def load_vector_store(db_path: str = "chroma_db") -> RAGEmbedder:
    """
    Load an existing vector store.
    """
    embedder = RAGEmbedder(db_path=db_path)
    print(f"Loaded vector store from {db_path}")
    return embedder


if __name__ == "__main__":
    # Build or load vector store
    embedder = build_vector_store()
    
    # Test retrieval
    print("\n=== TESTING RETRIEVAL ===")
    test_queries = [
        "How hard is Data Structures with Smith?",
        "What is the grade distribution in Algorithms?",
        "Which courses are weeder courses?",
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = embedder.retrieve(query, top_k=3)
        for i, result in enumerate(results, 1):
            print(f"\n  Result {i} (distance: {result['distance']:.3f}, source: {result['source']})")
            print(f"  {result['text'][:150]}...")
