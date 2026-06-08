"""
Document Ingestion and Chunking Pipeline

Loads documents from the documents/ folder, cleans them, and chunks them
according to the specifications in planning.md:
- Chunk size: 400 characters
- Overlap: 100 characters
- Metadata: source filename and chunk position
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Tuple


def load_documents(documents_dir: str = "documents") -> List[Tuple[str, str]]:
    """
    Load all .txt files from the documents directory.
    Returns list of (filename, content) tuples.
    """
    documents = []
    doc_path = Path(documents_dir)
    
    if not doc_path.exists():
        raise FileNotFoundError(f"Documents directory not found: {documents_dir}")
    
    for file in sorted(doc_path.glob("*.txt")):
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()
            documents.append((file.name, content))
    
    if not documents:
        raise ValueError(f"No .txt files found in {documents_dir}")
    
    print(f"Loaded {len(documents)} documents")
    return documents


def clean_text(text: str) -> str:
    """
    Clean text by removing unnecessary whitespace, HTML entities, and formatting artifacts.
    """
    # Remove HTML entities
    text = re.sub(r'&[a-z]+;', '', text)
    text = re.sub(r'&#\d+;', '', text)
    
    # Remove extra whitespace (multiple spaces, tabs, etc.)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Remove common navigation/boilerplate (if present)
    text = re.sub(r'(Click here|Read more|Share|Comment|Like|Subscribe)', '', text, flags=re.IGNORECASE)
    
    return text


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks of specified size.
    
    Args:
        text: Text to chunk
        chunk_size: Target chunk size in characters
        overlap: Overlap between consecutive chunks in characters
    
    Returns:
        List of chunk strings
    """
    chunks = []
    step = chunk_size - overlap  # Move forward by this amount each iteration
    
    if len(text) <= chunk_size:
        # Text is smaller than chunk size, return as-is
        if len(text.strip()) > 0:
            chunks.append(text.strip())
        return chunks
    
    for i in range(0, len(text), step):
        chunk = text[i:i + chunk_size]
        
        # Ensure chunk is not empty after stripping
        if len(chunk.strip()) > 0:
            chunks.append(chunk.strip())
        
        # If we've reached near the end, don't create partial overlapping chunks
        if i + chunk_size >= len(text):
            break
    
    return chunks


def process_documents(documents_dir: str = "documents") -> List[Dict]:
    """
    Main pipeline: load documents, clean, and chunk them.
    
    Returns list of dicts with keys: {text, source, chunk_id}
    """
    documents = load_documents(documents_dir)
    all_chunks = []
    
    for filename, content in documents:
        # Clean the document
        cleaned = clean_text(content)
        
        # Chunk it
        chunks = chunk_text(cleaned, chunk_size=400, overlap=100)
        
        # Add metadata
        for chunk_id, chunk_text_content in enumerate(chunks):
            all_chunks.append({
                "text": chunk_text_content,
                "source": filename,
                "chunk_id": chunk_id
            })
    
    print(f"Created {len(all_chunks)} chunks total")
    
    # Print statistics
    total_chars = sum(len(chunk["text"]) for chunk in all_chunks)
    avg_chunk_size = total_chars / len(all_chunks) if all_chunks else 0
    print(f"Average chunk size: {avg_chunk_size:.0f} characters")
    
    return all_chunks


def print_sample_chunks(chunks: List[Dict], num_samples: int = 5):
    """Print sample chunks for inspection."""
    import random
    
    if len(chunks) < num_samples:
        samples = chunks
    else:
        samples = random.sample(chunks, num_samples)
    
    print(f"\n=== SAMPLE CHUNKS ({len(samples)} of {len(chunks)}) ===\n")
    for i, chunk in enumerate(samples, 1):
        print(f"--- Chunk {i} (Source: {chunk['source']}, ID: {chunk['chunk_id']}) ---")
        print(f"{chunk['text'][:200]}..." if len(chunk['text']) > 200 else chunk['text'])
        print(f"Length: {len(chunk['text'])} chars\n")


if __name__ == "__main__":
    # Example usage
    try:
        chunks = process_documents("documents")
        print_sample_chunks(chunks, num_samples=5)
    except Exception as e:
        print(f"Error: {e}")
