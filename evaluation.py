"""
Evaluation Framework

Tests the RAG system on 5 test questions with ground truth answers.
Evaluates both retrieval quality and generation accuracy.
"""

from embedder import load_vector_store
from generator import GroundedGenerator
import json
from datetime import datetime


EVALUATION_QUESTIONS = [
    {
        "id": 1,
        "question": "What do students say about Professor Smith's grading practices and whether he curves exams?",
        "expected_answer": "Students should mention whether exams are curved, if grades are harsh or fair, and specific grading percentages if mentioned (e.g., 'Smith curves midterms but not finals')"
    },
    {
        "id": 2,
        "question": "Is Data Structures a hard course and what should students expect in terms of workload?",
        "expected_answer": "Should mention that it's challenging, includes heavy programming assignments, and typically has 8-12 hours per week of work outside class"
    },
    {
        "id": 3,
        "question": "How important is attending class for Professor Johnson's lectures, and what material appears on her tests?",
        "expected_answer": "Should indicate that attendance is critical, exams focus on lecture content over the textbook, and skipping class is risky"
    },
    {
        "id": 4,
        "question": "What is the typical grade distribution in an Algorithms course and how are students graded?",
        "expected_answer": "Should mention what percentage of grade comes from exams vs. projects, typical letter grade distribution, and any extra credit opportunities"
    },
    {
        "id": 5,
        "question": "Are there any 'weeder' courses students should be aware of and what makes them difficult?",
        "expected_answer": "Should name 1-2 courses known for being challenging gatekeepers, explain why (lots of theory, time-intensive projects, harsh grading), and mention what students recommend to succeed"
    }
]


def run_evaluation():
    """Run complete evaluation on all test questions."""
    
    print("=" * 80)
    print("🧪 EVALUATION: RAG System on Professor/Course Reviews")
    print("=" * 80 + "\n")
    
    # Load system components
    embedder = load_vector_store()
    generator = GroundedGenerator()
    
    results = []
    
    for test in EVALUATION_QUESTIONS:
        print(f"\n{'='*80}")
        print(f"TEST {test['id']}: {test['question']}")
        print(f"{'='*80}\n")
        
        # Retrieve chunks
        print("🔍 Retrieving relevant chunks...")
        chunks = embedder.retrieve(test['question'], top_k=5)
        
        print(f"✓ Retrieved {len(chunks)} chunks\n")
        print("Retrieved chunks (first 150 chars each):")
        for i, chunk in enumerate(chunks, 1):
            print(f"  {i}. [{chunk['source']}] {chunk['text'][:150]}...")
        
        # Generate answer
        print("\n💡 Generating answer...")
        response = generator.generate_answer(test['question'], chunks)
        
        print(f"✓ Generated response\n")
        print("SYSTEM RESPONSE:")
        print("-" * 80)
        print(response['answer'])
        print("-" * 80)
        
        # Manual evaluation
        print(f"\nEXPECTED ANSWER SHOULD INCLUDE:")
        print(f"{test['expected_answer']}\n")
        
        print("ACCURACY ASSESSMENT:")
        print("Please rate the response (type: ACCURATE / PARTIAL / INACCURATE):")
        accuracy = input(">>> ").strip().upper()
        
        if accuracy not in ["ACCURATE", "PARTIAL", "INACCURATE"]:
            accuracy = "INACCURATE"
        
        print("\nCHALLENGES OBSERVED:")
        challenges = input("Describe any challenges or failures: ")
        
        result = {
            "test_id": test['id'],
            "question": test['question'],
            "expected": test['expected_answer'],
            "system_response": response['answer'],
            "sources_retrieved": response['sources'],
            "chunks_retrieved": response['num_chunks_retrieved'],
            "accuracy": accuracy,
            "challenges": challenges,
            "retrieval_distances": [chunk['distance'] for chunk in chunks]
        }
        
        results.append(result)
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 EVALUATION SUMMARY")
    print("=" * 80 + "\n")
    
    accuracy_counts = {
        "ACCURATE": sum(1 for r in results if r['accuracy'] == "ACCURATE"),
        "PARTIAL": sum(1 for r in results if r['accuracy'] == "PARTIAL"),
        "INACCURATE": sum(1 for r in results if r['accuracy'] == "INACCURATE")
    }
    
    print(f"Total tests: {len(results)}")
    print(f"  ✓ Accurate:     {accuracy_counts['ACCURATE']}/5")
    print(f"  ~ Partial:      {accuracy_counts['PARTIAL']}/5")
    print(f"  ✗ Inaccurate:   {accuracy_counts['INACCURATE']}/5")
    print(f"\nSuccess rate: {accuracy_counts['ACCURATE']}/5 = {100*accuracy_counts['ACCURATE']//5}%\n")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"evaluation_results_{timestamp}.json"
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Evaluation results saved to: {output_file}\n")
    
    return results


if __name__ == "__main__":
    run_evaluation()
