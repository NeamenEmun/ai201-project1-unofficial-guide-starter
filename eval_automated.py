"""
Automated Evaluation (Non-Interactive)

Runs all test questions and outputs results without requiring user input.
Used for generating evaluation data for the README.
"""

from embedder import load_vector_store
from generator import GroundedGenerator
import json
from datetime import datetime


EVALUATION_QUESTIONS = [
    {
        "id": 1,
        "question": "What do students say about Professor Smith's grading practices and whether he curves exams?",
        "expected_answer": "Students should mention whether exams are curved, if grades are harsh or fair, and specific grading percentages"
    },
    {
        "id": 2,
        "question": "Is Data Structures a hard course and what should students expect in terms of workload?",
        "expected_answer": "Should mention that it's challenging, includes heavy programming assignments, typically 8-12 hours/week outside class"
    },
    {
        "id": 3,
        "question": "How important is attending class for Professor Johnson's lectures, and what material appears on her tests?",
        "expected_answer": "Attendance is critical, exams focus on lecture content over textbook, skipping class is risky"
    },
    {
        "id": 4,
        "question": "What is the typical grade distribution in an Algorithms course and how are students graded?",
        "expected_answer": "Percentage of grade from exams vs. projects, typical letter grade distribution, any extra credit opportunities"
    },
    {
        "id": 5,
        "question": "Are there any 'weeder' courses students should be aware of and what makes them difficult?",
        "expected_answer": "Names courses known as challenging gatekeepers, explains why (theory, time-intensive, harsh grading), mentions survival strategies"
    }
]


def run_automated_evaluation():
    """Run evaluation without user input."""
    
    print("=" * 80)
    print("🧪 AUTOMATED EVALUATION: RAG System")
    print("=" * 80 + "\n")
    
    # Load system
    embedder = load_vector_store()
    generator = GroundedGenerator()
    
    results = []
    
    for test in EVALUATION_QUESTIONS:
        print(f"Running Test {test['id']}...")
        
        # Retrieve
        chunks = embedder.retrieve(test['question'], top_k=5)
        
        # Generate
        response = generator.generate_answer(test['question'], chunks)
        
        result = {
            "test_id": test['id'],
            "question": test['question'],
            "expected_answer": test['expected_answer'],
            "system_response": response['answer'],
            "sources_retrieved": response['sources'],
            "num_chunks_retrieved": response['num_chunks_retrieved'],
            "retrieval_distances": [chunk['distance'] for chunk in chunks]
        }
        
        results.append(result)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"evaluation_data_{timestamp}.json"
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Evaluation data saved to: {output_file}")
    print(f"\nGenerated {len(results)} test results")
    
    return results


if __name__ == "__main__":
    run_automated_evaluation()
