# Project 1 Planning: The Unofficial Guide

> Write this document before you write any pipeline code.
> Your spec and architecture diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation — the more specific they are, the more useful the generated code will be.
> Update the Retrieval Approach and Chunking Strategy sections if you change your approach during implementation.
> Update this file before starting any stretch features.

---

## Domain

Professor and Course Reviews - Student-generated knowledge about CS/Engineering professors, their teaching styles, exam difficulty, grading practices, and course expectations. This knowledge is valuable and difficult to find through official channels because universities publish course descriptions and professor bios, but not candid student experiences—what professors actually ask on exams, how they grade, whether attendance matters, workload expectations, and which courses are "weeder" classes. Students share this on Rate My Professor, Reddit, course forums, and Discord, but it's scattered and not searchable through a single lens.

---

## Documents

| # | Source Type | Description | Sample Location |
|---|-----------|-------------|-----------------|
| 1 | Rate My Professor | Computer Science professors, teaching style and exam reviews | rmp_cs_reviews.txt |
| 2 | Rate My Professor | Engineering professors (various disciplines) | rmp_eng_reviews.txt |
| 3 | Reddit r/learnprogramming | Tips on learning CS, common pitfalls, course advice | reddit_learnprogramming.txt |
| 4 | Reddit r/cscareerquestions | Course selection and professor feedback | reddit_cscareerq.txt |
| 5 | Student Forum Post | "Hardest CS courses" thread - detailed discussion | forum_hardest_courses.txt |
| 6 | Student Blog | Course review blog - 3 detailed course analysis posts | blog_course_reviews.txt |
| 7 | Course Syllabus Summary | Compiled notes on course expectations and workload | syllabi_summary.txt |
| 8 | Student Discord Chat | Channel discussion on midterm prep and professor quirks | discord_exam_tips.txt |
| 9 | Wiki/FAQ | "Which professors should I take?" guide | wiki_professor_guide.txt |
| 10 | Quora Answers | Questions about specific professors answered by alumni | quora_prof_answers.txt |

---

## Chunking Strategy

**Chunk size:** 400 characters (approx 60-80 tokens)

**Overlap:** 100 characters

**Reasoning:** Professor reviews are typically short, opinion-based text (2-5 sentences per review). A 400-character chunk captures one complete review or thought about a professor's teaching style, grading practices, or course difficulty. 100-character overlap ensures that if a key fact spans two chunks, both chunks remain independently retrievable. This avoids losing information at chunk boundaries while keeping each chunk focused enough for semantic search to match specific queries (e.g., "How hard are the exams?" should return chunks specifically about exams, not general professor bios).

---

## Retrieval Approach

**Embedding model:** all-MiniLM-L6-v2 (via sentence-transformers). Runs locally with no API key required.

**Top-k:** 5 chunks per query

**Production tradeoff reflection:** For a production system, I would consider: (1) multilingual support (all-MiniLM supports 50+ languages, but models like multilingual-e5-large offer better cross-lingual performance), (2) context length (all-MiniLM max 256 tokens; for longer documents, models like e5-large-v2 with 512 token context would be better), (3) domain-specific accuracy (general-purpose embeddings may miss nuances in student slang or professor-specific terminology; a domain-finetuned model would help), (4) cost/latency tradeoff (all-MiniLM is fast and cheap locally; larger models like text-embedding-3-large from OpenAI cost more but are more accurate). For this use case, local all-MiniLM fits perfectly because latency doesn't matter for a one-shot query interface, and we want to avoid API dependencies.

---

## Evaluation Plan

| # | Question | Expected answer |
|---|----------|-----------------|
| 1 | What do students say about Professor Smith's grading practices and whether he curves exams? | Students should mention whether exams are curved, if grades are harsh or fair, and specific grading percentages if mentioned (e.g., "Smith curves midterms but not finals") |
| 2 | Is Data Structures a hard course and what should students expect in terms of workload? | Should mention that it's challenging, includes heavy programming assignments, and typically has 8-12 hours per week of work outside class |
| 3 | How important is attending class for Professor Johnson's lectures, and what material appears on her tests? | Should indicate that attendance is critical, exams focus on lecture content over the textbook, and skipping class is risky |
| 4 | What is the typical grade distribution in an Algorithms course and how are students graded? | Should mention what percentage of grade comes from exams vs. projects, typical letter grade distribution, and any extra credit opportunities |
| 5 | Are there any "weeder" courses students should be aware of and what makes them difficult? | Should name 1-2 courses known for being challenging gatekeepers, explain why (lots of theory, time-intensive projects, harsh grading), and mention what students recommend to succeed |

---

## Anticipated Challenges

1. **Noisy and subjective reviews:** Student reviews contain opinions, sarcasm, and highly subjective judgments ("Smith is the worst" vs. "Smith is amazing"). The same fact presented in different documents may conflict (some students say a course is hard, others say it's easy). The retrieval may pull contradictory information, and the LLM may struggle to reconcile them. Solution: Include multiple perspectives in responses and attribute each to its source.

2. **Incomplete or scattered information:** Key facts like "exams are curved" might be mentioned in one document but not others. If a query retrieves chunks that mention "curved" in different contexts (e.g., "curved like a rainbow" in a building description), the model may misinterpret. Also, a student asking "What's the grade distribution?" may get back generic statements like "fair grading" rather than specific percentages if the documents don't contain them. Solution: LLM response should acknowledge when documents don't contain specific information ("Students report fair grading, but no specific percentages were mentioned in available reviews").

---

## Architecture

```
┌─────────────────────┐
│ Document Ingestion  │
│  (load .txt files)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────────┐
│ Cleaning & Preprocessing            │
│ (remove HTML, extra whitespace)     │
└──────────┬──────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│ Chunking                             │
│ (400 char chunks, 100 char overlap)  │
└──────────┬───────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│ Embedding                            │
│ (all-MiniLM-L6-v2)                   │
└──────────┬───────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│ Vector Store                         │
│ (ChromaDB - local, persistent)       │
└──────────┬───────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│ Retrieval                            │
│ (semantic search, top-5 chunks)      │
└──────────┬───────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│ Generation                           │
│ (Groq LLM: llama-3.3-70b-versatile)  │
│ (grounded response + attribution)    │
└──────────────────────────────────────┘
```
     You can use ASCII art, a Mermaid diagram, or embed a sketch as an image.
     You'll use this diagram as context when prompting AI tools to implement each stage. -->

---

## AI Tool Plan

**Milestone 3 — Ingestion and chunking:**
I'll give Claude my Chunking Strategy section, Documents section, and Architecture diagram. I'll ask it to implement `ingestion.py` with functions `load_documents()`, `clean_text()`, and `chunk_text()` that match my 400-char chunk size with 100-char overlap. I'll verify by loading 3 sample documents, printing 5 chunks, and checking that each chunk is readable, contains document metadata, and is the right size.

**Milestone 4 — Embedding and retrieval:**
I'll give Claude my Architecture diagram, Retrieval Approach section, and 5 sample chunks from Milestone 3. I'll ask it to implement `embedder.py` that loads chunks, embeds with all-MiniLM-L6-v2, stores in ChromaDB with source metadata, and `retriever.py` that queries the vector store with top-k=5. I'll verify by running 3 of my evaluation questions and checking that retrieved chunks are relevant and have distance scores < 0.5.

**Milestone 5 — Generation and interface:**
I'll give Claude my Architecture diagram, Retrieval Approach, and Evaluation Plan, plus examples of the retrieved chunks from Milestone 4. I'll ask it to implement `generator.py` that calls Groq API with a grounding prompt (answer only from retrieved context), formats responses with source attribution, and `cli.py` for a command-line interface. I'll verify by running 2-3 queries end-to-end and checking that responses cite sources and don't hallucinate.
