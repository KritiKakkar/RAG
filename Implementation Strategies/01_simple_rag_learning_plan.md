# Simple RAG Pipeline --- Learning & Hands-on Plan

## 1. Objective

Build a small but complete Retrieval-Augmented Generation (RAG)
application from scratch.

The purpose of this project is **learning the RAG fundamentals**, not
building a production system.

By the end, you should be able to explain and implement:

-   Document ingestion
-   Text extraction
-   Text cleaning
-   Chunking
-   Embeddings
-   Vector storage
-   Similarity search
-   Prompt construction
-   LLM generation
-   Source/citation handling
-   Basic RAG evaluation

------------------------------------------------------------------------

# 2. Target Architecture

``` text
                Documents
             PDF / TXT / MD
                    |
                    v
            +---------------+
            | Document      |
            | Loader        |
            +-------+-------+
                    |
                    v
            +---------------+
            | Text Cleaning |
            +-------+-------+
                    |
                    v
            +---------------+
            | Chunking      |
            +-------+-------+
                    |
                    v
            +---------------+
            | Embeddings    |
            +-------+-------+
                    |
                    v
            +---------------+
            | Vector Store  |
            | FAISS         |
            +-------+-------+
                    |
                    |
User Question ------+
                    |
                    v
            +---------------+
            | Similarity    |
            | Search        |
            +-------+-------+
                    |
                    v
            +---------------+
            | Prompt +      |
            | Retrieved     |
            | Context       |
            +-------+-------+
                    |
                    v
            +---------------+
            | LLM           |
            +-------+-------+
                    |
                    v
          Answer + Sources
```

------------------------------------------------------------------------

# 3. Recommended Technology Stack

Keep the first project intentionally simple.

  Component         Technology
  ----------------- -----------------------------------------------
  Language          Python
  LLM               OpenAI-compatible API or another hosted LLM
  Embeddings        Sentence Transformers or hosted embedding API
  Vector Store      FAISS
  API               Optional FastAPI
  UI                Optional Streamlit
  Environment       Python virtual environment
  Testing           pytest
  Version Control   Git + GitHub

Do **not** start with multiple frameworks. First understand the RAG
mechanics.

------------------------------------------------------------------------

# 4. Learning Outcomes

After completing this project, you should be able to answer:

### RAG Fundamentals

-   What problem does RAG solve?
-   Why not simply put the entire document into the LLM?
-   What is an embedding?
-   What is vector similarity?
-   What is cosine similarity?
-   What is a vector database?
-   What is chunking?
-   What is the difference between retrieval and generation?
-   What is hallucination?
-   What is grounding?

### Retrieval

-   How are relevant chunks retrieved?
-   What does Top-K mean?
-   How does similarity search work?
-   How does chunk size affect retrieval quality?

### Generation

-   How does retrieved context get passed to the LLM?
-   How do you prevent the model from answering outside the supplied
    context?
-   How do you return source references?

------------------------------------------------------------------------

# 5. Project Scope

Use a small document collection.

Suggested dataset:

``` text
data/
├── spark/
│   ├── spark_overview.pdf
│   ├── spark_sql.pdf
│   └── spark_tuning.pdf
│
├── kafka/
│   ├── kafka_basics.pdf
│   └── kafka_operations.pdf
│
└── python/
    ├── python_basics.md
    └── python_best_practices.md
```

You can also use your own technical notes.

The initial system should answer questions such as:

``` text
What is Spark Adaptive Query Execution?

How does Kafka partitioning work?

What are common Spark performance optimization techniques?

What is the difference between repartition and coalesce?
```

------------------------------------------------------------------------

# 6. Phase 1 --- Environment Setup

## Tasks

1.  Create a GitHub repository.
2.  Create a Python virtual environment.
3.  Create `requirements.txt`.
4.  Add `.env` for API keys.
5.  Add `.gitignore`.
6.  Create the initial project structure.

Suggested structure:

``` text
simple-rag/
│
├── data/
├── notebooks/
├── src/
│   ├── loaders/
│   ├── chunking/
│   ├── embeddings/
│   ├── retrieval/
│   ├── generation/
│   └── pipeline.py
│
├── tests/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Deliverable

A clean GitHub repository with a working Python environment.

------------------------------------------------------------------------

# 7. Phase 2 --- Document Ingestion

## Learn

Understand:

-   PDF parsing
-   Text extraction
-   Document metadata
-   File handling
-   Error handling

Implement:

``` text
PDF
 ↓
Text
 ↓
Document object
```

Each document should preserve metadata:

``` json
{
  "text": "...",
  "source": "spark_tuning.pdf",
  "page": 12
}
```

## Hands-on tasks

-   Load one PDF.
-   Print extracted text.
-   Extract multiple PDFs.
-   Preserve filename.
-   Preserve page number.
-   Handle empty or malformed documents.

## Deliverable

A reusable document loader.

------------------------------------------------------------------------

# 8. Phase 3 --- Text Cleaning

Implement basic cleaning:

-   Remove excessive whitespace
-   Remove repeated headers/footers where possible
-   Normalize line breaks
-   Remove obvious extraction artifacts
-   Preserve meaningful section boundaries

Do not over-clean the documents.

The goal is to preserve information that retrieval needs.

------------------------------------------------------------------------

# 9. Phase 4 --- Chunking

This is one of the most important learning stages.

Start with:

``` text
chunk_size = 500
chunk_overlap = 100
```

Then experiment with:

``` text
chunk_size = 250
chunk_size = 500
chunk_size = 1000
```

Compare results.

## Understand

### Chunk size

Too small:

``` text
Important context gets separated.
```

Too large:

``` text
Irrelevant information enters the context.
```

### Chunk overlap

Overlap helps preserve information that crosses chunk boundaries.

## Deliverable

A chunker that outputs:

``` json
{
  "chunk_id": "spark_tuning_12_03",
  "text": "...",
  "source": "spark_tuning.pdf",
  "page": 12
}
```

------------------------------------------------------------------------

# 10. Phase 5 --- Embeddings

Learn:

> Text → vector representation

Example:

``` text
"What is Spark AQE?"
        |
        v
[0.12, -0.48, 0.91, ...]
```

Implement:

``` text
chunks
  ↓
embedding model
  ↓
vectors
```

Understand:

-   Embedding dimensions
-   Semantic similarity
-   Cosine similarity
-   Why similar meanings produce nearby vectors

## Experiment

Compare embeddings for:

``` text
"Spark memory optimization"
"How can I reduce Spark memory usage?"
"Kafka consumer groups"
```

Observe similarity scores.

------------------------------------------------------------------------

# 11. Phase 6 --- Vector Store

Use FAISS.

Store:

``` text
Vector
+
Chunk ID
+
Metadata
```

Implement:

``` text
add_documents()
search(query, top_k)
```

Test:

``` text
query = "How do I optimize Spark memory?"

top_k = 5
```

Inspect the retrieved chunks manually.

This is important.

Do not immediately send results to the LLM.

First understand whether the retriever is actually retrieving the
correct information.

------------------------------------------------------------------------

# 12. Phase 7 --- Retrieval Pipeline

Implement:

``` text
Question
   |
   v
Query Embedding
   |
   v
Vector Search
   |
   v
Top-K Chunks
```

Return:

``` json
{
  "query": "...",
  "results": [
    {
      "score": 0.87,
      "source": "spark_tuning.pdf",
      "page": 12,
      "text": "..."
    }
  ]
}
```

## Experiments

Try:

``` text
Top K = 1
Top K = 3
Top K = 5
Top K = 10
```

Observe how retrieval quality changes.

------------------------------------------------------------------------

# 13. Phase 8 --- Prompt + LLM

Create a grounded prompt:

``` text
You are a technical assistant.

Answer the user's question using ONLY the
provided context.

If the answer cannot be found in the context,
say that the information is not available.

Context:
{retrieved_context}

Question:
{question}

Answer:
```

Pipeline:

``` text
Question
   ↓
Retriever
   ↓
Top K chunks
   ↓
Prompt construction
   ↓
LLM
   ↓
Answer
```

------------------------------------------------------------------------

# 14. Phase 9 --- Source Citations

The answer should not just be:

``` text
AQE improves Spark query execution...
```

It should provide:

``` text
AQE improves Spark query execution by dynamically
optimizing query plans during runtime.

Sources:
1. spark_tuning.pdf — Page 12
2. spark_sql.pdf — Page 28
```

This teaches an important production RAG concept:

> The answer should be traceable back to retrieved evidence.

------------------------------------------------------------------------

# 15. Phase 10 --- Build the Complete Pipeline

Create one function:

``` python
def ask(question):
    ...
```

Internally:

``` text
question
   ↓
embed
   ↓
retrieve
   ↓
build_context
   ↓
build_prompt
   ↓
LLM
   ↓
answer + sources
```

Example:

``` text
User:
Why does increasing Spark partitions sometimes improve performance?

System:
Increasing partitions can improve parallelism, but excessive
partitioning can introduce scheduling and shuffle overhead.

Sources:
- spark_tuning.pdf, page 15
- spark_overview.pdf, page 31
```

------------------------------------------------------------------------

# 16. Phase 11 --- Basic Evaluation

Create a small evaluation dataset:

``` json
[
  {
    "question": "What is AQE?",
    "expected_source": "spark_tuning.pdf"
  },
  {
    "question": "What is Kafka partitioning?",
    "expected_source": "kafka_basics.pdf"
  }
]
```

Measure at least:

-   Retrieved source correctness
-   Answer correctness
-   Unsupported answers
-   Retrieval failure cases

Manually inspect 20--30 questions.

------------------------------------------------------------------------

# 17. Phase 12 --- Add a Small UI

Optional.

Use Streamlit:

``` text
+--------------------------------------+
|        Simple RAG Assistant          |
+--------------------------------------+
| Ask a question                       |
|                                      |
| [ What is Spark AQE?              ] |
|                                      |
|              [Ask]                   |
+--------------------------------------+
| Answer                               |
| ...                                  |
|                                      |
| Sources                              |
| spark_tuning.pdf — Page 12           |
+--------------------------------------+
```

Do this only after the backend works.

------------------------------------------------------------------------

# 18. Final Deliverables

Your repository should contain:

-   Working ingestion pipeline
-   Chunking implementation
-   Embedding implementation
-   FAISS vector index
-   Retriever
-   LLM generation
-   Source citations
-   Basic evaluation dataset
-   Optional Streamlit UI
-   Unit tests
-   README
-   Architecture diagram

------------------------------------------------------------------------

# 19. Definition of Done

You are done when you can demonstrate:

``` text
PDF
 ↓
Extract
 ↓
Clean
 ↓
Chunk
 ↓
Embed
 ↓
FAISS
 ↓
Retrieve
 ↓
Prompt
 ↓
LLM
 ↓
Answer + Sources
```

And explain every step without relying on framework abstractions.

------------------------------------------------------------------------

# 20. What to Learn After This

Once this project works, move to:

``` text
Simple RAG
   ↓
Better Chunking
   ↓
Hybrid Search
   ↓
Reranking
   ↓
Query Rewriting
   ↓
Evaluation
   ↓
Production RAG
   ↓
Agentic RAG
```

The next project should be your **Data Engineering Copilot**.
