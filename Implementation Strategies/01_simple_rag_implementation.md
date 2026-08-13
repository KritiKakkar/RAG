# Simple RAG Pipeline Implementation Guide

## Objective

Implement a Retrieval-Augmented Generation (RAG) system to answer questions based on technical documentation using:

- **LangChain** for orchestration and component integration
- **Vector Database** (Pinecone) for embedding storage and retrieval
- **Embedding Model** for vector representations
- **LLM** for generation with retrieved context
- **Document Loaders** for text extraction

The learning goal is to understand both **RAG concepts** and how the components connect in a functional implementation.

---

## Target Architecture

```text
PDF / Markdown
      |
      v
LangChain Document Loaders
      |
      v
Document Objects + Metadata
      |
      v
Recursive Text Splitter
      |
      v
Chunks
      |
      v
Embedding Model
      |
      v
Vector Database
      |
      |
User Question
      |
      v
Query Embedding
      |
      v
Similarity Search
      |
      v
Top-K Documents
      |
      v
Prompt + Retrieved Context
      |
      v
LLM
      |
      v
Answer + Sources
```

---

# 1. Project Structure

```text
simple-rag/
|
├── data/
│   ├── spark/
│   │   ├── spark_overview.pdf
│   │   ├── spark_sql.pdf
│   │   └── spark_tuning.pdf
│   ├── kafka/
│   │   ├── kafka_basics.pdf
│   │   └── kafka_operations.pdf
│   └── python/
│       ├── python_basics.md
│       └── python_best_practices.md
|
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── loaders.py
│   ├── splitter.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── prompts.py
│   ├── generator.py
│   └── rag_pipeline.py
|
├── scripts/
│   ├── ingest.py
│   └── query.py
|
├── tests/
│   ├── test_splitter.py
│   └── test_retriever.py
|
├── evaluation/
│   └── questions.json
|
├── .env
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

# 2. Setup & Installation

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

## Choose Your Model Provider

### Option A: OpenAI

```bash
pip install \
    langchain \
    langchain-community \
    langchain-openai \
    langchain-pinecone \
    langchain-text-splitters \
    pinecone \
    pypdf \
    python-dotenv
```

### Option B: Google Gemini

```bash
pip install \
    langchain \
    langchain-community \
    langchain-google-genai \
    langchain-pinecone \
    langchain-text-splitters \
    pinecone \
    pypdf \
    python-dotenv
```

Optional packages for later:

```bash
pip install fastapi uvicorn streamlit pytest
```

---

# 3. Environment Variables

Create `.env`:

## For OpenAI
```text
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=simple-rag
PINECONE_NAMESPACE=technical-docs
```

## For Google Gemini
```text
GOOGLE_API_KEY=your_google_gemini_api_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=simple-rag
PINECONE_NAMESPACE=technical-docs
```

Create `.env.example` with the same keys but no secrets.

---

# 4. Configuration

`src/config.py`

```python
import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME",
    "simple-rag",
)

PINECONE_NAMESPACE = os.getenv(
    "PINECONE_NAMESPACE",
    "technical-docs",
)

# Choose your model configuration based on provider

# OpenAI Configuration
EMBEDDING_MODEL_OPENAI = "text-embedding-3-small"
CHAT_MODEL_OPENAI = "gpt-4o-mini"

# Google Gemini Configuration
EMBEDDING_MODEL_GEMINI = "models/gemini-embedding-001"
CHAT_MODEL_GEMINI = "gemini-2.5-flash"

# Configure which provider to use
USE_OPENAI = True  # Set to False to use Gemini

# Use the appropriate models based on the provider
EMBEDDING_MODEL = EMBEDDING_MODEL_OPENAI if USE_OPENAI else EMBEDDING_MODEL_GEMINI
CHAT_MODEL = CHAT_MODEL_OPENAI if USE_OPENAI else CHAT_MODEL_GEMINI

# Chunking parameters
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 5
```

---

# 5. Document Loading

`src/loaders.py`

```python
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)


def load_documents(data_dir: str):
    documents = []

    for path in Path(data_dir).rglob("*"):
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            loader = PyPDFLoader(str(path))
            docs = loader.load()

        elif suffix in {".md", ".txt"}:
            loader = TextLoader(
                str(path),
                encoding="utf-8",
            )
            docs = loader.load()

        else:
            continue

        for doc in docs:
            doc.metadata["source_file"] = path.name
            doc.metadata["file_type"] = suffix

        documents.extend(docs)

    return documents
```

To inspect the output:

```python
documents = load_documents("data")

print(len(documents))

for doc in documents[:3]:
    print(doc.metadata)
    print(doc.page_content[:500])
```

A LangChain `Document` contains:

```text
page_content
metadata
```

Metadata is crucial for source tracking and filtering.

---

# 6. Text Chunking

`src/splitter.py`

```python
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    return splitter.split_documents(documents)


def add_chunk_metadata(chunks):
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index

    return chunks
```

Experiment with different chunk sizes to find the optimal balance:

```text
300
500
800
1200
```

No chunk size is universally best; it depends on your documents and use case.

---

# 7. Embedding Configuration

`src/embeddings.py`

## For OpenAI:

```python
from langchain_openai import OpenAIEmbeddings

from src.config import EMBEDDING_MODEL


def get_embeddings():
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
    )
```

## For Google Gemini:

```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import EMBEDDING_MODEL


def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
    )
```

Conceptually, embeddings convert text into vectors:

```text
"What is Spark AQE?"
        |
        v
Embedding model
        |
        v
[0.12, -0.48, 0.91, ...]
```

---

# 8. Vector Database Setup

`src/vector_store.py`

```python
from pinecone import Pinecone, ServerlessSpec

from langchain_pinecone import PineconeVectorStore

from src.config import (
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
    USE_OPENAI,
)


def get_pinecone_client():
    import os

    return Pinecone(
        api_key=os.environ["PINECONE_API_KEY"]
    )


def create_index_if_needed():
    # OpenAI embedding dimension is 1536
    # Gemini embedding dimension is 3072
    dimension = 1536 if USE_OPENAI else 3072
    
    pc = get_pinecone_client()

    existing = [
        item["name"]
        for item in pc.list_indexes()
    ]

    if PINECONE_INDEX_NAME not in existing:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )

    return pc


def get_vector_store(embeddings):
    return PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
        namespace=PINECONE_NAMESPACE,
    )
```

**Important**: The Pinecone index dimension must match your embedding model's output dimension. OpenAI's text-embedding-3-small has 1536 dimensions, while Gemini's embedding model has 3072 dimensions.

---

# 9. Ingestion Pipeline

`scripts/ingest.py`

```python
from src.loaders import load_documents
from src.splitter import split_documents, add_chunk_metadata
from src.embeddings import get_embeddings

from src.vector_store import (
    create_index_if_needed,
    get_vector_store,
)


def main():
    documents = load_documents("data")

    print(
        f"Documents loaded: {len(documents)}"
    )

    chunks = split_documents(documents)
    chunks = add_chunk_metadata(chunks)

    print(
        f"Chunks generated: {len(chunks)}"
    )

    embeddings = get_embeddings()

    create_index_if_needed()

    vector_store = get_vector_store(
        embeddings
    )

    vector_store.add_documents(
        chunks
    )

    print(
        "Documents indexed in Pinecone."
    )


if __name__ == "__main__":
    main()
```

Run:

```bash
python scripts/ingest.py
```

---

# 10. Retrieval Setup

`src/retriever.py`

```python
from src.config import TOP_K


def get_retriever(vector_store):
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": TOP_K,
        },
    )
```

The flow is now:

```text
Question
   |
   v
Embedding
   |
   v
Pinecone
   |
   v
Top-K chunks
```

---

# 11. Testing Retrieval

`scripts/query.py`

```python
from src.embeddings import get_embeddings
from src.vector_store import get_vector_store
from src.retriever import get_retriever


def main():
    embeddings = get_embeddings()

    vector_store = get_vector_store(
        embeddings
    )

    retriever = get_retriever(
        vector_store
    )

    question = (
        "What is Spark Adaptive "
        "Query Execution?"
    )

    documents = retriever.invoke(
        question
    )

    for index, doc in enumerate(
        documents,
        start=1,
    ):
        print("=" * 80)
        print(f"RESULT {index}")
        print("Metadata:", doc.metadata)
        print("Content:")
        print(doc.page_content)


if __name__ == "__main__":
    main()
```

Run:

```bash
python scripts/query.py
```

Your first evaluation question should be:

> **Did Pinecone retrieve the correct information?**

Not:

> Does the LLM give a nice answer?

---

# 12. Debugging Similarity Search

For debugging:

```python
results = vector_store.similarity_search_with_score(
    question,
    k=5,
)

for doc, score in results:
    print("Score:", score)
    print(
        "Source:",
        doc.metadata.get("source_file"),
    )
    print(doc.page_content)
```

Experiment with different K values:

```text
K = 1
K = 3
K = 5
K = 10
```

Inspect each retrieved chunk to evaluate quality.

---

# 13. Prompt Template

`src/prompts.py`

```python
from langchain_core.prompts import (
    ChatPromptTemplate,
)


RAG_PROMPT = ChatPromptTemplate.from_template(
    """
You are a technical assistant.

Answer the question using only the
provided context.

If the answer cannot be determined
from the context, say:

"I don't have enough information in
the provided documents."

Do not invent technical details.

Context:
{context}

Question:
{question}

Answer:
"""
)
```

---

# 14. LLM Configuration

`src/generator.py`

## For OpenAI:

```python
from langchain_openai import ChatOpenAI

from src.config import CHAT_MODEL


def get_llm():
    return ChatOpenAI(
        model=CHAT_MODEL,
        temperature=0,
    )
```

## For Google Gemini:

```python
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import CHAT_MODEL


def get_llm():
    return ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        temperature=0,
    )
```

---

# 15. Document Formatting with Sources

```python
def format_docs(documents):
    formatted = []

    for doc in documents:
        source = doc.metadata.get(
            "source_file",
            "unknown",
        )

        page = doc.metadata.get(
            "page",
            "unknown",
        )

        formatted.append(
            f"""
SOURCE: {source}
PAGE: {page}

CONTENT:
{doc.page_content}
"""
        )

    return "\n\n".join(formatted)
```

This makes retrieved evidence traceable.

---

# 16. Complete RAG Pipeline

`src/rag_pipeline.py`

```python
from src.prompts import RAG_PROMPT


def build_rag_pipeline(
    retriever,
    llm,
):
    def ask(question):
        documents = retriever.invoke(
            question
        )

        context = format_docs(
            documents
        )

        prompt = RAG_PROMPT.invoke(
            {
                "context": context,
                "question": question,
            }
        )

        response = llm.invoke(
            prompt
        )

        return {
            "answer": response.content,
            "documents": documents,
        }

    return ask


def format_docs(documents):
    formatted = []

    for doc in documents:
        source = doc.metadata.get(
            "source_file",
            "unknown",
        )

        page = doc.metadata.get(
            "page",
            "unknown",
        )

        formatted.append(
            f"""
SOURCE: {source}
PAGE: {page}

CONTENT:
{doc.page_content}
"""
        )

    return "\n\n".join(formatted)
```

---

# 17. Running the Complete RAG System

```python
from src.embeddings import get_embeddings
from src.vector_store import get_vector_store
from src.retriever import get_retriever
from src.generator import get_llm
from src.rag_pipeline import (
    build_rag_pipeline,
)


def main():
    embeddings = get_embeddings()

    vector_store = get_vector_store(
        embeddings
    )

    retriever = get_retriever(
        vector_store
    )

    llm = get_llm()

    rag = build_rag_pipeline(
        retriever,
        llm,
    )

    question = (
        "What is Spark Adaptive "
        "Query Execution?"
    )

    result = rag(question)

    print("\nANSWER")
    print(result["answer"])

    print("\nSOURCES")

    for doc in result["documents"]:
        print(
            "-",
            doc.metadata.get(
                "source_file"
            ),
            "page",
            doc.metadata.get("page"),
        )


if __name__ == "__main__":
    main()
```

Expected response structure:

```text
ANSWER

Adaptive Query Execution allows Spark
to use runtime statistics to optimize
parts of a query plan.

SOURCES

- spark_tuning.pdf page 4
- spark_overview.pdf page 6
```

---

# 18. Advanced: Metadata Filtering

One major benefit of vector databases is combining semantic retrieval with metadata filtering:

```python
results = vector_store.similarity_search(
    question,
    k=5,
    filter={
        "file_type": ".pdf"
    },
)
```

Later, add custom metadata such as:

```text
technology = spark
document_type = tutorial
```

Then filter:

```python
filter={
    "technology": "spark"
}
```

This powerful pattern combines:

```text
Semantic Search
       +
Metadata Filtering
       =
Better Retrieval
```

---

# 19. Evaluation Dataset

Create:

```text
evaluation/
└── questions.json
```

Example:

```json
[
  {
    "question": "What is Spark AQE?",
    "expected_sources": [
      "spark_tuning.pdf"
    ]
  },
  {
    "question": "What is Kafka consumer lag?",
    "expected_sources": [
      "kafka_operations.pdf"
    ]
  },
  {
    "question": "Why should production Python use logging?",
    "expected_sources": [
      "python_best_practices.md"
    ]
  }
]
```

Create at least:

```text
10 Spark questions
10 Kafka questions
10 Python questions
```

---

# 20. Retrieval Metrics

Measure:

### Recall@K

Did the expected source appear in the top K?

### Precision@K

How many retrieved documents were relevant?

### MRR

How high was the first relevant document ranked?

Example experiment:

  Configuration     Recall@5
  --------------- ----------
  Chunk 300          measure
  Chunk 500          measure
  Chunk 800          measure
  Chunk 1200         measure

Do not fabricate results - record your actual measurements.

---

# 21. Critical "No Answer" Test

Ask:

```text
How does Snowflake Time Travel work?
```

Your dataset contains no Snowflake documentation.

Expected behavior:

```text
I don't have enough information in
the provided documents.
```

This tests whether your application is properly grounded in the retrieved evidence.

---

# 22. LangChain LCEL Version

Once you understand the manual implementation, refactor to LangChain's composable Runnable architecture:

```python
from operator import itemgetter

from langchain_core.output_parsers import (
    StrOutputParser,
)
from langchain_core.runnables import (
    RunnablePassthrough,
)


def format_docs(docs):
    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


chain = (
    {
        "context": (
            itemgetter("question")
            | retriever
            | format_docs
        ),
        "question": itemgetter("question"),
    }
    | RAG_PROMPT
    | llm
    | StrOutputParser()
)
```

Invoke:

```python
answer = chain.invoke(
    {
        "question":
            "What is Spark AQE?"
    }
)
```

Learn this **after** understanding the explicit pipeline.

---

# 23. Simple CLI Interface

Build a CLI:

```text
==================================================
Simple RAG Assistant
==================================================

Ask a question or type 'exit'

You: What is Spark AQE?