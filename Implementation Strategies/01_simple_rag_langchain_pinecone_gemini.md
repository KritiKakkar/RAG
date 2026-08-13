# Simple RAG Pipeline — LangChain + Pinecone + Google Gemini

## Objective

Implement the Simple RAG learning project using:

-   **LangChain** for orchestration and integrations
-   **Pinecone** as the managed vector database
-   **Google Gemini Embeddings** for vector representations
-   **Google Gemini Chat Model** for generation
-   **PyPDF** and LangChain loaders for documents

The learning goal is to understand both **RAG concepts** and how
LangChain connects the components.

------------------------------------------------------------------------

## Target Architecture

``` text
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
Google Gemini Embeddings
      |
      v
Pinecone Vector Index
      |
      |
User Question
      |
      v
Query Embedding
      |
      v
Pinecone Similarity Search
      |
      v
Top-K Documents
      |
      v
Prompt + Retrieved Context
      |
      v
Google Gemini Chat Model
      |
      v
Answer + Sources
```

------------------------------------------------------------------------

# 1. Project Structure

``` text
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
├── .env
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

------------------------------------------------------------------------

# 2. Install Dependencies

Create a virtual environment:

``` bash
python -m venv .venv
source .venv/bin/activate
```

Install:

``` bash
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

Optional later:

``` bash
pip install fastapi uvicorn streamlit pytest
```

Keep provider integrations in their dedicated LangChain packages. For
example, use `langchain-openai` for OpenAI and `langchain-pinecone` for
Pinecone.

------------------------------------------------------------------------

# 3. Environment Variables

Create `.env`:

``` text
GOOGLE_API_KEY=your_google_gemini_api_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=simple-rag
PINECONE_NAMESPACE=technical-docs
```

Create `.env.example` with the same keys but no secrets.

Never commit `.env`.

### Getting a Google Gemini API key

Create a Gemini API key using **Google AI Studio** and put it in `.env`:

```text
GOOGLE_API_KEY=your_google_gemini_api_key
```

The LangChain Google Gemini integration reads this environment variable.
Do not hard-code the API key in Python or commit it to GitHub.

The generation model and embedding model are configured separately:

```text
Gemini Embedding Model
        ↓
Document / Query → vectors
        ↓
Pinecone

Gemini Chat Model
        ↓
Retrieved context + question → answer
```

------------------------------------------------------------------------

# 4. Configuration

`src/config.py`

``` python
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

EMBEDDING_MODEL = "models/gemini-embedding-001"
CHAT_MODEL = "gemini-2.5-flash"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 5
```

------------------------------------------------------------------------

# 5. Load Documents

`src/loaders.py`

``` python
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

Inspect the output before proceeding:

``` python
documents = load_documents("data")

print(len(documents))

for doc in documents[:3]:
    print(doc.metadata)
    print(doc.page_content[:500])
```

A LangChain `Document` contains:

``` text
page_content
metadata
```

Metadata is important because it will later support filtering and
citations.

------------------------------------------------------------------------

# 6. Chunking

`src/splitter.py`

``` python
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
```

Experiment with:

``` text
300
500
800
1200
```

Do not assume one chunk size is universally best.

------------------------------------------------------------------------

# 7. Add Chunk Metadata

``` python
def add_chunk_metadata(chunks):

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index

    return chunks
```

Example metadata:

``` json
{
  "source_file": "spark_tuning.pdf",
  "file_type": ".pdf",
  "page": 5,
  "chunk_id": 42
}
```

------------------------------------------------------------------------

# 8. Embeddings

`src/embeddings.py`

``` python
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import EMBEDDING_MODEL


def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
    )
```

Conceptually:

``` text
"What is Spark AQE?"
        |
        v
Embedding model
        |
        v
[0.12, -0.48, 0.91, ...]
```

The vector is stored and searched in Pinecone.

------------------------------------------------------------------------

# 9. Create Pinecone Index

`src/vector_store.py`

``` python
from pinecone import Pinecone, ServerlessSpec

from langchain_pinecone import PineconeVectorStore

from src.config import (
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
)


def get_pinecone_client():
    import os

    return Pinecone(
        api_key=os.environ["PINECONE_API_KEY"]
    )


def create_index_if_needed(
    dimension=3072,
):
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

Important:

The Pinecone index dimension must match the output dimension of the
embedding model. If you change embedding models, verify the model's
dimension and use a compatible index.

------------------------------------------------------------------------

# 10. Ingestion Pipeline

`scripts/ingest.py`

``` python
from src.loaders import load_documents
from src.splitter import split_documents
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

``` bash
python scripts/ingest.py
```

------------------------------------------------------------------------

# 11. Pinecone Namespace

The project uses:

``` text
technical-docs
```

as the namespace.

Conceptually:

``` text
Pinecone Index
|
├── technical-docs
|      ├── Spark chunks
|      ├── Kafka chunks
|      └── Python chunks
|
└── future namespace
```

Namespaces are useful when you need logical separation within an index.

------------------------------------------------------------------------

# 12. Build the Retriever

`src/retriever.py`

``` python
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

``` text
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

------------------------------------------------------------------------

# 13. Test Retrieval Before Adding an LLM

This is one of the most important learning steps.

`scripts/query.py`

``` python
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

``` bash
python scripts/query.py
```

Your first question should be:

> **Did Pinecone retrieve the correct information?**

Not:

> Does the LLM give a nice answer?

------------------------------------------------------------------------

# 14. Inspect Similarity Results

For debugging:

``` python
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

Experiment with:

``` text
K = 1
K = 3
K = 5
K = 10
```

Inspect the actual retrieved chunks.

------------------------------------------------------------------------

# 15. Prompt

`src/prompts.py`

``` python
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

------------------------------------------------------------------------

# 16. Chat Model

`src/generator.py`

``` python
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import CHAT_MODEL


def get_llm():

    return ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        temperature=0,
    )
```

------------------------------------------------------------------------

# 17. Format Documents With Sources

Use metadata in the context:

``` python
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

This makes the retrieved evidence traceable.

------------------------------------------------------------------------

# 18. Complete RAG Pipeline

`src/rag_pipeline.py`

``` python
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

------------------------------------------------------------------------

# 19. Run the Complete RAG System

``` python
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

``` text
ANSWER

Adaptive Query Execution allows Spark
to use runtime statistics to optimize
parts of a query plan.

SOURCES

- spark_tuning.pdf page 4
- spark_overview.pdf page 6
```

------------------------------------------------------------------------

# 20. Metadata Filtering

One major benefit of a vector database is combining semantic retrieval
with metadata filtering.

For example:

``` python
results = vector_store.similarity_search(
    question,
    k=5,
    filter={
        "file_type": ".pdf"
    },
)
```

Later add metadata such as:

``` text
technology = spark
document_type = tutorial
```

Then filter:

``` python
filter={
    "technology": "spark"
}
```

This teaches an important production RAG pattern:

``` text
Semantic Search
       +
Metadata Filtering
       =
Better Retrieval
```

------------------------------------------------------------------------

# 21. Evaluation Dataset

Create:

``` text
evaluation/
└── questions.json
```

Example:

``` json
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

``` text
10 Spark questions
10 Kafka questions
10 Python questions
```

------------------------------------------------------------------------

# 22. Retrieval Metrics

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

Do not put fabricated results in the GitHub README. Record your actual
measurements.

------------------------------------------------------------------------

# 23. Critical "No Answer" Test

Ask:

``` text
How does Snowflake Time Travel work?
```

Your dataset contains no Snowflake documentation.

Expected behavior:

``` text
I don't have enough information in
the provided documents.
```

This tests whether your application is grounded in retrieved evidence.

------------------------------------------------------------------------

# 24. LangChain LCEL Version

Once the manual implementation is understood, refactor toward
LangChain's composable Runnable architecture.

``` python
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

``` python
answer = chain.invoke(
    {
        "question":
            "What is Spark AQE?"
    }
)
```

Learn this **after** understanding the explicit pipeline.

------------------------------------------------------------------------

# 25. Simple CLI

Build a CLI:

``` text
==================================================
Simple RAG Assistant
==================================================

Ask a question or type 'exit'

You: What is Spark AQE?

Assistant:
Adaptive Query Execution...

Sources:
1. spark_tuning.pdf — Page 4
2. spark_overview.pdf — Page 6

You:
```

This gives you a simple portfolio demo before building a UI.

------------------------------------------------------------------------

# 26. Optional Streamlit UI

After the backend works:

``` text
+----------------------------------------------+
|             Simple RAG Assistant             |
+----------------------------------------------+
| Ask a technical question                     |
|                                              |
| [ What is Spark AQE?                       ] |
|                                              |
|                    [ ASK ]                   |
+----------------------------------------------+
| Answer                                       |
|                                              |
| Adaptive Query Execution...                  |
|                                              |
| Sources                                      |
| spark_tuning.pdf — Page 4                    |
| spark_overview.pdf — Page 6                  |
+----------------------------------------------+
```

Do not build the UI first.

------------------------------------------------------------------------

# 27. Optional LangSmith

After the basic system works, add tracing/evaluation.

Trace:

``` text
Question
   |
   v
Retriever
   |
   v
Pinecone
   |
   v
Retrieved Documents
   |
   v
Prompt
   |
   v
LLM
   |
   v
Answer
```

Use tracing to investigate:

-   retrieval quality
-   latency
-   prompts
-   model responses
-   failures
-   evaluation results

------------------------------------------------------------------------

# 28. What Not to Add Yet

Do not immediately add:

``` text
Agents
LangGraph
Hybrid Search
Reranking
Graph RAG
Multi-agent systems
Memory
```

First make this reliable:

``` text
Documents
   ↓
LangChain Loader
   ↓
Chunking
   ↓
Embeddings
   ↓
Pinecone
   ↓
Retriever
   ↓
Prompt
   ↓
LLM
   ↓
Answer + Sources
```

Then upgrade it.

------------------------------------------------------------------------

# 29. Learning Milestones

## Milestone 1 --- LangChain

Understand:

``` text
Document
DocumentLoader
TextSplitter
Embeddings
VectorStore
Retriever
Prompt
ChatModel
OutputParser
Runnable / LCEL
```

## Milestone 2 --- Pinecone

Understand:

``` text
Index
Vector
Dimension
Metric
Namespace
Metadata
Upsert
Query
Top-K
Similarity
```

## Milestone 3 --- RAG

Understand:

``` text
Ingestion
Chunking
Embedding
Retrieval
Context
Generation
Grounding
Citation
Evaluation
```

------------------------------------------------------------------------

# 30. Final Architecture

``` text
                 DOCUMENTS
             PDF / Markdown
                    |
                    v
          ┌──────────────────┐
          │ LangChain Loader │
          └────────┬─────────┘
                   |
                   v
          ┌──────────────────┐
          │ Text Splitter    │
          └────────┬─────────┘
                   |
                   v
          ┌──────────────────┐
          │ Google Gemini Embedding │
          └────────┬─────────┘
                   |
                   v
          ┌──────────────────┐
          │    Pinecone      │
          │   Vector Store   │
          └────────┬─────────┘
                   |
                   |
             USER QUESTION
                   |
                   v
          ┌──────────────────┐
          │ LangChain        │
          │ Retriever        │
          └────────┬─────────┘
                   |
                   v
             TOP-K CHUNKS
                   |
                   v
          ┌──────────────────┐
          │ Prompt Template  │
          └────────┬─────────┘
                   |
                   v
          ┌──────────────────┐
          │ Google Gemini ChatModel │
          └────────┬─────────┘
                   |
                   v
          ANSWER + SOURCES
```

------------------------------------------------------------------------

# 31. Recommended Learning Sequence

``` text
DAY 1
Environment + API keys
        ↓
DAY 2
LangChain document loaders
        ↓
DAY 3
Chunking experiments
        ↓
DAY 4
Google Gemini embeddings
        ↓
DAY 5
Pinecone index + ingestion
        ↓
DAY 6
Pinecone retrieval
        ↓
DAY 7
LLM + prompt
        ↓
DAY 8
Complete RAG
        ↓
DAY 9
Citations + metadata
        ↓
DAY 10
Evaluation
        ↓
DAY 11
LangChain LCEL
        ↓
DAY 12
Streamlit
        ↓
DAY 13+
Hybrid search / reranking
```

------------------------------------------------------------------------

# 32. Key Learning Principle

Do not memorize:

``` python
vector_store.as_retriever()
```

Understand what happens underneath:

``` text
Question
   ↓
Embedding Model
   ↓
Query Vector
   ↓
Pinecone
   ↓
Similarity Search
   ↓
Top-K Vectors
   ↓
Metadata → Documents
   ↓
Context
   ↓
LLM
```

If you understand this flow, moving between Pinecone, FAISS, Qdrant,
Weaviate, or another vector store becomes much easier.

Likewise, understanding the RAG architecture makes switching between
LangChain, LlamaIndex, and custom implementations much easier.

------------------------------------------------------------------------

# 33. Gemini Troubleshooting

### API key not detected

Check that `.env` contains:

```text
GOOGLE_API_KEY=your_google_gemini_api_key
```

and that `load_dotenv()` runs before the Gemini integration is initialized.

### Model unavailable

Gemini model availability can vary by API access and provider updates. If
`gemini-2.5-flash` is unavailable, choose a currently available Gemini model
and update `CHAT_MODEL`.

### Pinecone dimension mismatch

The Pinecone index dimension must match the embedding vector dimension.

For the configuration in this guide:

```text
Embedding model:
models/gemini-embedding-001

Pinecone dimension:
3072
```

If you choose a different embedding model or configure a different output
dimension, recreate/use a Pinecone index with the matching dimension.

### Do not mix embedding models

If you already created your Pinecone index using OpenAI embeddings or another
embedding model, do **not** start inserting Gemini embeddings into that same
index. Create a new index with the appropriate dimension and re-ingest the
documents.

# 33. Next Upgrade

After this project is working:

``` text
Simple RAG
     ↓
LangChain + Pinecone
     ↓
Hybrid Retrieval
     ↓
Reranking
     ↓
Evaluation
     ↓
Query Rewriting
     ↓
Data Engineering Copilot
     ↓
Agentic RAG
```

That progression gives you both strong RAG fundamentals and a path
toward a portfolio-grade **AI Data Engineering** project.
