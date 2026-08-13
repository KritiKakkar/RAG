# Simple RAG System

A Retrieval-Augmented Generation (RAG) system that enhances AI responses with information from your documents.

## Overview

This project implements a RAG system using:
- **LangChain** for orchestration and component integration
- **Pinecone** as the vector database for efficient similarity search
- **OpenAI or Google Gemini** for embeddings and text generation
- **Document Loaders** for handling various file formats

## Features

- Load and process documents from various formats (PDF, Markdown, text)
- Split documents into manageable chunks
- Generate embeddings using OpenAI or Google Gemini
- Store embeddings in Pinecone for efficient retrieval
- Answer questions based on your documents with citations to sources
- Configurable model selection through a simple config file

## Setup Instructions

### 1. Clone the repository

```bash
git clone <repository-url>
cd simple-rag
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file with the following content:

For OpenAI:
```
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=simple-rag
PINECONE_NAMESPACE=technical-docs
```

For Google Gemini:
```
GOOGLE_API_KEY=your_google_gemini_api_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=simple-rag
PINECONE_NAMESPACE=technical-docs
```

### 5. Configure your model provider

Edit `src/config.py` and set `USE_OPENAI` to either `True` (for OpenAI) or `False` (for Google Gemini).

### 6. Add your documents

Place your documents in the `data` directory. The system supports:
- PDF files (.pdf)
- Markdown files (.md)
- Text files (.txt)

## Usage

### 1. Process your documents

Run the ingestion script to process and index your documents:

```bash
python scripts/ingest.py
```

This will:
1. Load documents from the data directory
2. Split them into chunks
3. Generate embeddings
4. Store them in Pinecone

### 2. Ask questions about your documents

Run the query script to test retrieval quality (without LLM):

```bash
python scripts/query.py
```

For the complete RAG pipeline with LLM:

```bash
python scripts/rag_query.py "What is Apache Spark?"
```

## Configuration

Edit `src/config.py` to change:

- Switch between OpenAI and Google Gemini
- Chunk size and overlap for document splitting
- Model names for embeddings and chat
- Number of documents to retrieve (TOP_K)

## Troubleshooting

**API Key Issues**
- Ensure your API keys are correctly set in the `.env` file
- For Google Gemini, make sure you have API access enabled

**Document Processing Issues**
- Check unsupported file formats
- Try adjusting chunk sizes in `config.py`

**Retrieval Quality Issues**
- Increase TOP_K in `config.py` to retrieve more documents
- Adjust the chunk size and overlap to find the optimal settings for your documents
- Use the evaluation dataset to benchmark different configurations

## Evaluation and Metrics

The project includes an evaluation framework to measure retrieval quality:

- **Recall@K**: Did the expected source appear in the top K results?
- **Precision@K**: How many retrieved documents were relevant?
- **MRR**: Mean Reciprocal Rank - how high was the first relevant document ranked?

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

Built with [LangChain](https://www.langchain.com/), [Pinecone](https://www.pinecone.io/), and either [OpenAI](https://openai.com/) or [Google Gemini](https://ai.google.dev/).