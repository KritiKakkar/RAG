# RAG System Explained: Making AI Smarter With Your Documents

## What is RAG?

**RAG** stands for **Retrieval-Augmented Generation**. Let's break down what that means in simple terms:

- **Retrieval**: Finding relevant information from your documents when needed
- **Augmented**: Adding this information to enhance the AI's knowledge
- **Generation**: Creating accurate, helpful responses based on this combined knowledge

Think of RAG like giving an AI assistant your own personal library. When you ask a question, it:
1. Searches your library for relevant information
2. Reads the most helpful sections
3. Uses both what it already knows AND what it just read to answer you

## Why Use RAG?

- **Up-to-date information**: The AI can access your latest documents, not just what it was trained on
- **Domain-specific knowledge**: It can answer questions about your specialized documents
- **Accuracy**: Responses are grounded in your actual documents, reducing "hallucinations" (made-up information)
- **Transparency**: You can see exactly which documents were used to generate an answer

## How Our RAG System Works

Our RAG system processes your documents and answers questions in a series of steps:

### 1. Document Processing (Done Once)

```
Your Documents → Loading → Splitting → Embedding → Storage
```

- **Loading**: We take your PDFs, text files, and other documents and convert them to a format the system can work with
- **Splitting**: Long documents are divided into smaller chunks that are easier to search through
- **Embedding**: Each chunk is converted into a special numerical representation (like a fingerprint)
- **Storage**: These "fingerprints" are stored in a special database called a vector store

### 2. Question Answering (Each Time You Ask)

```
Your Question → Embedding → Search → Retrieval → Answer Generation
```

- **Question Embedding**: Your question is converted into the same type of "fingerprint"
- **Search**: The system finds document chunks with similar "fingerprints" to your question
- **Retrieval**: The most relevant chunks are pulled from the database
- **Answer Generation**: An AI model reads these chunks and your question, then creates an answer

## Components of Our System

### 1. Document Loaders
Tools that read different file types (PDF, text, Markdown) and extract their content

### 2. Text Splitter
Breaks down long documents into manageable chunks while preserving meaning

### 3. Embedding Model
Converts text into number sequences (vectors) that capture meaning - we use OpenRouter API to access embedding models

### 4. Vector Store (Pinecone)
A specialized database that stores and searches through these vectors efficiently

### 5. Retriever
Finds the most relevant document chunks for each question

### 6. Language Model (via OpenRouter)
Generates helpful answers based on the retrieved information - we use OpenRouter API to access various language models

## How to Use the System

### Adding Your Documents

Run the ingest script to process and store your documents:

```
python scripts/ingest.py
```

Place your documents in the `data` folder before running this.

### Asking Questions

Ask questions about your documents:

```
python scripts/query_rag.py "Your question here?"
```

For interactive mode:

```
python scripts/query_rag.py --interactive
```

## Technical Architecture

```
                      DOCUMENT PROCESSING
                     ┌───────────────────┐
                     │                   │
                     │     Documents     │
                     │  (PDF, TXT, MD)   │
                     │                   │
                     └─────────┬─────────┘
                               │
                     ┌─────────▼─────────┐
                     │                   │
                     │  Document Loaders │
                     │                   │
                     └─────────┬─────────┘
                               │
                     ┌─────────▼─────────┐
                     │                   │
                     │   Text Splitter   │
                     │                   │
                     └─────────┬─────────┘
                               │
                     ┌─────────▼─────────┐
                     │                   │
                     │ Embedding Service │
                     │   (OpenRouter)    │
                     │                   │
                     └─────────┬─────────┘
                               │
                     ┌─────────▼─────────┐
                     │                   │
                     │   Vector Store    │
                     │    (Pinecone)     │
                     │                   │
                     └─────────┬─────────┘
                               │
                               ▼
                      QUESTION ANSWERING
                     ┌───────────────────┐
                     │                   │
                     │  User Question    │
                     │                   │
                     └─────────┬─────────┘
                               │
                     ┌─────────▼─────────┐
                     │                   │
                     │     Retriever     │
                     │                   │
                     └─────────┬─────────┘
                               │
                     ┌─────────▼─────────┐
                     │                   │
                     │  Retrieved Docs   │
                     │                   │
                     └─────────┬─────────┘
                               │
                     ┌─────────▼─────────┐
                     │                   │
                     │  Prompt Builder   │
                     │                   │
                     └─────────┬─────────┘
                               │
                     ┌─────────▼─────────┐
                     │                   │
                     │ Language Model    │
                     │   (OpenRouter)    │
                     │                   │
                     └─────────┬─────────┘
                               │
                     ┌─────────▼─────────┐
                     │                   │
                     │  Final Answer     │
                     │  + Sources        │
                     │                   │
                     └───────────────────┘
```

## Key Terms Explained

- **Embeddings**: Mathematical representations of text that capture meaning. Similar texts have similar embeddings, which allows the system to find related information.

- **Vector**: Another name for the mathematical representation (embedding) of text - essentially a list of numbers that represents the meaning of the text.

- **Vector Store**: A database designed to efficiently store and search through embeddings/vectors.

- **Chunks**: Smaller pieces of documents that are easier to search through and retrieve relevant information from.

- **Prompt**: Instructions given to the language model, including the retrieved documents and your question.

- **Context Window**: The amount of text a language model can consider at once. Chunking helps fit more relevant information into this window.

## Benefits of Our Implementation

- **Simple yet powerful**: Built on proven open-source technologies
- **Flexible**: Can work with various document types
- **Unified API access**: Uses OpenRouter to access multiple AI models with a single API key
- **Source transparency**: Always shows which documents were used to generate an answer
- **Modular design**: Each component can be improved or replaced independently

## Limitations

- **Knowledge is limited to your documents**: If information isn't in your documents, the system will say it doesn't know
- **Quality depends on your documents**: Better organized, well-written documents lead to better answers
- **Text-only for now**: Currently works best with text, not images or other media
- **Requires good chunking**: How documents are split affects retrieval quality

## Future Improvements

- Hybrid search combining keyword and semantic search
- Better handling of document structure and relationships
- Support for more document formats
- Improved answer generation with citations
- User feedback loop to improve retrieval quality

## Project Structure

```
simple-rag/
├── data/                  # Your documents go here
├── src/                   # Core system components
│   ├── config.py          # Configuration settings
│   ├── embeddings.py      # Embedding generation
│   ├── generator.py       # Answer generation
│   ├── loaders.py         # Document loading
│   ├── prompts.py         # Prompt templates
│   ├── rag_pipeline.py    # Main RAG pipeline
│   ├── retriever.py       # Document retrieval
│   ├── splitter.py        # Text splitting
│   └── vector_store.py    # Vector database interface
├── scripts/               # Runnable scripts
│   ├── ingest.py          # Process and store documents
│   └── query_rag.py       # Ask questions about documents
└── README.md              # This documentation
```

## Getting Started

1. Clone the repository
2. Put your documents in the `data` folder
3. Run `python scripts/ingest.py` to process your documents
4. Run `python scripts/query_rag.py "Your question?"` to start asking questions

For more details, see the installation and setup instructions in the main README.md file.

---

Built with ❤️ using OpenRouter API, Pinecone, and the power of modern AI.