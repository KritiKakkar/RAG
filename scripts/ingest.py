import sys
import os
# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Add virtual environment site-packages to the Python path
sys.path.insert(0, '/home/kriti.kakkar/Desktop/AI/RAG/.venv/lib/python3.12/site-packages')

from src.loaders import load_document
from src.splitter import split_documents
from src.embeddings import get_embeddings
from src.vector_store import create_index_if_needed, get_vector_store

def main():
    print("Starting document ingestion process...")

    documents = load_document("../data")
    print(f"Documents Loaded: {len(documents)}")

    chunks = split_documents(documents)
    print(f"Chunks Generated: {len(chunks)}")

    embeddings = get_embeddings()
    print("Embeddings model initialized")

    create_index_if_needed()
    print("Pinecone index verified/created")

    vector_store = get_vector_store(embeddings)
    print("Vector store connected")

    print("Adding documents to vector store (this might take a while)...")
    vector_store.add_documents(chunks)

    print("✅ Documents successfully indexed in Pinecone!")


if __name__ == "__main__":
    main()