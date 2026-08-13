#!/usr/bin/env python3

import os
import sys
import json
import requests
from dotenv import load_dotenv

# Add the parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.embeddings import OpenRouterEmbeddings
from src.vector_store import get_vector_store

# Load environment variables
load_dotenv()

def query_documents(query_text, top_k=3):
    """
    Find relevant documents and generate a response using OpenRouter.

    Args:
        query_text: The user's question
        top_k: Number of most relevant documents to retrieve
    """
    try:
        print(f"Query: {query_text}")

        # Initialize embeddings
        embeddings = OpenRouterEmbeddings()

        # Get vector store
        vector_store = get_vector_store(embeddings)

        # Search for similar documents
        print(f"Searching for relevant documents...")
        similar_docs = vector_store.similarity_search(query_text, k=top_k)

        # Extract content from similar documents
        context_texts = [doc.page_content for doc in similar_docs]
        context = "\n\n---\n\n".join(context_texts)

        print(f"Found {len(similar_docs)} relevant documents")

        # Use OpenRouter to generate a response
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment")

        # Create system prompt with retrieved context
        messages = [
            {
                "role": "system",
                "content": f"""You are an assistant answering questions based on retrieved context.
Use the context below to answer the user's question. If the context doesn't contain
enough information to provide a complete answer, say so and answer with what you know.

Context:
{context}"""
            },
            {
                "role": "user",
                "content": query_text
            }
        ]

        # Call OpenRouter API
        print(f"Generating response...")
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "http://localhost:8000",
                "X-OpenRouter-Title": "RAG System",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "model": "nvidia/nemotron-3.5-lightning:free",  # Use a free model
                "messages": messages
            })
        )

        if response.status_code == 200:
            result = response.json()
            answer = result["choices"][0]["message"]["content"]
            model = result.get("model", "unknown")

            print("\n" + "-"*50)
            print(f"Response from {model}:")
            print("-"*50)
            print(answer)
        else:
            print(f"Error: {response.status_code} {response.text}")

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    # Check if documents have been indexed
    if len(sys.argv) < 2:
        print("Usage: python query_openrouter.py 'your question here'")
        sys.exit(1)

    query = ' '.join(sys.argv[1:])
    query_documents(query)