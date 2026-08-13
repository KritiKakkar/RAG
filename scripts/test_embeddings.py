#!/usr/bin/env python3

import os
import sys
import json
import requests
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment
api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    print("Error: OPENROUTER_API_KEY not found in .env file")
    sys.exit(1)

print(f"Found API key (first 5 chars: {api_key[:5]}...)")

# Example texts to embed
texts = [
    "This is a sample document about artificial intelligence.",
    "Machine learning is a subset of AI that involves training models on data.",
    "Natural language processing allows computers to understand human language."
]

print(f"\nGenerating embeddings for {len(texts)} texts...")

# Make the API request for embeddings
try:
    response = requests.post(
        url="https://openrouter.ai/api/v1/embeddings",
        headers={
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "http://localhost:8000",
            "X-OpenRouter-Title": "RAG System Test",
            "Content-Type": "application/json",
        },
        data=json.dumps({
            "model": "nvidia/llama-nemotron-embed-vl-1b-v2:free",
            "input": texts
        })
    )

    # Check if the request was successful
    if response.status_code == 200:
        result = response.json()
        embeddings = [item["embedding"] for item in result["data"]]

        print("\nEmbeddings Generation Successful! ✅")
        print(f"Model used: {result.get('model', 'unknown')}")
        print(f"Number of embeddings: {len(embeddings)}")
        print(f"Embedding dimensions: {len(embeddings[0])}")

        # Calculate similarity between embeddings
        def cosine_similarity(vec1, vec2):
            dot = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = sum(a * a for a in vec1) ** 0.5
            norm2 = sum(b * b for b in vec2) ** 0.5
            return dot / (norm1 * norm2)

        # Compare similarities
        print("\nSimilarity between texts:")
        print(f"Text 1 & Text 2: {cosine_similarity(embeddings[0], embeddings[1]):.4f}")
        print(f"Text 1 & Text 3: {cosine_similarity(embeddings[0], embeddings[2]):.4f}")
        print(f"Text 2 & Text 3: {cosine_similarity(embeddings[1], embeddings[2]):.4f}")

    else:
        print(f"\nEmbeddings API Test Failed! ❌ Status code: {response.status_code}")
        print(f"Error: {response.text}")

except Exception as e:
    print(f"\nEmbeddings API Test Failed! ❌")
    print(f"Exception: {str(e)}")