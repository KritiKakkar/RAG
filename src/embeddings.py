import os
import json
import requests
import numpy as np

class OpenRouterEmbeddings:
    """Embeddings provider using OpenRouter API."""

    def __init__(self, api_key=None):
        """Initialize with API key from args or environment."""
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key not found. Set OPENROUTER_API_KEY in .env file.")

        self.model_name = "nvidia/nemotron-3-embed-1b:free"
        self.dimensions = 2048  # Dimensions for this model

        print(f"Using OpenRouter embeddings with model: {self.model_name}")

    def embed_documents(self, texts):
        """Generate embeddings for a list of documents."""
        all_embeddings = []

        # Process in batches to avoid timeouts on large document sets
        batch_size = 20
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            print(f"Generating embeddings for batch {i//batch_size + 1} ({len(batch)} texts)")

            embeddings = self._get_embeddings_batch(batch)
            all_embeddings.extend(embeddings)

        return all_embeddings

    def embed_query(self, text):
        """Generate embeddings for a single query."""
        return self._get_embeddings_batch([text])[0]

    def _get_embeddings_batch(self, texts):
        """Call OpenRouter API to get embeddings."""
        response = requests.post(
            url="https://openrouter.ai/api/v1/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "http://localhost:8000",
                "X-OpenRouter-Title": "RAG System",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "model": self.model_name,
                "input": texts
            })
        )

        if response.status_code != 200:
            error_msg = f"OpenRouter API error: {response.status_code} {response.text}"
            print(error_msg)
            raise ValueError(error_msg)

        result = response.json()
        embeddings = [item["embedding"] for item in result["data"]]

        return embeddings


class DummyEmbeddings:
    """A fallback embeddings class that creates random embeddings."""

    def __init__(self, dimension=2048):
        self.dimension = dimension
        print("Using dummy embeddings generator (for testing only)")

    def embed_documents(self, texts):
        """Generate random embeddings for documents."""
        print(f"Generating dummy embeddings for {len(texts)} texts")
        embeddings = []
        for text in texts:
            # Use text hash for consistent embeddings
            np.random.seed(hash(text) % 2**32)
            embedding = np.random.randn(self.dimension)
            # Normalize to unit length
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding.tolist())
        return embeddings

    def embed_query(self, text):
        """Generate random embedding for a query."""
        return self.embed_documents([text])[0]


def get_embeddings():
    """Factory function to get the appropriate embeddings provider."""
    try:
        # First try to use OpenRouter
        return OpenRouterEmbeddings()
    except (ValueError, ImportError) as e:
        print(f"Warning: {str(e)}")
        print("Falling back to dummy embeddings")
        return DummyEmbeddings()