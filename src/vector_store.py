import os
from pinecone import Pinecone, ServerlessSpec

from langchain_pinecone import PineconeVectorStore

from src.config import PINECONE_NAMESPACE, PINECONE_INDEX_NAME


def get_pinecone_client():
    return Pinecone(
        api_key=os.getenv("PINECONE_API_KEY"),
    )


def create_index_if_needed(dimensions = 2048):  
    client = get_pinecone_client()

    existing = [item["name"] for item in client.list_indexes()]

    if PINECONE_INDEX_NAME not in existing:
        client.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=dimensions,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ))

    return client


def get_vector_store(embeddings):
    return PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
        namespace=PINECONE_NAMESPACE,
    )