import os
from dotenv import load_dotenv 

load_dotenv()

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE")

# OpenRouter model names
EMBEDDING_MODEL = "nvidia/nemotron-3-embed-1b:free"  # Free embedding model on OpenRouter
CHAT_MODEL = "nvidia/nemotron-3.5-lightning:free"    # Free chat model on OpenRouter

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3