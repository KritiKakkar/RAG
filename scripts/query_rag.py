#!/usr/bin/env python3

import sys
import os
import argparse

# Add the parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.embeddings import get_embeddings
from src.vector_store import get_vector_store
from src.retriever import get_retriver
from src.generator import get_llm
from src.rag_pipeline import build_rag_pipeline


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Query the RAG system")
    parser.add_argument("query", nargs="*", help="The question to ask")
    parser.add_argument("--model", "-m", default=None, help="The model to use (default from config)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    args = parser.parse_args()

    # Initialize the RAG pipeline
    print("Initializing the RAG system...")

    # Get embeddings
    embeddings = get_embeddings()

    # Get vector store
    vector_store = get_vector_store(embeddings)

    # Get retriever
    retriever = get_retriver(vector_store)

    # Get LLM
    llm = get_llm(model_name=args.model)

    # Build RAG pipeline
    rag = build_rag_pipeline(retriever, llm)

    # Run in interactive mode or process a single query
    if args.interactive:
        run_interactive_mode(rag)
    else:
        # Join all arguments as the query
        query = " ".join(args.query)
        if not query:
            parser.print_help()
            sys.exit(1)

        # Process the query
        process_query(rag, query)


def run_interactive_mode(rag):
    """Run the RAG system in interactive mode."""
    print("\n📚 RAG Query System - Interactive Mode")
    print("Type 'exit', 'quit', or press Ctrl+C to exit")

    while True:
        try:
            # Get user input
            user_input = input("\n🔎 Query: ")

            # Check for exit command
            if user_input.lower() in ["exit", "quit"]:
                break

            # Skip empty input
            if not user_input.strip():
                continue

            # Process the query
            process_query(rag, user_input)

        except KeyboardInterrupt:
            print("\nExiting...")
            break


def process_query(rag, query):
    """Process a single query using the RAG pipeline."""
    print(f"\nProcessing query: {query}")

    try:
        # Get response from RAG pipeline
        result = rag(query)

        # Print the answer
        print("\n" + "="*50)
        print("ANSWER:")
        print(result["answer"])

        # Print the sources
        print("\nSOURCES:")
        for i, doc in enumerate(result["documents"], 1):
            print(f"{i}. {doc.metadata.get('source_file', 'unknown')} - Page {doc.metadata.get('page', 'unknown')}")

    except Exception as e:
        print(f"\nError processing query: {str(e)}")


if __name__ == "__main__":
    main()