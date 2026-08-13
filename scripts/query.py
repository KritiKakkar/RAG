import sys
import os
# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.embeddings import get_embeddings
from src.vector_store import get_vector_store
from src.retriever import get_retriver
from src.generator import get_llm
from src.rag_pipeline import build_rag_pipeline


def main():

    embeddings = get_embeddings()

    vector_store = get_vector_store(embeddings)

    retriever = get_retriver(vector_store)

    llm = get_llm()

    rag = build_rag_pipeline(retriever, llm)

    question = (
        "What is my name?"
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