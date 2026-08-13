from src.prompts import RAG_PROMPT


def build_rag_pipeline(retriever, llm):

    def ask(question):

        docs = retriever.invoke(question)

        context = format_docs(docs)

        prompt = RAG_PROMPT.invoke({"context": context, "question": question})

        response = llm.invoke(prompt)

        return {
            "answer": response.content,
            "documents": docs
        }

    return ask

# Format documents with source information
def format_docs(documents):
    """Format documents for inclusion in the prompt."""
    formatted = []

    for doc in documents:
        source = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page", "unknown")

        formatted.append(
            f"""
                SOURCE: {source}
                PAGE: {page}

                CONTENT:
                {doc.page_content}
                """
        )

    return "\n\n".join(formatted)