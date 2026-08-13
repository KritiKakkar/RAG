from langchain_core.prompts import (
    ChatPromptTemplate,
)


RAG_PROMPT = ChatPromptTemplate.from_template(
    """
You are a technical assistant.

Answer the question using only the
provided context.

If the answer cannot be determined
from the context, say:

"I don't have enough information in
the provided documents."

Do not invent technical details.

Context:
{context}

Question:
{question}

Answer:
"""
)