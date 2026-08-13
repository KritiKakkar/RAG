from pathlib import Path

from langchain_community.document_loaders import (PyPDFLoader, TextLoader)


def load_document(data_dir: str):
    documents = []

    for path in Path(data_dir).rglob("*"):

        suffix = path.suffix.lower()

        if suffix == ".pdf":
            loader = PyPDFLoader(str(path))
            docs = loader.load()

        elif suffix in {".md", ".txt"}:
            loader = TextLoader(str(path), encoding="utf-8")
            docs = loader.load()

        else:
            print(f"Unsupported file type: {suffix}")
            continue

        for doc in docs:
            doc.metadata["source_file"] = path.name
            doc.metadata["file_type"] = suffix

        documents.extend(docs)

    return documents
