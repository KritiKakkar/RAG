from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_SIZE, CHUNK_OVERLAP

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )

    return splitter.split_documents(documents)


def add_chunk_metadata(chunks):
    for index, chunck in enumerate(chunks):
        chunck.metadata["chunk_index"] = index
        chunck.metadata["chunk_size"] = len(chunck.page_content)

        return chunks