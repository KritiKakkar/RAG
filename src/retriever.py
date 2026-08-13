from src.config import TOP_K

def get_retriver(vector_store):
    return vector_store.as_retriever(
        search_type = "similarity" ,
        search_kwargs = {"k": TOP_K}
        )