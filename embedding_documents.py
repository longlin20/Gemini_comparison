import time
from langchain_google_vertexai import VertexAIEmbeddings
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from data_processing import load_document


# Utility functions for Embeddings API with rate limiting
def rate_limit(max_per_minute):
    period = 60 / max_per_minute
    print("Waiting")
    while True:
        before = time.time()
        yield
        after = time.time()
        elapsed = after - before
        sleep_time = max(0, period - elapsed)
        if sleep_time > 0:
            print(".", end="")
            time.sleep(sleep_time)


class CustomVertexAIEmbeddings(VertexAIEmbeddings):
    requests_per_minute: int
    num_instances_per_batch: int

    # Overriding embed_documents method
    def embed_documents(self, texts: List[str]):
        limiter = rate_limit(self.requests_per_minute)
        results = []
        docs = list(texts)

        while docs:
            # Working in batches because the API accepts maximum 5
            # documents per request to get embeddings
            head, docs = (
                docs[: self.num_instances_per_batch],
                docs[self.num_instances_per_batch:],
            )
            chunk = self.client.get_embeddings(head)
            results.extend(chunk)
            next(limiter)

        return [r.values for r in results]


def get_vertexai_embeddings():
    EMBEDDING_QPM = 100
    EMBEDDING_NUM_BATCH = 5
    vertexai_embeddings = CustomVertexAIEmbeddings(
        requests_per_minute=EMBEDDING_QPM,
        num_instances_per_batch=EMBEDDING_NUM_BATCH,
        model_name="textembedding-gecko@001"
    )
    return vertexai_embeddings


def embed_documents_from_text(doc_name: str, directory: str, chunk_size: int, chunk_overlap: int) -> None:
    """
    Embeds documents from a text file.
    """
    docs = load_document(doc_name, "txt")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""])
    chunks = text_splitter.split_documents(docs)
    ids = [f"{chunk.metadata['source']}-chunk{idx}" for idx, chunk in enumerate(chunks)]
    embedding = get_vertexai_embeddings(100, 5, "textembedding-gecko@001")
    persist_directory = f"./chroma_db/{directory}"
    Chroma.from_documents(chunks, embedding, ids=ids, persist_directory=persist_directory)


def embed_documents_from_pdf(doc_name: str, directory: str, chunk_size: int, chunk_overlap: int) -> None:
    """
    Embeds documents from a PDF file.
    """
    docs = load_document(doc_name, "pdf")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""])
    chunks = text_splitter.split_documents(docs)
    ids = [f"{chunk.metadata['source']}-chunk{idx}" for idx, chunk in enumerate(chunks)]
    embedding = get_vertexai_embeddings(100, 5, "textembedding-gecko@001")
    persist_directory = f"./chroma_db/{directory}"
    Chroma.from_documents(chunks, embedding, ids=ids, persist_directory=persist_directory)