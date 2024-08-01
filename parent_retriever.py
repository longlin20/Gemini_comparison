from langchain.storage._lc_store import create_kv_docstore
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore
from langchain_community.vectorstores.chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from embedding_documents import get_vertexai_embeddings


def get_parents_retriever():
    """
    This function initializes and returns a ParentDocumentRetriever object.

    The retriever is initialized with a Chroma vectorstore, a LocalFileStore docstore,
    and RecursiveCharacterTextSplitter objects for parent and child splitting.

    Returns:
        ParentDocumentRetriever: The initialized retriever object.
    """

    # Initialize embedding function
    embedding = get_vertexai_embeddings()

    # Initialize LocalFileStore docstore
    fs = LocalFileStore("./store_location")
    store = create_kv_docstore(fs)

    # Initialize Chroma vectorstore
    vectorstore = Chroma(
        collection_name="split_parent",
        embedding_function=embedding,
        persist_directory="./db"
    )

    # Initialize RecursiveCharacterTextSplitter for parent and child splitting
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1200)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)

    # Initialize ParentDocumentRetriever
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    return retriever

# docs = load_txt("set_file.txt")
# retriever.add_documents(docs, ids=None)
