import re
import time
from random import seed
import google.auth
import pandas as pd

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_community.vectorstores.chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from sentence_transformers import CrossEncoder

from choose_llm import gemini_llm, gemini15_llm
from data_processing import load_document
from embedding_documents import get_vertexai_embeddings, embed_documents_from_text, embed_documents_from_pdf

seed(41)


def add_data_to_vector_db(file_path: str, db_name: str, chunk_size: int, chunk_overlap: int, file_type: str) -> None:
    """
    Adds data from a file to a vector database.

    Args:
        file_path (str): Path to the file.
        db_name (str): Name of the vector database.
        chunk_size (int): Size of the chunks.
        chunk_overlap (int): Overlap between chunks.
        file_type (str): Type of the file.
    """
    if file_type == "txt":
        embed_documents_from_text(file_path, db_name, chunk_size, chunk_overlap)
    elif file_type == "pdf":
        embed_documents_from_pdf(file_path, db_name, chunk_size, chunk_overlap)


def get_retriever(db, k: int):
    """
    Returns a retriever for a given database.

    Args:
        db (Database): The database.
        k (int): Number of documents to retrieve.

    Returns:
        Retriever: The retriever.
    """
    return db.as_retriever(search_kwargs={"k": k})


def get_chroma_db(db_name: str) -> Chroma:
    """
    Returns a Chroma database.

    Args:
        db_name (str): Name of the database.

    Returns:
        Chroma: The Chroma database.
    """
    persist_directory = "./chroma_db/" + db_name
    embedding = get_vertexai_embeddings()
    return Chroma(persist_directory=persist_directory, embedding_function=embedding)


def get_retriever_from_db(retriever_type: str, data_name: str, chunk_size: int, chunk_overlap: int, number_of_chunks: int, database_name: str):
    """
    Returns a retriever based on the type of retriever and the database.

    Args:
        retriever_type (str): Type of the retriever.
        data_name (str): Name of the data.
        chunk_size (int): Size of the chunks.
        chunk_overlap (int): Overlap between chunks.
        number_of_chunks (int): Number of chunks.
        database_name (str): Name of the database.

    Returns:
        Union[BM25Retriever, EnsembleRetriever, ParentsRetriever]: The retriever.

    Raises:
        ValueError: If the retriever type is invalid.
    """
    if retriever_type == "bm25":
        return get_bm25_retriever(data_name, chunk_size, chunk_overlap, number_of_chunks)
    elif retriever_type == "dense":
        return get_dense_retriever(database_name, number_of_chunks)
    elif retriever_type == "parents":
        return get_parents_retriever(data_name)
    elif retriever_type == "ensemble":
        bm25_retriever = get_bm25_retriever(data_name, chunk_size, chunk_overlap, number_of_chunks)
        dense_retriever = get_dense_retriever(database_name, number_of_chunks)
        return EnsembleRetriever(retrievers=[bm25_retriever, dense_retriever], weights=[0.5, 0.5])
    elif retriever_type == "ensemble_code":
        bm25_retriever_code = get_bm25_retriever_code(data_name, chunk_size, chunk_overlap, number_of_chunks)
        dense_retriever = get_dense_retriever(database_name, number_of_chunks)
        return EnsembleRetriever(retrievers=[bm25_retriever_code, dense_retriever], weights=[0.5, 0.5])
    else:
        raise ValueError(
            "Invalid retriever type. Please choose 'bm25', 'dense', 'parents', 'ensemble' or 'ensemble_code'.")


def get_llm(model: str):
    """
    Returns an LLM (Language Learning Model) based on the model name.

    Args:
        model (str): Name of the model.

    Returns:
        Any: The LLM.

    Raises:
        ValueError: If the model name is invalid.
    """
    credentials, project_id = google.auth.default()
    LOCATION = "us-central1"

    if model == "gemini1.0":
        return gemini_llm(project_id, LOCATION)
    elif model == "gemini1.5":
        return gemini15_llm(project_id, LOCATION)
    else:
        raise ValueError("Invalid model. Please choose 'gemini1.0' or 'gemini1.5'.")


def get_bm25_retriever(data_name: str, chunk_size: int, chunk_overlap: int, number_of_chunks: int) -> BM25Retriever:
    """
    Returns a BM25 retriever for the given data.

    Args:
        data_name (str): Name of the data.
        chunk_size (int): Size of the chunks.
        chunk_overlap (int): Overlap between chunks.
        number_of_chunks (int): Number of chunks.

    Returns:
        BM25Retriever: The BM25 retriever.
    """
    docs = load_document(data_name, "txt")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
    )
    chunks = text_splitter.split_documents(docs)
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = number_of_chunks
    return bm25_retriever


def get_bm25_retriever_code(data_name: str, chunk_size: int, chunk_overlap: int, number_of_chunks: int) -> BM25Retriever:
    """
    Returns a BM25 retriever for the given code data.

    Args:
        data_name (str): Name of the code data.
        chunk_size (int): Size of the chunks.
        chunk_overlap (int): Overlap between chunks.
        number_of_chunks (int): Number of chunks.

    Returns:
        BM25Retriever: The BM25 retriever.
    """
    docs = load_document(data_name, "pdf")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
    )
    chunks = text_splitter.split_documents(docs)
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = number_of_chunks
    return bm25_retriever


def get_parents_retriever(data_name: str):
    """
    Returns a parents retriever for the given data.

    Args:
        data_name (str): Name of the data.

    Returns:
        ParentsRetriever: The parents retriever.
    """
    parents_retriever = get_parents_retriever()
    docs = load_document(data_name, "txt")
    parents_retriever.add_documents(docs, ids=None)
    return parents_retriever


def get_dense_retriever(database_name: str, number_of_chunks: int):
    """
    Returns a dense retriever for the given database.

    Args:
        database_name (str): Name of the database.
        number_of_chunks (int): Number of chunks.

    Returns:
        Retriever: The dense retriever.
    """
    dense_retriever = get_retriever(get_chroma_db(database_name), number_of_chunks)
    return dense_retriever

def generate_llm_answer_and_similarity(qa, llm, compare_llm, cross_encoder_model, question: str, ground_truth: str,
                                       compare_prompt, col_name: str, df, i: int, rag: bool = True) -> None:
    """
    Generates an LLM answer and calculates similarity scores.

    Args:
        qa: The QA model.
        llm: The LLM model.
        compare_llm: The LLM used for comparison.
        cross_encoder_model: The cross-encoder model.
        question (str): The question.
        ground_truth (str): The ground truth answer
        compare_prompt: The prompt used for comparison.
        col_name (str): The column name to store the LLM answer.
        df: The DataFrame to store results.
        i (int): The index of the current question.
        rag (bool, optional): Whether to use retrieval-augmented generation (RAG). Defaults to True.
    """
    gemini_sim_col_name = "similarity " + col_name
    cross_sim_col_name = "similarity CrossEncoder " + col_name
    context_name = "contexts"
    best_compare_chain = compare_prompt | compare_llm

    try:
        print(f"Attempting to get llm_answer for question {i}")
        if rag:
            result = qa.invoke(question)
            llm_answer = result["result"]
            answer_context = result['source_documents']
            context = "\n".join(doc.page_content for doc in answer_context)
            df.loc[i, context_name] = context
        else:
            llm_answer = llm.invoke(question)
        print(f"Successfully obtained llm_answer for question {i}: {llm_answer}")
    except Exception as e:
        print(f"Error obtaining llm_answer for question {i}: {e}")
        llm_answer = "Error processing the question"
        df.loc[i, context_name] = "Error processing the question"

    try:
        llm_answer_similarity = best_compare_chain.invoke(
            {"context": question, "phrase1": ground_truth, "phrase2": llm_answer})
        print(f"Successfully calculated llm_answer_similarity for question {i}: {llm_answer_similarity}")
    except Exception as e:
        print(f"Error calculating llm_answer_similarity for question {i}: {e}")
        llm_answer_similarity = 0

    try:
        cross_sim_similarity = cross_encoder_model.predict([ground_truth, llm_answer])
    except Exception as e:
        print(f"Error calculating cross_sim_similarity for question {i}: {e}")
        cross_sim_similarity = 0

    df.loc[i, col_name] = llm_answer
    df.loc[i, gemini_sim_col_name] = llm_answer_similarity
    df.loc[i, cross_sim_col_name] = cross_sim_similarity

def sanitize_string(value: str) -> str:
    ILLEGAL_CHARACTERS_RE = re.compile(
        r'[\000-\010]|[\013-\014]|[\016-\037]'
    )
    return ILLEGAL_CHARACTERS_RE.sub("", value)

def generate_llm_answer_and_similarity_with_rag(
    model: str,
    compare_model: str,
    retriever_type: str = "",
    data_name: str = "",
    chunk_size: int = 0,
    chunk_overlap: int = 0,
    number_of_chunks: int = 0,
    database_name: str = "",
    prompt_type = None,
    compare_prompt = None,
    input_excel_file: str = "",
    col_name: str = "",
    result_excel_file: str = "",
    rag: bool = True
):
    """
    Generate LLM answers and calculate similarity scores.

    Args:
        model (str): The LLM model.
        compare_model (str): The LLM model for comparison.
        retriever_type (str): The type of retriever. Defaults to "".
        data_name (str): The name of the data. Defaults to "".
        chunk_size (int): The size of chunks. Defaults to 0.
        chunk_overlap (int): The overlap between chunks. Defaults to 0.
        number_of_chunks (int): The number of chunks. Defaults to 0.
        database_name (str): The name of the database. Defaults to "".
        prompt_type (Prompt): The type of prompt. Defaults to None.
        compare_prompt (Prompt): The prompt for comparison. Defaults to None.
        input_excel_file (str): The path to the input Excel file. Defaults to "".
        col_name (str): The column name to store the LLM answer. Defaults to "".
        result_excel_file (str): The path to the result Excel file. Defaults to "".
        rag (bool, optional): Whether to use retrieval-augmented generation (RAG). Defaults to True.

    Returns:
        None
    """
    # Get the LLM models
    llm = get_llm(model)
    compare_llm = get_llm(compare_model)

    # Get the cross-encoder model
    cross_encoder_model = CrossEncoder('cross-encoder/stsb-roberta-large')

    # Get the retriever based on the retriever type
    if rag:
        retriever = get_retriever_from_db(
            retriever_type,
            data_name,
            chunk_size,
            chunk_overlap,
            number_of_chunks,
            database_name
        )

        # Create the QA model
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            return_source_documents=True,
            chain_type_kwargs={
                "verbose": False,
                "prompt": prompt_type,
            }
        )
    else:
        qa = None

    # Read the input Excel file
    df = pd.read_excel(input_excel_file)

    # Iterate over the rows of the DataFrame
    question_count = 0
    for i, row in df.iterrows():
        question = row['question']
        ground_truth = row['ground_truth']

        # Generate the LLM answer and calculate similarity
        generate_llm_answer_and_similarity(
            qa,
            llm,
            compare_llm,
            cross_encoder_model,
            question,
            ground_truth,
            compare_prompt,
            col_name,
            df,
            i,
            rag
        )

        question_count += 1

        # Delay if the model is "gemini1.5" and the question count is a multiple of 50
        if model == "gemini1.5" and question_count % 50 == 0:
            time.sleep(10)

    # Sanitize the object columns in the DataFrame
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].apply(lambda x: sanitize_string(x) if isinstance(x, str) else x)

    # Save the modified DataFrame to the result Excel file
    df.to_excel(result_excel_file, index=False)