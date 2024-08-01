import json
import re
import time
from random import random
from random import seed
import google.auth
import pandas as pd
from datasets import load_dataset
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_community.vectorstores.chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from sentence_transformers import CrossEncoder

from choose_llm import gemini_llm, gemini15_llm
from data_processing import load_txt, load_pdf
from embedding_documents import get_vertexai_embeddings, embedding_txt_documents, \
    embedding_pdf_documents

seed(41)


def add_txt_data_to_vector_db(file_path, db_name, chunk_size, chunk_overlap):
    embedding_txt_documents(file_path, db_name, chunk_size, chunk_overlap)


def add_pdf_data_to_vector_db(file_path, db_name, chunk_size, chunk_overlap):
    embedding_pdf_documents(file_path, db_name, chunk_size, chunk_overlap)


# add_txt_data_to_vector_db("context.txt", "context_256_32", 256, 32)
# add_pdf_data_to_vector_db("Python Programming.pdf", "context_python3_256_32", 256, 32)

def get_retriever(db, k):
    return db.as_retriever(search_kwargs={"k": k})


def embedding_docs(db_name):
    # "./chroma_db/vertexai/ + dataset_name"
    persist_directory = "./chroma_db/" + db_name
    embedding = get_vertexai_embeddings()
    return Chroma(persist_directory=persist_directory, embedding_function=embedding)


def get_bm25_retriever(data_name, chunk_size, chunk_overlap, number_of_chunks):
    docs = load_txt(data_name)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
    )

    chunks = text_splitter.split_documents(docs)
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = number_of_chunks
    return bm25_retriever


def get_bm25_retriever_code(data_name, chunk_size, chunk_overlap, number_of_chunks):
    docs = load_pdf(data_name)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
    )

    chunks = text_splitter.split_documents(docs)
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = number_of_chunks
    return bm25_retriever


def get_parents_retriever(data_name):
    parents_retriever = get_parents_retriever()
    docs = load_txt(data_name)
    parents_retriever.add_documents(docs, ids=None)
    return parents_retriever


def get_dense_retriever(database_name, number_of_chunks):
    dense_retriever = get_retriever(embedding_docs(database_name), number_of_chunks)
    return dense_retriever


def select_model(model):
    credentials, project_id = google.auth.default()
    LOCATION = "us-central1"

    if model == "gemini1.0":
        return gemini_llm(project_id, LOCATION)
    elif model == "gemini1.5":
        return gemini15_llm(project_id, LOCATION)
    else:
        raise ValueError("Invalid model. Please choose 'gemini1.0' or 'gemini1.5'.")


def select_retriever(retriever_type, data_name, chunk_size, chunk_overlap, number_of_chunks, database_name):
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


def generate_llm_answer_and_similarity(qa, llm, compare_llm, cross_encoder_model, question, ground_truth,
                                       compare_prompt, col_name, df, i, rag=True):
    gemini_sim_col_name = "similarity " + col_name
    cross_sim_col_name = "similarity CrossEncoder " + col_name
    context_name = "contexts"
    best_compare_prompt = compare_prompt
    best_compare_chain = best_compare_prompt | compare_llm
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
    """ Remove characters that cannot be used in Excel worksheets. """
    ILLEGAL_CHARACTERS_RE = re.compile(
        r'[\000-\010]|[\013-\014]|[\016-\037]'
    )
    return ILLEGAL_CHARACTERS_RE.sub("", value)


def generate_llm_answer_and_similarity_with_rag(model, compare_model, retriever_type="", data_name="", chunk_size=0,
                                                chunk_overlap=0,
                                                number_of_chunks=0, database_name="", prompt_type=None,
                                                compare_prompt=None, input_excel_file="",
                                                col_name="", result_excel_file="", rag=True):
    llm = select_model(model)
    compare_llm = select_model(compare_model)
    cross_encoder_model = CrossEncoder('cross-encoder/stsb-roberta-large')

    if rag:
        retriever = select_retriever(retriever_type, data_name, chunk_size, chunk_overlap, number_of_chunks,
                                     database_name)
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            return_source_documents=True,
            chain_type_kwargs={
                "verbose": False,
                "prompt": prompt_type,
            })
    else:
        qa = None

    df = pd.read_excel(input_excel_file)

    question_count = 0

    for i, row in df.iterrows():
        question = row['question']
        ground_truth = row['ground_truth']

        generate_llm_answer_and_similarity(qa, llm, compare_llm, cross_encoder_model, question, ground_truth,
                                           compare_prompt, col_name, df, i, rag)

        question_count += 1
        if model == "gemini1.5" and question_count % 50 == 0:
            time.sleep(10)
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].apply(lambda x: sanitize_string(x) if isinstance(x, str) else x)
    df.to_excel(result_excel_file, index=False)


"""
generate_llm_answer_and_similarity_with_rag("gemini1.0", "gemini1.5", compare_prompt=best_compare_template2(), input_excel_file="data_hotpot.xlsx",
                                           col_name="without RAG", result_excel_file="result_hotpot_without_rag.xlsx", rag=False)

generate_llm_answer_and_similarity_with_rag("gemini1.0", "gemini1.5", "ensemble", "hotpot_context.txt", 256, 32, 4, "context_hotpot_256_32_1", knowledge_full_context_template(),
                                            compare_prompt=best_compare_template2(), input_excel_file="data_hotpot.xlsx",
                                            col_name="answer", result_excel_file="results/hotpot/result_hotpot_best_rag_gemini1.0.xlsx", rag=True)

generate_llm_answer_and_similarity_with_rag("gemini1.5", "gemini1.0", "ensemble", "hotpot_context.txt", 256, 32, 4, "context_hotpot_256_32_1", knowledge_brief_answer_template(),
                                            compare_prompt=best_compare_template2(), input_excel_file="data_hotpot.xlsx",
                                            col_name="answer", result_excel_file="results/hotpot/result_hotpot_best_rag_gemini1.5brief1.0.xlsx", rag=True)

generate_llm_answer_and_similarity_with_rag("gemini1.5", "gemini1.5", "ensemble", "hotpot_context.txt", 256, 32, 4, "context_hotpot_256_32_1", knowledge_brief_answer_template(),
                                            compare_prompt=best_compare_template2(), input_excel_file="data_hotpot.xlsx",
                                            col_name="answer", result_excel_file="results/hotpot/result_hotpot_best_rag_gemini1.5_brief1.5.xlsx", rag=True)

"""

# NEED TO CHANGE THE DATABASE FOR CONTEXT(database_name) AND DAT FOR CONTEXT(data_name)

# generate_llm_answer_and_similarity_with_rag("gemini1.0", "gemini1.5", compare_prompt=best_compare_template2(), input_excel_file="data_openai_humaneval.xlsx",
#                                            col_name="answer", result_excel_file="results/humaneval/result_openai_humaneval_without_rag.xlsx", rag=False)

"""
generate_llm_answer_and_similarity_with_rag("gemini1.0", "gemini1.5", "ensemble_code", "Python Programming.pdf", 256, 32,
                                            6, "context_python3_256_32", python_code_generation_template(),
                                            compare_prompt=best_compare_template2(),
                                            input_excel_file="data_openai_humaneval.xlsx",
                                            col_name="answer",
                                            result_excel_file="results/squad/result_openai_humaneval_best_rag_gemini1.0.xlsx",
                                            rag=True)

generate_llm_answer_and_similarity_with_rag("gemini1.5", "gemini1.5", "ensemble_code", "Python Programming.pdf", 256, 32,
                                            6, "context_python3_256_32", python_code_generation_template(),
                                            compare_prompt=best_compare_template2(),
                                            input_excel_file="data_openai_humaneval.xlsx",
                                            col_name="answer",
                                            result_excel_file="results/squad/result_openai_humaneval_best_rag_gemini1.5.xlsx",
                                            rag=True)
"""
# generate_llm_answer_and_similarity_with_rag("gemini1.5", "gemini1.5", compare_prompt=best_compare_template2(), input_excel_file="data_openai_humaneval.xlsx",
#                                           col_name="answer", result_excel_file="result_openai_humaneval_without_rag_1.5.xlsx", rag=False)
