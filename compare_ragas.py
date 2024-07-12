import logging
import os

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper

from choose_llm import gemini15_llm, gemini_llm
from embedding_documents import get_vertexai_embeddings
import google.auth
import time
from ragas.metrics import (
    answer_similarity,
    answer_correctness,
    faithfulness
)

def configure_metrics(metrics, ragas_vertexai_llm):
    for m in metrics:
        m.llm = ragas_vertexai_llm  # Set the LLM for the metric

        # Check if this metric needs embeddings and configure accordingly
        if hasattr(m, "embeddings"):
            m.embeddings = LangchainEmbeddingsWrapper(get_vertexai_embeddings())

def read_excel_to_df(file_path, rag):
    df = pd.read_excel(file_path)
    df['answer'] = df['answer'].apply(lambda x: str(x))
    if rag:
        df["contexts"] = df["contexts"].apply(lambda x: [x] if isinstance(x, str) else x)
    return df

def convert_df_to_dataset(df):
    return Dataset.from_pandas(df)

def run_evaluation_sequential(dataset, metrics):
    all_results = []
    try:
        for i in range(0, len(dataset), 20):
            batch = dataset.select(range(i, min(i + 20, len(dataset))))
            try:
                result = evaluate(
                    batch,
                    metrics=metrics,
                    raise_exceptions=False
                )
                all_results.append(result.to_pandas())
                logging.info(f"Batch {i // 20 + 1} processed successfully.")
            except Exception as e:
                logging.error(f"An error occurred during evaluation of batch {i // 20 + 1}: {e}")
            time.sleep(50)
    except Exception as e:
        logging.error(f"An error occurred during the evaluation process: {e}")
    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

def save_results_to_excel(results_df, file_path):
    results_df.to_excel(file_path, index=False, engine='openpyxl')
    logging.info("Results successfully saved to {}".format(file_path))

def evaluate_ragas_metrics(model, input_file_name, output_file_name, rag = True):
    # Define the metrics to use in evaluation
    metrics = [
        faithfulness
        #answer_similarity,
        #answer_correctness
    ]

    # Authenticate and set up the LLM and embeddings
    credentials, project_id = google.auth.default()
    LOCATION = "us-central1"
    if model == "gemini1.0":
        llm = gemini_llm(project_id, LOCATION)
    elif model == "gemini1.5":
        llm = gemini15_llm(project_id, LOCATION)
    else:
        raise ValueError("Invalid model. Please choose 'gemini1.0' or 'gemini1.5'.")
    ragas_vertexai_llm = LangchainLLMWrapper(llm)

    # Configure each metric with the appropriate LLM and embeddings
    configure_metrics(metrics, ragas_vertexai_llm)

    # Read the Excel file into a DataFrame
    df = read_excel_to_df(input_file_name, rag)

    # Convert the DataFrame to a Hugging Face Dataset with defined features
    dataset = convert_df_to_dataset(df)
    print(dataset)

    # Run evaluation sequentially
    results_df = run_evaluation_sequential(dataset, metrics)

    if not results_df.empty:
        logging.info(results_df.head())

        # Save the results DataFrame to an Excel file
        save_results_to_excel(results_df, output_file_name)
    else:
        logging.info("Evaluation failed.")

model = "gemini1.5"

all_items = os.listdir("results_test2")
files = [item for item in all_items if os.path.isfile(os.path.join("results_test2", item))]

for f in files:
    input_file_name = output_file_name = os.path.join("results_test2", f)
#input_file_name = output_file_name = "results/humaneval/result_openai_humaneval_without_rag.xlsx"
    evaluate_ragas_metrics(model, input_file_name, output_file_name)
