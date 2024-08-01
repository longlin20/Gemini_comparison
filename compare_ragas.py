import logging
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
    """
    Configures the LLM and embeddings for each metric.

    Args:
        metrics (list): List of metrics to configure.
        ragas_vertexai_llm (LangchainLLMWrapper): The LLM to use.
    """
    for m in metrics:
        m.llm = ragas_vertexai_llm  # Set the LLM for the metric

        # Check if this metric needs embeddings and configure accordingly
        if hasattr(m, "embeddings"):
            m.embeddings = LangchainEmbeddingsWrapper(get_vertexai_embeddings())


def read_excel_to_df(file_path, rag):
    """
    Reads an Excel file into a DataFrame.

    Args:
        file_path (str): Path to the Excel file.
        rag (bool): Whether to read the 'contexts' column as a list.

    Returns:
        pd.DataFrame: The DataFrame containing the Excel data.
    """
    df = pd.read_excel(file_path)
    df['answer'] = df['answer'].apply(lambda x: str(x))
    if rag:
        df["contexts"] = df["contexts"].apply(lambda x: [x] if isinstance(x, str) else x)
    return df


def run_evaluation_sequential(dataset, metrics):
    """
    Runs the evaluation of a dataset in batches.

    Args:
        dataset (Dataset): The dataset to evaluate.
        metrics (list): List of metrics to use.

    Returns:
        pd.DataFrame: The results of the evaluation.
    """
    all_results = []
    try:
        for i in range(0, len(dataset), 30):
            batch = dataset.select(range(i, min(i + 30, len(dataset))))
            try:
                result = evaluate(
                    batch,
                    metrics=metrics,
                    raise_exceptions=False
                )
                all_results.append(result.to_pandas())
                logging.info(f"Batch {i // 30 + 1} processed successfully.")
            except Exception as e:
                logging.error(f"An error occurred during evaluation of batch {i // 30 + 1}: {e}")
            time.sleep(20)
    except Exception as e:
        logging.error(f"An error occurred during the evaluation process: {e}")
    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()


def save_results_to_excel(results_df, file_path):
    """
    Saves the results of an evaluation to an Excel file.

    Args:
        results_df (pd.DataFrame): The results to save.
        file_path (str): The path to the Excel file.
    """
    results_df.to_excel(file_path, index=False, engine='openpyxl')
    logging.info("Results successfully saved to {}".format(file_path))

def evaluate_ragas_metrics(model, input_file_name, output_file_name, rag=True):
    """
    Evaluates the RAGAS metrics for a given model and input file.

    Args:
        model (str): The model to use ('gemini1.0' or 'gemini1.5').
        input_file_name (str): The path to the input Excel file.
        output_file_name (str): The path to the output Excel file.
        rag (bool): Whether to use RAG-based processing (default is True).
    """
    # Define the metrics to use in evaluation
    if rag:
        metrics = [
            faithfulness,
            answer_similarity,
            answer_correctness
        ]
    else:
        metrics = [
            answer_similarity,
            answer_correctness
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
    dataset = Dataset.from_pandas(df)
    print(dataset)

    # Run evaluation sequentially
    results_df = run_evaluation_sequential(dataset, metrics)

    if not results_df.empty:
        logging.info(results_df.head())
        save_results_to_excel(results_df, output_file_name)
    else:
        logging.info("Evaluation failed.")