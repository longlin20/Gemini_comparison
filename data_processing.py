import pandas as pd
from datasets import load_dataset
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders import PDFMinerLoader
import random


def load_document(document_name, document_type):
    """
    Load a document based on its type.

    Args:
        document_name (str): The name of the document.
        document_type (str): The type of the document.

    Returns:
        list: The loaded document.
    """
    loader = TextLoader if document_type == "txt" else PDFMinerLoader
    document_path = f"data_file/{document_name}"
    loader = loader(document_path)
    return loader.load()


def load_data(data_name, data_size):
    """
    Load a dataset and select a random subset.

    Args:
        data_name (str): The name of the dataset.
        data_size (int): The size of the subset.

    Returns:
        list: The selected subset of the dataset.
    """
    dataset = load_dataset(data_name, 'distractor')
    train_data = dataset['train']
    train_size = len(train_data)
    random_indices = random.sample(range(train_size), data_size)
    selected_data = train_data.select(random_indices)
    return selected_data


def save_context(all_context, file_name):
    """
    Save a set of contexts to a file.

    Args:
        all_context (set): The set of contexts.
        file_name (str): The name of the file.
    """
    with open(f"./data_file/{file_name}", 'w', encoding='utf-8') as file:
        for context in all_context:
            file.write(f"{context}\n")


def generate_test_data(size, data_name, context_file_name):
    """
    Generate test data and save the context to a file.

    Args:
        size (int): The size of the test data.
        data_name (str): The name of the dataset.
        context_file_name (str): The name of the file to save the context.
    """
    data = load_data(data_name, size)
    save_context(set(data["context"]), context_file_name)

    results = []
    for i in range(size):
        question = data['question'][i]
        ground_truth = data['answers'][i]['text']
        context = data['context'][i]

        results.append({
            "question": question,
            "contexts": context,
            "ground_truth": ground_truth
        })

    df_results = pd.DataFrame(results)
    df_results.to_excel(f"test_data.xlsx", index=False)


def generate_test_data_hotpot(size, data_name, context_file_name):
    """
    Generate test data for HotPot and save the context to a file.

    Args:
        size (int): The size of the test data.
        data_name (str): The name of the dataset.
        context_file_name (str): The name of the file to save the context.
    """
    data = load_data(data_name, size)
    save_context(set(data["context"]), context_file_name)

    results = []
    for i in range(size):
        question = data['question'][i]
        ground_truth = data['answer'][i]
        context = data['context'][i]["sentences"]

        results.append({
            "question": question,
            "contexts": context,
            "ground_truth": ground_truth
        })

    df_results = pd.DataFrame(results)
    df_results.to_excel(f"test_data_hotpot.xlsx", index=False)