import pandas as pd
from datasets import load_dataset
from langchain_community.document_loaders.text import TextLoader
import random
from langchain_community.document_loaders import PDFMinerLoader


def load_txt(txt_name):
    txt = "data_file/" + txt_name
    loader = TextLoader(txt)
    docs = loader.load()
    return docs


def load_pdf(pdf_name):
    pdf = "data_file/" + pdf_name
    loader = PDFMinerLoader(pdf)
    docs = loader.load()
    return docs


def load_data(data_name, data_size):
    dataset = load_dataset(data_name, 'distractor')
    train_size = len(dataset['train'])
    random_indices = random.sample(range(train_size), data_size)
    print(random_indices)
    random_data = dataset['train'].select(random_indices)
    return random_data


def save_context(all_context, name):
    file_name = './data_file/' + name
    with open(file_name, 'w', encoding='utf-8') as file:
        for c in all_context:
            file.write(f"{c}\n")


def generate_test_excel(size, data_name):
    data = load_data(size, data_name)

    save_context(set(data["context"]))

    results = []

    for i in range(size):
        question = data['question'][i]
        print(f"Procesando pregunta {i}: {question}")
        ground_truth = data['answers'][i]['text']
        context = data['context'][i]

        results.append({
            "question": question,
            "contexts": context,
            "ground_truth": ground_truth
        })

    df_results = pd.DataFrame(results)
    df_results.to_excel("test_data.xlsx", index=False)


def generate_test_excel_hotpot(size, data_name):
    data = load_data(size, data_name)

    context = []

    for i in data['context']:
        for j in range(len(i["sentences"])):
            context.append(i["sentences"][j])

    save_context(context, "hotpot_context.txt")

    results = []

    for i in range(size):
        question = data['question'][i]
        print(f"Procesando pregunta {i}: {question}")
        ground_truth = data['answer'][i]
        context = data['context'][i]["sentences"]

        results.append({
            "question": question,
            "contexts": context,
            "ground_truth": ground_truth
        })

    df_results = pd.DataFrame(results)
    df_results.to_excel("test_data_hotpot.xlsx", index=False)
