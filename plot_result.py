import matplotlib.pyplot as plt
import pandas as pd

def plot_performance(averages, low_scores, mid_scores, high_scores):
    chunk_sizes = range(3, 7)
    labels = [f"{chunk} chunks" for chunk in chunk_sizes]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(labels, averages, color='b', alpha=0.5, label='Average of 500 examples')
    ax.set_xlabel('Number of Chunks')
    ax.set_ylabel('Average Scores', color='b')
    ax.tick_params(axis='y', labelcolor='b')

    ax2 = ax.twinx()
    ax2.plot(labels, low_scores, label='Responses <= 0.5', marker='o', color='red')
    ax2.plot(labels, mid_scores, label='Responses > 0.5 & <=0.75', marker='o', color='green')
    ax2.plot(labels, high_scores, label='Responses > 0.75', marker='o', color='purple')
    ax2.set_ylabel('Count of Responses', color='k')
    ax2.tick_params(axis='y', labelcolor='k')

    ax.set_title('Performance Analysis by Chunks with RAG 4')
    fig.legend(loc='upper left', bbox_to_anchor=(0.12,0.7))

    plt.grid(True)
    plt.savefig('performance_analysis.png')
    plt.show()


def extract_data(file_path, start_row, end_row):
    data = pd.read_excel(file_path, header=0)

    template_columns = [f"similarity Advanced Ensemble Retrieval (bm25 and dense) {chunk} chunks" for chunk in range(3, 7)]
    crossencoder_columns = [f"similarity CrossEncoder Advanced Ensemble Retrieval (bm25 and dense) {chunk} chunks" for chunk in range(3, 7)]

    averages_template = data.loc[start_row, template_columns].values
    averages_crossencoder = data.loc[start_row, crossencoder_columns].values
    low_scores_template = data.loc[start_row + 1, template_columns].values
    low_scores_crossencoder = data.loc[start_row + 1, crossencoder_columns].values
    mid_scores_template = data.loc[start_row + 2, template_columns].values
    mid_scores_crossencoder = data.loc[start_row + 2, crossencoder_columns].values
    high_scores_template = data.loc[start_row + 3, template_columns].values
    high_scores_crossencoder = data.loc[start_row + 3, crossencoder_columns].values

    return averages_template, averages_crossencoder, low_scores_template, low_scores_crossencoder, mid_scores_template, mid_scores_crossencoder, high_scores_template, high_scores_crossencoder

#file_path = 'result3_hotpot.xlsx'
#averages_template, averages_crossencoder, low_scores_template, low_scores_crossencoder, mid_scores_template, mid_scores_crossencoder, high_scores_template, high_scores_crossencoder = extract_data(file_path, 500, 503)

#plot_performance(averages_template, low_scores_template, mid_scores_template, high_scores_template)