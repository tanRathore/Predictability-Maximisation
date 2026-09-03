import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from word_order_analysis import (
    extract_orders_from_sud,
    compute_distribution,
    compute_positional_distribution,
    kl_divergence,
    calculate_positional_kl_divergence,
)

# --- 1. CONFIGURATION ---
SUD_DIRECTORY = "./SUD"
LANGUAGES = {
    "hindi": "hi_sud-train.conllu",
    "english": "en-sud-train.conllu",
    "japanese": "ja_sud-train.conllu"
}
SENTENCE_LENGTH = 5
LANG_NAMES = [lang.capitalize() for lang in LANGUAGES.keys()]

# --- 2. DATA ANALYSIS FUNCTION (Adapted from your script) ---
def get_analysis_data():
    """
    Runs the core analysis and returns the data structures needed for plotting.
    """
    orders_by_lang = {}
    for lang, filename in LANGUAGES.items():
        file_path = os.path.join(SUD_DIRECTORY, filename)
        if os.path.exists(file_path):
            orders_by_lang[lang] = extract_orders_from_sud(file_path, length_filter=SENTENCE_LENGTH)
        else:
            print(f"Warning: File not found for {lang.capitalize()}: {file_path}")
            orders_by_lang[lang] = []

    dist_by_lang = {lang: compute_distribution(orders) for lang, orders in orders_by_lang.items()}
    positional_dist_by_lang = {lang: compute_positional_distribution(orders) for lang, orders in orders_by_lang.items()}
    
    langs = list(LANGUAGES.keys())
    num_langs = len(langs)
    kl_matrix = np.zeros((num_langs, num_langs))
    avg_pos_kl_matrix = np.zeros((num_langs, num_langs))
    
    all_positional_kl = {}

    for i in range(num_langs):
        for j in range(num_langs):
            lang1, lang2 = langs[i], langs[j]
            if i != j and dist_by_lang[lang1] and dist_by_lang[lang2]:
                kl_matrix[i, j] = kl_divergence(dist_by_lang[lang1], dist_by_lang[lang2])
                
                positional_kl = calculate_positional_kl_divergence(positional_dist_by_lang[lang1], positional_dist_by_lang[lang2])
                all_positional_kl[f'{lang1.capitalize()} -> {lang2.capitalize()}'] = positional_kl
                
                if positional_kl:
                    avg_pos_kl_matrix[i, j] = sum(positional_kl.values()) / len(positional_kl)

    return kl_matrix, avg_pos_kl_matrix, all_positional_kl

# --- 3. VISUALIZATION ---
def create_dashboard(kl_matrix, avg_pos_kl_matrix, all_positional_kl):
    """
    Generates and saves a comprehensive visualization dashboard.
    """
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 2) # Create a 2x2 grid for subplots

    # Plot 1: Global KL Divergence Heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(kl_matrix, annot=True, fmt=".4f", cmap="viridis",
                xticklabels=LANG_NAMES, yticklabels=LANG_NAMES, ax=ax1)
    ax1.set_title('Global KL Divergence (P(Row) || P(Column))', fontsize=16, pad=20)
    ax1.set_ylabel('P (Source Distribution)', fontsize=12)
    ax1.set_xlabel('Q (Target Distribution)', fontsize=12)

    # Plot 2: Average Positional KL Divergence Heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    sns.heatmap(avg_pos_kl_matrix, annot=True, fmt=".4f", cmap="magma",
                xticklabels=LANG_NAMES, yticklabels=LANG_NAMES, ax=ax2)
    ax2.set_title('Average Positional KL Divergence', fontsize=16, pad=20)
    ax2.set_ylabel('P (Source Distribution)', fontsize=12)
    ax2.set_xlabel('Q (Target Distribution)', fontsize=12)

    # Plot 3: Detailed Positional KL Bar Chart
    ax3 = fig.add_subplot(gs[1, :]) # This plot spans the entire bottom row
    
    bar_data = {
        'Comparison': [],
        'Position': [],
        'KL Divergence': []
    }
    for comparison, pos_kl_data in all_positional_kl.items():
        for pos, kl_val in pos_kl_data.items():
            bar_data['Comparison'].append(comparison)
            bar_data['Position'].append(f'Pos {pos+1}')
            bar_data['KL Divergence'].append(kl_val)
    
    import pandas as pd
    df = pd.DataFrame(bar_data)
    
    sns.barplot(x='Position', y='KL Divergence', hue='Comparison', data=df, ax=ax3, palette='muted')
    ax3.set_title('Detailed KL Divergence at Each Sentence Position', fontsize=16, pad=20)
    ax3.set_xlabel('Sentence Position', fontsize=12)
    ax3.set_ylabel('KL Divergence', fontsize=12)
    ax3.legend(title='Comparison (P || Q)', loc='upper right')

    # Final adjustments and saving
    fig.suptitle(f'Cross-Linguistic Word Order Dissimilarity Analysis (Sentence Length = {SENTENCE_LENGTH})', fontsize=22, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('advanced_visualization_report.png', dpi=300)
    print("Generated 'advanced_visualization_report.png'")
    plt.show()


# --- 4. MAIN EXECUTION ---
if __name__ == "__main__":
    print("Running analysis to generate data for visualizations...")
    kl_matrix, avg_pos_kl_matrix, all_positional_kl = get_analysis_data()
    print("Data generation complete. Creating dashboard...")
    create_dashboard(kl_matrix, avg_pos_kl_matrix, all_positional_kl)