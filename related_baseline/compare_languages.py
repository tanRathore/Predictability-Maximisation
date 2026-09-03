#does the following:
#   1. Extracts word orders (POS tag sequences) from SUD files for each language.
#   2. Computes the probability distribution for each language.
#   3. Calculates pairwise KL divergence between these distributions.
#   4. Optionally, computes Euclidean distances between representative word orders.
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from word_order_analysis import (
    extract_orders_from_sud,
    compute_distribution,
    kl_divergence,
    euclidean_distance
)

# Step 1: Define file paths for each language's SUD file 
sud_directory = "/Users/tanishqsingh/Desktop/IIT_KANPUR/SUD Files"
file_paths = {
    "chinese": os.path.join(sud_directory, "zh_sud-train.conllu"),
    "english": os.path.join(sud_directory, "en_sud-train.conllu"),
    "czech":   os.path.join(sud_directory, "cs_sud-train.conllu"),
    "arabic":  os.path.join(sud_directory, "ar_sud-train.conllu"),
    "hindi":   os.path.join(sud_directory, "hi_sud-train.conllu")
}

#  Step 2: Extract word orders for each language 
orders_by_lang = {}
for lang, path in file_paths.items():
    try:
        orders = extract_orders_from_sud(path)
        orders_by_lang[lang] = orders
        print(f"{lang.capitalize()} orders (sample): {orders[:5]}")
    except Exception as e:
        print(f"Error extracting orders for {lang}: {e}")

# Step 3: Compute probability distributions for each language 
dist_by_lang = {}
for lang, orders in orders_by_lang.items():
    if orders:
        distribution = compute_distribution(orders)
        dist_by_lang[lang] = distribution
        print(f"{lang.capitalize()} distribution: {distribution}")
    else:
        print(f"No orders found for {lang}")

#Step 4: Compute Pairwise KL Divergence
def pairwise_kl(dist_dict):
    """
    Compute pairwise KL divergence for all languages.
    Returns:
      - langs: List of language keys.
      - kl_matrix: A 2D numpy array of KL divergence values,
                   where kl_matrix[i][j] = KL(divergence from language i to language j)
    """
    langs = list(dist_dict.keys())
    kl_matrix = np.zeros((len(langs), len(langs)))
    for i, lang1 in enumerate(langs):
        for j, lang2 in enumerate(langs):
            kl_val = kl_divergence(dist_dict[lang1], dist_dict[lang2])
            kl_matrix[i, j] = kl_val
    return langs, kl_matrix

langs, kl_matrix = pairwise_kl(dist_by_lang)
print("\nPairwise KL Divergence Matrix:")
for i, lang1 in enumerate(langs):
    for j, lang2 in enumerate(langs):
        print(f"{lang1} -> {lang2}: {kl_matrix[i, j]:.4f}", end="  ")
    print()

# Step 5: Compute Pairwise Euclidean Distances (Representative Orders) 
def pairwise_euclidean(orders_dict):
    """
    Compute Euclidean distances between representative word orders from each language.
    Uses the first order from each language as a representative.
    """
    langs = list(orders_dict.keys())
    euclid_matrix = np.zeros((len(langs), len(langs)))
    rep_orders = {lang: orders[0] for lang, orders in orders_dict.items() if orders}
    for i, lang1 in enumerate(langs):
        for j, lang2 in enumerate(langs):
            if lang1 in rep_orders and lang2 in rep_orders:
                d = euclidean_distance(rep_orders[lang1], rep_orders[lang2])
                euclid_matrix[i, j] = d
            else:
                euclid_matrix[i, j] = np.nan
    return langs, euclid_matrix

langs_e, euclid_matrix = pairwise_euclidean(orders_by_lang)
print("\nPairwise Euclidean Distance Matrix (Representative Orders):")
for i, lang1 in enumerate(langs_e):
    for j, lang2 in enumerate(langs_e):
        print(f"{lang1} -> {lang2}: {euclid_matrix[i, j]:.4f}", end="  ")
    print()

#  Step 6: Visualization 
# 6a: Heatmap of KL Divergence
plt.figure(figsize=(8,6))
sns.heatmap(kl_matrix, xticklabels=langs, yticklabels=langs, annot=True, cmap="viridis")
plt.title("Pairwise KL Divergence (P(lang1) || P(lang2))")
plt.xlabel("Language")
plt.ylabel("Language")
plt.tight_layout()
plt.show()

# 6b: Heatmap of Euclidean Distances
plt.figure(figsize=(8,6))
sns.heatmap(euclid_matrix, xticklabels=langs_e, yticklabels=langs_e, annot=True, cmap="magma")
plt.title("Pairwise Euclidean Distances (Representative Orders)")
plt.xlabel("Language")
plt.ylabel("Language")
plt.tight_layout()
plt.show()

# 6c: Bar charts of individual language distributions
for lang, distribution in dist_by_lang.items():
    plt.figure()
    orders = list(distribution.keys())
    probabilities = list(distribution.values())
    plt.bar(orders, probabilities, color="skyblue")
    plt.title(f"{lang.capitalize()} Word Order Distribution")
    plt.xlabel("Word Order")
    plt.ylabel("Probability")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

#Step 7: Specific Comparisons
print("KL divergence from Czech to Hindi:", kl_matrix[langs.index("czech")][langs.index("hindi")])
print("KL divergence from Arabic to English:", kl_matrix[langs.index("arabic")][langs.index("english")])
print("KL divergence from Arabic to Hindi:", kl_matrix[langs.index("arabic")][langs.index("hindi")])
  