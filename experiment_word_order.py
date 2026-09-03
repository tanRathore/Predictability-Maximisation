import os
import math
import matplotlib.pyplot as plt
import networkx as nx
from conllu import parse
from Third_baseline import Random_base
from word_order_analysis import (
    extract_orders_from_sud,
    get_orders_from_trees,
    compute_distribution,
    kl_divergence,
    euclidean_distance
)

# Step 1: Extract Real Word Orders from SUD Files
# Update the directory path to your SUD files
directory = "/Users/tanishqsingh/Desktop/IIT_KANPUR/SUD Files"
hindi_file = os.path.join(directory, "hi_sud-train.conllu")
english_file = os.path.join(directory, "en_sud-train.conllu")

# Extract orders using the helper function from word_order_analysis.py
hindi_orders = extract_orders_from_sud(hindi_file)
english_orders = extract_orders_from_sud(english_file)

# Compute distributions for real orders
Q_hindi = compute_distribution(hindi_orders)
Q_english = compute_distribution(english_orders)

# Step 2: Compute Real Distribution Metrics
# Calculate entropy for the real distributions
def compute_overall_entropy(distribution):
    return sum(-p * math.log(p, 2) for p in distribution.values() if p > 0)

hindi_entropy = compute_overall_entropy(Q_hindi)
english_entropy = compute_overall_entropy(Q_english)
kl_hindi_to_english = kl_divergence(Q_hindi, Q_english)

print("Real Hindi Entropy:", hindi_entropy)
print("Real English Entropy:", english_entropy)
print("KL Divergence (Hindi -> English):", kl_hindi_to_english)
# Step 3: Generate Random Trees Using Weighted Baselines
def get_one_tree_from_file(file_path):
    """
    Extract one example tree from a SUD file (for testing).
    Returns a networkx.DiGraph if found.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        sentences = parse(f.read())
    for sentence in sentences[1:]: 
        tree = nx.DiGraph()
        for nodeinfo in sentence:
            entry = list(nodeinfo.items())
            if not entry[7][1] == 'punct':
                tree.add_node(entry[0][1],
                              form=entry[1][1],
                              lemma=entry[2][1],
                              upostag=entry[3][1],
                              xpostag=entry[4][1],
                              feats=entry[5][1],
                              head=entry[6][1],
                              deprel=entry[7][1],
                              deps=entry[8][1],
                              misc=entry[9][1])
        ROOT = 0
        tree.add_node(ROOT)
        for nodex in tree.nodes:
            if nodex != 0 and tree.has_node(tree.nodes[nodex]['head']):
                tree.add_edge(tree.nodes[nodex]['head'], nodex, drel=tree.nodes[nodex]['deprel'])
        if 1 < len(tree.edges) < 12:
            return tree
    return None

# Get one seed tree for Hindi and English
hindi_tree = get_one_tree_from_file(hindi_file)
english_tree = get_one_tree_from_file(english_file)

# Define weight combinations for the experiment
weights_list = [(0.65, 0.35), (0.5, 0.5), (0.8, 0.2), (0.35, 0.65),
                (0.2, 0.8), (0.95, 0.05), (0.05, 0.95)]

def generate_trees_for_language(tree, weights, num_trees=20, num_cross_real=10):
    """
    Generate a specified number of random trees for a given seed tree and weight combination.
    """
    generated_trees = []
    generator = Random_base(tree)
    while len(generated_trees) < num_trees:
        trees = generator.gen_random(num_cross_real, weights=weights)
        if trees:
            generated_trees.extend(trees)
    return generated_trees[:num_trees]

# Generate random trees for Hindi (similar code can be used for English)
results_hindi = {}
for w in weights_list:
    trees = generate_trees_for_language(hindi_tree, w, num_trees=20, num_cross_real=10)
    results_hindi[w] = trees

# Step 4: Analyze Word Order Distributions from Generated Trees
def analyze_tree_orders(trees, real_distribution):
    """
    Extract orders from a list of trees, compute the distribution,
    and calculate KL divergence with the provided real distribution.
    """
    orders = get_orders_from_trees(trees)
    P_random = compute_distribution(orders)
    kl_val = kl_divergence(P_random, real_distribution)
    return P_random, kl_val

analysis_hindi = {}
for w, trees in results_hindi.items():
    dist, kl_val = analyze_tree_orders(trees, Q_hindi)
    analysis_hindi[w] = {"distribution": dist, "kl_divergence": kl_val}
    print(f"Weights {w} - KL Divergence (Random -> Hindi): {kl_val}")


# Step 5: Euclidean Distance Example
# Example: compare specific orders
dist_example = euclidean_distance("NVNN", "NNNV")
print("Euclidean distance between 'NVNN' and 'NNNV':", dist_example)


# Step 6: Visualization of Random Tree Distribution
for w, data_dict in analysis_hindi.items():
    plt.figure()
    orders = list(data_dict["distribution"].keys())
    probabilities = list(data_dict["distribution"].values())
    plt.bar(orders, probabilities)
    plt.title(f"Hindi Random Tree Distribution for Weights {w}")
    plt.xlabel("Word Order")
    plt.ylabel("Probability")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()