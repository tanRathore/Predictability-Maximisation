# run_baseline_entropy_comparison.py (FINAL SCIENTIFIC VERSION)

import os
import random
from Third_baseline import Random_base
from word_order_analysis import (
    extract_orders_from_sud,
    get_orders_from_trees,
    compute_distribution,
    calculate_entropy
)
from conllu import parse
import networkx as nx

# --- 1. CONFIGURATION ---
SUD_DIRECTORY = "./SUD"
HINDI_FILE = os.path.join(SUD_DIRECTORY, "hi_sud-train.conllu")
ENGLISH_FILE = os.path.join(SUD_DIRECTORY, "en-sud-train.conllu")
JAPANESE_FILE = os.path.join(SUD_DIRECTORY, "ja_sud-train.conllu")
WEIGHTS = (0.7, 0.3)
NUM_TREES_TO_GENERATE = 5000
SENTENCE_LENGTH = 12

# --- 2. HELPER AND ANALYSIS FUNCTIONS ---
def get_all_trees_from_file(file_path):
    """
    Extracts ALL trees of a specific length from a SUD file to use as seeds.
    """
    print(f"Loading all seed trees of length {SENTENCE_LENGTH} from {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        sentences = parse(f.read())
    
    seed_trees = []
    for sentence in sentences:
        pos_tags = [word for word in sentence if word.get('upostag')]
        if len(pos_tags) == SENTENCE_LENGTH:
            tree = nx.DiGraph()
            for nodeinfo in sentence:
                entry = list(nodeinfo.items())
                if not entry[7][1] == 'punct':
                    tree.add_node(entry[0][1], form=entry[1][1], lemma=entry[2][1],
                                  upostag=entry[3][1], xpostag=entry[4][1],
                                  feats=entry[5][1], head=entry[6][1],
                                  deprel=entry[7][1], deps=entry[8][1], misc=entry[9][1])
            ROOT = 0
            tree.add_node(ROOT)
            for nodex in tree.nodes:
                if nodex != 0 and tree.has_node(tree.nodes[nodex]['head']):
                    tree.add_edge(tree.nodes[nodex]['head'], nodex, drel=tree.nodes[nodex]['deprel'])
            
            if 1 < len(tree.edges):
                seed_trees.append(tree)

    print(f"Found {len(seed_trees)} seed trees.")
    return seed_trees

def run_analysis_for_lang(lang_name, file_path):
    """
    Runs the complete analysis pipeline for a single language using multiple seeds.
    """
    print("\n" + "="*50)
    print(f"ANALYZING {lang_name.upper()}")
    print("="*50)
    
    # 1. Analyze the real language data
    real_orders = extract_orders_from_sud(file_path, length_filter=SENTENCE_LENGTH)
    if not real_orders:
        print(f"Could not find sentences of length {SENTENCE_LENGTH} for {lang_name}.")
        return

    real_dist = compute_distribution(real_orders)
    real_entropy = calculate_entropy(real_dist)
    print(f"Real {lang_name} Word Order Entropy (len={SENTENCE_LENGTH}): {real_entropy:.4f}")

    # 2. Get ALL possible seed trees from the corpus
    seed_trees = get_all_trees_from_file(file_path)
    if not seed_trees:
        print(f"Could not find any seed trees for {lang_name}.")
        return

    # 3. Generate the baseline by starting from a NEW SEED each time
    print(f"Generating {NUM_TREES_TO_GENERATE} trees by resampling from {len(seed_trees)} seeds...")
    baseline_trees = []
    while len(baseline_trees) < NUM_TREES_TO_GENERATE:
        # For each tree we want, pick a NEW random seed tree
        seed_tree = random.choice(seed_trees)
        generator = Random_base(seed_tree)
        
        # Ask the generator to find just ONE similar tree for THIS seed
        new_trees = generator.gen_random(1, weights=WEIGHTS) 
        
        if new_trees:
            baseline_trees.extend(new_trees)
            if len(baseline_trees) % 50 == 0:
                print(f"  ... {len(baseline_trees)} / {NUM_TREES_TO_GENERATE} generated ...")

    # 4. Analyze the generated baseline trees
    if baseline_trees:
        baseline_orders = get_orders_from_trees(baseline_trees)
        baseline_dist = compute_distribution(baseline_orders)
        baseline_entropy = calculate_entropy(baseline_dist)
        print(f"Baseline {lang_name} Word Order Entropy (generated): {baseline_entropy:.4f}")
        print(f" -> Difference (Real - Baseline): {real_entropy - baseline_entropy:.4f}")

# --- 3. MAIN EXECUTION ---
if __name__ == "__main__":
    print("--- Baseline Generation and Entropy Comparison ---")
    run_analysis_for_lang("Hindi", HINDI_FILE)
    run_analysis_for_lang("English", ENGLISH_FILE)
    run_analysis_for_lang("Japanese", JAPANESE_FILE) 
    # # --- Analyze JAPANESE ---
    # print("\n" + "="*50)
    # print("ANALYZING JAPANESE")
    # print("="*50)
    # real_japanese_orders = extract_orders_from_sud(JAPANESE_FILE, length_filter=SENTENCE_LENGTH)
    # if real_japanese_orders:
    #     real_japanese_dist = compute_distribution(real_japanese_orders)
    #     real_japanese_entropy = calculate_entropy(real_japanese_dist)
    #     print(f"Real Japanese Word Order Entropy (len={SENTENCE_LENGTH}): {real_japanese_entropy:.4f}")

    #     japanese_seed_tree = get_one_tree_from_file(JAPANESE_FILE)
    #     japanese_baseline_trees = generate_baseline_trees(japanese_seed_tree, NUM_TREES_TO_GENERATE)
        
    #     if japanese_baseline_trees:
    #         baseline_japanese_orders = get_orders_from_trees(japanese_baseline_trees)
    #         baseline_japanese_dist = compute_distribution(baseline_japanese_orders)
    #         baseline_japanese_entropy = calculate_entropy(baseline_japanese_dist)
    #         print(f"Baseline Japanese Word Order Entropy (generated): {baseline_japanese_entropy:.4f}")
    #         print(f" -> Difference (Real - Baseline): {real_japanese_entropy - baseline_japanese_entropy:.4f}")
    # else:
    #     print(f"Could not find sentences of length {SENTENCE_LENGTH} for Japanese to run analysis.")
