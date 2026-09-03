import os
from word_order_analysis import (
    extract_orders_from_sud,
    compute_distribution,
    compute_positional_distribution,
    kl_divergence,
    calculate_positional_kl_divergence,
    euclidean_distance
)
SUD_DIRECTORY = "./SUD" 
LANGUAGES = {
    "hindi": "hi_sud-train.conllu",
    "english": "en-sud-train.conllu",
    "japanese": "ja_sud-train.conllu"
}
SENTENCE_LENGTH = 5 
def run_comparison():
    print(f" Running Word Order Comparison for Sentence Length: {SENTENCE_LENGTH} \n")

    orders_by_lang = {}
    for lang, filename in LANGUAGES.items():
        file_path = os.path.join(SUD_DIRECTORY, filename)
        if os.path.exists(file_path):
            orders_by_lang[lang] = extract_orders_from_sud(file_path, length_filter=SENTENCE_LENGTH)
            print(f"Found {len(orders_by_lang[lang])} sentences of length {SENTENCE_LENGTH} for {lang.capitalize()}.")
            if not orders_by_lang[lang]:
                 print(f"Warning: No sentences of length {SENTENCE_LENGTH} found for {lang.capitalize()}. Try a different length.")
        else:
            print(f"Error: File not found for {lang.capitalize()}: {file_path}")

    print("\n" + "="*50)
    print("Part A: Global Word Order Dissimilarity")
    print("="*50)

    dist_by_lang = {lang: compute_distribution(orders) for lang, orders in orders_by_lang.items()}
    langs = list(orders_by_lang.keys())
    
    print("\n1. Global KL Divergence (Cross-Entropy):")
    for i in range(len(langs)):
        for j in range(len(langs)):
            if i != j:
                lang1, lang2 = langs[i], langs[j]
                if dist_by_lang[lang1] and dist_by_lang[lang2]:
                    kl_val = kl_divergence(dist_by_lang[lang1], dist_by_lang[lang2])
                    print(f"  KL({lang1.capitalize()} || {lang2.capitalize()}) = {kl_val:.4f}")

    print("\n2. Euclidean Distance (between most common word orders):")
    rep_orders = {}
    for lang, dist in dist_by_lang.items():
        if dist:
            most_common_order = max(dist, key=dist.get)
            rep_orders[lang] = most_common_order
            print(f"  Most common order for {lang.capitalize()}: '{most_common_order}'")

    for i in range(len(langs)):
        for j in range(i + 1, len(langs)):
            lang1, lang2 = langs[i], langs[j]
            if lang1 in rep_orders and lang2 in rep_orders:
                distance = euclidean_distance(rep_orders[lang1], rep_orders[lang2])
                print(f"  Distance({lang1.capitalize()} <-> {lang2.capitalize()}): {distance:.2f}")

    print("\n" + "="*50)
    print("Part B: Positional Dissimilarity")
    print("="*50)
    
    positional_dist_by_lang = {lang: compute_positional_distribution(orders) for lang, orders in orders_by_lang.items()}

    print("\n3. Positional KL Divergence (Cross-Entropy at each position):")
    for i in range(len(langs)):
        for j in range(len(langs)):
            if i != j:
                lang1, lang2 = langs[i], langs[j]
                if positional_dist_by_lang[lang1] and positional_dist_by_lang[lang2]:
                    print(f"\nComparing {lang1.capitalize()} to {lang2.capitalize()}:")
                    positional_kl = calculate_positional_kl_divergence(positional_dist_by_lang[lang1], positional_dist_by_lang[lang2])
                    for pos, kl_val in positional_kl.items():
                        print(f"  Position {pos+1}: KL = {kl_val:.4f}")
                    if positional_kl:
                        avg_kl = sum(positional_kl.values()) / len(positional_kl)
                        print(f"  -> Average Positional KL: {avg_kl:.4f}")

if __name__ == "__main__":
    run_comparison()