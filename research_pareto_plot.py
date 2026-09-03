import os
import random
import csv
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from conllu import parse

PATHS = {
    "English": "./SUD/en-sud-train.conllu",
    "Hindi": "./SUD/hi_sud-train.conllu",
    "Japanese": "./SUD/ja_sud-train.conllu"
}

COLORS = {"English": "blue", "Hindi": "green", "Japanese": "red"}

SENTENCE_LENGTH = 12
MAX_SENTENCES = 50
RANDOM_SAMPLES = 50
SEED = 42

BASELINE = "projective_like"  # "uniform" or "projective_like"
OUTFIG = "pareto_frontier_results.png"
OUTCSV = "pareto_stats.csv"


def get_dep_len(tree, order):
    pos_map = {node_id: i for i, node_id in enumerate(order)}
    total_dl = 0
    for node in tree.nodes:
        if node == 0:
            continue
        head = tree.nodes[node].get("head")
        if head is not None and head in pos_map and node in pos_map:
            total_dl += abs(pos_map[head] - pos_map[node])
    return total_dl


def get_intervener_complexity(tree, order):
    pos_map = {node_id: i for i, node_id in enumerate(order)}
    total_ic = 0

    for node in tree.nodes:
        if node == 0:
            continue
        head = tree.nodes[node].get("head")

        if head is None or head not in pos_map or node not in pos_map:
            continue

        p1, p2 = sorted((pos_map[node], pos_map[head]))

        interveners = 0
        for candidate in tree.nodes:
            if candidate == 0 or candidate == node or candidate == head:
                continue
            if candidate not in pos_map:
                continue
            c_pos = pos_map[candidate]
            if p1 < c_pos < p2:
                c_head = tree.nodes[candidate].get("head")
                if c_head == head:
                    interveners += 1

        total_ic += interveners

    return total_ic


def load_and_process(file_path, length_filter):
    print(f"Loading {file_path}...")
    if not os.path.exists(file_path):
        print(f"ERROR: File not found: {file_path}")
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()

    sentences = parse(data)
    valid_trees = []

    for sent in sentences:
        tokens = [t for t in sent if isinstance(t.get("id"), int)]
        if len(tokens) != length_filter:
            continue

        G = nx.DiGraph()
        G.add_node(0)

        for tok in tokens:
            G.add_node(
                tok["id"],
                head=tok.get("head"),
                form=tok.get("form"),
                upostag=tok.get("upostag"),
            )
            if tok.get("head") is not None:
                G.add_edge(tok["head"], tok["id"])

        valid_trees.append(G)

    print(f"  Found {len(valid_trees)} trees of length {length_filter}")
    return valid_trees


def linearize_uniform(tree):
    nodes = [n for n in tree.nodes if n != 0]
    random.shuffle(nodes)
    return nodes


def linearize_projective_like(tree):
    # Projective-like: recursively linearize each subtree as a contiguous block.
    # Each child subtree is randomly assigned to head's left or right.
    def lin(head):
        children = list(tree.successors(head))
        random.shuffle(children)

        left = []
        right = []
        for c in children:
            if random.random() < 0.5:
                left.append(c)
            else:
                right.append(c)

        out = []
        for c in left:
            out.extend(lin(c))

        if head != 0:
            out.append(head)

        for c in right:
            out.extend(lin(c))

        return out

    return lin(0)


def linearize_baseline(tree):
    if BASELINE == "uniform":
        return linearize_uniform(tree)
    if BASELINE == "projective_like":
        return linearize_projective_like(tree)
    raise ValueError(f"Unknown BASELINE: {BASELINE}")


def run():
    random.seed(SEED)
    np.random.seed(SEED)

    results = {}

    for lang, path in PATHS.items():
        trees = load_and_process(path, SENTENCE_LENGTH)

        if len(trees) > MAX_SENTENCES:
            trees = random.sample(trees, MAX_SENTENCES)

        res = {
            "real_dl": [],
            "real_ic": [],
            "rand_dl": [],
            "rand_ic": [],
            "p_dl": [],
            "p_ic": [],
            "p_both": [],
            "p_real_dominates": [],
        }

        print(f"Processing {lang}...")
        for tree in trees:
            real_order = sorted([n for n in tree.nodes if n != 0])

            rdl = get_dep_len(tree, real_order)
            ric = get_intervener_complexity(tree, real_order)

            res["real_dl"].append(rdl)
            res["real_ic"].append(ric)

            bdls = []
            bics = []
            for _ in range(RANDOM_SAMPLES):
                rand_order = linearize_baseline(tree)
                bdl = get_dep_len(tree, rand_order)
                bic = get_intervener_complexity(tree, rand_order)
                bdls.append(bdl)
                bics.append(bic)

            res["rand_dl"].extend(bdls)
            res["rand_ic"].extend(bics)

            bdls_np = np.array(bdls, dtype=float)
            bics_np = np.array(bics, dtype=float)

            p_dl = float(np.mean(bdls_np <= rdl))
            p_ic = float(np.mean(bics_np <= ric))
            p_both = float(np.mean((bdls_np <= rdl) & (bics_np <= ric)))
            p_real_dom = float(np.mean((bdls_np >= rdl) & (bics_np >= ric)))

            res["p_dl"].append(p_dl)
            res["p_ic"].append(p_ic)
            res["p_both"].append(p_both)
            res["p_real_dominates"].append(p_real_dom)

        results[lang] = res

    print("\nGenerating Pareto Plot...")
    plt.figure(figsize=(12, 10))

    for lang, data in results.items():
        if not data["real_dl"]:
            continue

        color = COLORS.get(lang, "gray")

        plt.scatter(
            data["rand_dl"],
            data["rand_ic"],
            color=color,
            alpha=0.1,
            s=10,
            label=f"{lang} (Random, {BASELINE})",
        )

        plt.scatter(
            data["real_dl"],
            data["real_ic"],
            color=color,
            marker="*",
            s=150,
            edgecolor="black",
            zorder=10,
            label=f"{lang} (Real)",
        )

        mean_rdl = float(np.mean(data["real_dl"]))
        mean_ric = float(np.mean(data["real_ic"]))
        plt.text(mean_rdl, mean_ric, f" {lang}", fontsize=12, fontweight="bold", color="black")

    plt.title(
        f"Efficiency Trade-off: Dependency Length vs. Intervener Complexity\n"
        f"(Sentence Length = {SENTENCE_LENGTH}, baseline = {BASELINE})",
        fontsize=16,
    )
    plt.xlabel("Total Dependency Length (minimize)", fontsize=12)
    plt.ylabel("Intervener Complexity (minimize)", fontsize=12)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.grid(True, alpha=0.3)
    plt.savefig(OUTFIG, dpi=300)
    print(f"\n[SUCCESS] Graph saved to: {OUTFIG}")

    rows = []
    for lang, data in results.items():
        if not data["real_dl"]:
            continue

        rows.append(
            {
                "language": lang,
                "baseline": BASELINE,
                "sentence_length": SENTENCE_LENGTH,
                "max_sentences": MAX_SENTENCES,
                "random_samples_per_sentence": RANDOM_SAMPLES,
                "n_sentences": len(data["real_dl"]),
                "mean_real_dl": float(np.mean(data["real_dl"])),
                "mean_real_ic": float(np.mean(data["real_ic"])),
                "mean_p_dl_random_beats_real": float(np.mean(data["p_dl"])) if data["p_dl"] else float("nan"),
                "mean_p_ic_random_beats_real": float(np.mean(data["p_ic"])) if data["p_ic"] else float("nan"),
                "mean_p_both_random_dominates_real": float(np.mean(data["p_both"])) if data["p_both"] else float("nan"),
                "mean_p_real_dominates_random": float(np.mean(data["p_real_dominates"])) if data["p_real_dominates"] else float("nan"),
            }
        )

    if rows:
        with open(OUTCSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        print(f"[STATS] Wrote: {OUTCSV}")
        for r in rows:
            print(
                f"{r['language']} | baseline={r['baseline']} | "
                f"mean_real_dl={r['mean_real_dl']:.2f} mean_real_ic={r['mean_real_ic']:.2f} | "
                f"p_both(random<=real in both)={r['mean_p_both_random_dominates_real']:.3f} | "
                f"p_real_dominates={r['mean_p_real_dominates_random']:.3f}"
            )

    print(
        "\nInterpretation guide:\n"
        "- p_both ~ 0 means random almost never beats real on BOTH objectives (good for efficiency claim).\n"
        "- p_real_dominates closer to 1 means real beats random on BOTH objectives frequently (strong evidence).\n"
        "- Compare uniform vs projective_like baselines: projective_like is the more linguistically meaningful baseline."
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", choices=["uniform", "projective_like"], default=BASELINE)
    ap.add_argument("--length", type=int, default=SENTENCE_LENGTH)
    ap.add_argument("--max_sentences", type=int, default=MAX_SENTENCES)
    ap.add_argument("--samples", type=int, default=RANDOM_SAMPLES)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--outfig", default=OUTFIG)
    ap.add_argument("--outcsv", default=OUTCSV)
    args = ap.parse_args()

    BASELINE = args.baseline
    SENTENCE_LENGTH = args.length
    MAX_SENTENCES = args.max_sentences
    RANDOM_SAMPLES = args.samples
    SEED = args.seed
    OUTFIG = args.outfig
    OUTCSV = args.outcsv

    run()
