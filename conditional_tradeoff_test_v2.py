import os
import csv
import math
import random
import argparse
import numpy as np
import networkx as nx
from conllu import parse


PATHS = {
    "English": "./SUD/en-sud-train.conllu",
    "Hindi": "./SUD/hi_sud-train.conllu",
    "Japanese": "./SUD/ja_sud-train.conllu",
}

CONTENT_UPOS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM"}


def get_dep_len(tree, order):
    pos = {nid: i for i, nid in enumerate(order)}
    total = 0
    for dep in tree.nodes:
        if dep == 0:
            continue
        head = tree.nodes[dep].get("head")
        if head is None:
            continue
        if head in pos and dep in pos:
            total += abs(pos[head] - pos[dep])
    return total


def compute_subtree_sizes(tree):
    children = {n: list(tree.successors(n)) for n in tree.nodes}
    sizes = {}

    def dfs(u):
        if u in sizes:
            return sizes[u]
        s = 1
        for v in children.get(u, []):
            s += dfs(v)
        sizes[u] = s
        return s

    for n in tree.nodes:
        dfs(n)
    return sizes


def ic_content_between(tree, order):
    pos = {nid: i for i, nid in enumerate(order)}
    total = 0.0
    nodes = [n for n in tree.nodes if n != 0]

    for dep in nodes:
        head = tree.nodes[dep].get("head")
        if head is None or head == 0:
            continue
        if head not in pos or dep not in pos:
            continue

        p1, p2 = sorted((pos[head], pos[dep]))
        if p2 - p1 <= 1:
            continue

        for cand in nodes:
            if cand == head or cand == dep:
                continue
            cp = pos.get(cand, None)
            if cp is None or not (p1 < cp < p2):
                continue
            upos = tree.nodes[cand].get("upostag")
            total += 1.0 if upos in CONTENT_UPOS else 0.0

    return total


def ic_subtree_between(tree, order, log=False):
    pos = {nid: i for i, nid in enumerate(order)}
    subtree = compute_subtree_sizes(tree)
    total = 0.0
    nodes = [n for n in tree.nodes if n != 0]

    for dep in nodes:
        head = tree.nodes[dep].get("head")
        if head is None or head == 0:
            continue
        if head not in pos or dep not in pos:
            continue

        p1, p2 = sorted((pos[head], pos[dep]))
        if p2 - p1 <= 1:
            continue

        for cand in nodes:
            if cand == head or cand == dep:
                continue
            cp = pos.get(cand, None)
            if cp is None or not (p1 < cp < p2):
                continue
            s = float(subtree.get(cand, 1))
            total += math.log1p(s) if log else s

    return total


def ic_verb_cues(tree, order):
    pos = {nid: i for i, nid in enumerate(order)}
    total = 0.0
    nodes = [n for n in tree.nodes if n != 0]

    for dep in nodes:
        head = tree.nodes[dep].get("head")
        if head is None or head == 0:
            continue
        if head not in pos or dep not in pos:
            continue

        head_upos = tree.nodes[head].get("upostag")
        if head_upos not in ("VERB", "AUX"):
            continue

        if pos[dep] < pos[head]:
            dep_upos = tree.nodes[dep].get("upostag")
            total += 1.0 if dep_upos in CONTENT_UPOS else 0.0

    return total


def get_intervener_complexity(tree, order, ic_metric):
    if ic_metric == "content":
        return ic_content_between(tree, order)
    if ic_metric == "subtree":
        return ic_subtree_between(tree, order, log=False)
    if ic_metric == "subtree_log":
        return ic_subtree_between(tree, order, log=True)
    if ic_metric == "verb_cues":
        return ic_verb_cues(tree, order)
    raise ValueError(f"Unknown ic_metric: {ic_metric}")


def is_projective_under_order(tree, order):
    pos = {nid: i for i, nid in enumerate(order)}
    arcs = []
    for dep in tree.nodes:
        if dep == 0:
            continue
        head = tree.nodes[dep].get("head")
        if head is None or head == 0:
            continue
        if head not in pos or dep not in pos:
            continue
        i, j = pos[head], pos[dep]
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        arcs.append((a, b))

    for x in range(len(arcs)):
        a, b = arcs[x]
        for y in range(x + 1, len(arcs)):
            c, d = arcs[y]
            if a < c < b < d or c < a < d < b:
                return False
    return True


def load_trees(file_path, length_filter):
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()

    sentences = parse(data)
    out = []
    for sent in sentences:
        toks = [t for t in sent if isinstance(t.get("id"), int)]
        if len(toks) != length_filter:
            continue

        G = nx.DiGraph()
        G.add_node(0)

        for tok in toks:
            G.add_node(
                tok["id"],
                head=tok.get("head"),
                upostag=tok.get("upostag"),
                form=tok.get("form"),
            )
            if tok.get("head") is not None:
                G.add_edge(tok["head"], tok["id"])

        out.append(G)

    return out


def linearize_uniform(tree):
    nodes = [n for n in tree.nodes if n != 0]
    random.shuffle(nodes)
    return nodes


def linearize_projective_like(tree):
    def lin(h):
        children = list(tree.successors(h))
        random.shuffle(children)
        left, right = [], []
        for c in children:
            (left if random.random() < 0.5 else right).append(c)

        out = []
        for c in left:
            out.extend(lin(c))
        if h != 0:
            out.append(h)
        for c in right:
            out.extend(lin(c))
        return out

    return lin(0)


def sample_baseline_order(tree, baseline):
    if baseline == "uniform":
        return linearize_uniform(tree)
    if baseline == "projective_like":
        return linearize_projective_like(tree)
    raise ValueError(baseline)


def summarize_relative_quality(matched, real_ic, ic_objective):
    matched = np.asarray(matched, dtype=float)
    lt = float(np.mean(matched < real_ic))
    eq = float(np.mean(matched == real_ic))
    gt = float(np.mean(matched > real_ic))

    if ic_objective == "max":
        p_strict = lt
        p_leq = lt + eq
        p_mid = lt + 0.5 * eq
    else:
        p_strict = gt
        p_leq = gt + eq
        p_mid = gt + 0.5 * eq

    return lt, eq, gt, p_strict, p_mid, p_leq


def dl_matched_stats(tree, real_order, baseline, samples, min_matches, delta_start, delta_max, ic_metric, ic_objective):
    rdl = get_dep_len(tree, real_order)
    ric = get_intervener_complexity(tree, real_order, ic_metric)

    pairs = []
    for _ in range(samples):
        o = sample_baseline_order(tree, baseline)
        pairs.append((get_dep_len(tree, o), get_intervener_complexity(tree, o, ic_metric)))

    pairs = np.array(pairs, dtype=float)
    bdls = pairs[:, 0]
    bics = pairs[:, 1]

    delta = delta_start
    while True:
        mask = np.abs(bdls - rdl) <= delta
        nmatch = int(np.sum(mask))
        if nmatch >= min_matches:
            matched = bics[mask]
            break
        if delta >= delta_max:
            matched = bics[mask] if nmatch > 0 else np.array([], dtype=float)
            break
        delta += 1

    if matched.size == 0:
        return rdl, ric, delta, 0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan")

    lt, eq, gt, p_strict, p_mid, p_leq = summarize_relative_quality(matched, ric, ic_objective)
    diff = float(ric - np.mean(matched)) if matched.size else float("nan")
    return rdl, ric, delta, int(matched.size), lt, eq, gt, p_strict, p_mid, p_leq, diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", choices=["uniform", "projective_like"], default="projective_like")
    ap.add_argument("--ic_metric", choices=["content", "subtree", "subtree_log", "verb_cues"], default="verb_cues")
    ap.add_argument("--ic_objective", choices=["min", "max"], default="max")
    ap.add_argument("--length", type=int, default=12)
    ap.add_argument("--max_sentences", type=int, default=50)
    ap.add_argument("--samples", type=int, default=5000)
    ap.add_argument("--min_matches", type=int, default=80)
    ap.add_argument("--delta_start", type=int, default=0)
    ap.add_argument("--delta_max", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only_projective", action="store_true")
    ap.add_argument("--out", default="dl_matched_results.csv")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    rows = []

    for lang, path in PATHS.items():
        trees = load_trees(path, args.length)
        if len(trees) > args.max_sentences:
            trees = random.sample(trees, args.max_sentences)

        used = 0
        kept = 0
        p_strict_list, p_mid_list, p_leq_list = [], [], []
        matches, deltas, diffs = [], [], []

        for idx, tree in enumerate(trees):
            real_order = sorted([n for n in tree.nodes if n != 0])

            if args.only_projective:
                if not is_projective_under_order(tree, real_order):
                    continue

            used += 1
            rdl, ric, delta, nmatch, lt, eq, gt, p_strict, p_mid, p_leq, diff = dl_matched_stats(
                tree,
                real_order,
                args.baseline,
                args.samples,
                args.min_matches,
                args.delta_start,
                args.delta_max,
                args.ic_metric,
                args.ic_objective,
            )

            rows.append(
                {
                    "language": lang,
                    "sent_index": idx,
                    "baseline": args.baseline,
                    "ic_metric": args.ic_metric,
                    "ic_objective": args.ic_objective,
                    "length": args.length,
                    "real_dl": rdl,
                    "real_ic": ric,
                    "delta_used": delta,
                    "n_dl_matched": nmatch,
                    "frac_lt": lt,
                    "frac_eq": eq,
                    "frac_gt": gt,
                    "p_strict": p_strict,
                    "p_mid": p_mid,
                    "p_leq": p_leq,
                    "diff_real_minus_matched_mean": diff,
                }
            )

            if not np.isnan(p_mid):
                kept += 1
                p_strict_list.append(p_strict)
                p_mid_list.append(p_mid)
                p_leq_list.append(p_leq)
                matches.append(nmatch)
                deltas.append(delta)
                diffs.append(diff)

        if kept > 0:
            print(
                f"{lang} | baseline={args.baseline} | ic_metric={args.ic_metric} | ic_objective={args.ic_objective} | only_projective={args.only_projective} | "
                f"n_used={used} n_kept={kept} | "
                f"mean_p_strict={np.mean(p_strict_list):.3f} mean_p_mid={np.mean(p_mid_list):.3f} mean_p_leq={np.mean(p_leq_list):.3f} | "
                f"mean_diff(real-matched_mean)={np.mean(diffs):.3f} | mean_matches={np.mean(matches):.1f} mean_delta={np.mean(deltas):.2f}"
            )
        else:
            print(
                f"{lang} | baseline={args.baseline} | ic_metric={args.ic_metric} | ic_objective={args.ic_objective} | only_projective={args.only_projective} | "
                f"n_used={used} n_kept=0"
            )

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            w.writeheader()
            w.writerows(rows)

    print(f"\n[SAVED] {args.out}")
    print("Interpretation:")
    print("- With ic_objective=max: larger IC = more predictability cues (better). You want Hindi/Japanese mean_p_mid > English.")
    print("- With ic_objective=min: smaller IC = less complexity (better). You want Hindi/Japanese mean_p_mid > English.")


if __name__ == "__main__":
    main()
