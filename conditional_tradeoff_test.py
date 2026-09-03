import os
import csv
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


def get_dep_len(tree, order):
    pos = {nid: i for i, nid in enumerate(order)}
    total = 0
    for node in tree.nodes:
        if node == 0:
            continue
        head = tree.nodes[node].get("head")
        if head is None:
            continue
        if head in pos and node in pos:
            total += abs(pos[head] - pos[node])
    return total


def get_intervener_complexity(tree, order):
    pos = {nid: i for i, nid in enumerate(order)}
    total_ic = 0

    for dep in tree.nodes:
        if dep == 0:
            continue
        head = tree.nodes[dep].get("head")
        if head is None:
            continue
        if head not in pos or dep not in pos:
            continue

        p1, p2 = sorted((pos[dep], pos[head]))
        interveners = 0

        for cand in tree.nodes:
            if cand in (0, dep, head):
                continue
            if cand not in pos:
                continue
            cp = pos[cand]
            if p1 < cp < p2:
                if tree.nodes[cand].get("head") == head:
                    interveners += 1

        total_ic += interveners

    return total_ic


def is_projective_under_surface(tree, surface_order):
    # Detect crossing dependencies under the given surface order.
    pos = {nid: i for i, nid in enumerate(surface_order)}
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

    # crossing: a < c < b < d
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
            G.add_node(tok["id"], head=tok.get("head"), upostag=tok.get("upostag"), form=tok.get("form"))
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


def dl_matched_p_value(tree, real_order, baseline, samples, min_matches, delta_start, delta_max):
    rdl = get_dep_len(tree, real_order)
    ric = get_intervener_complexity(tree, real_order)

    pairs = []
    for _ in range(samples):
        o = sample_baseline_order(tree, baseline)
        pairs.append((get_dep_len(tree, o), get_intervener_complexity(tree, o)))

    pairs = np.array(pairs, dtype=float)
    bdls = pairs[:, 0]
    bics = pairs[:, 1]

    delta = delta_start
    matched = None

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
        return rdl, ric, delta, 0, float("nan")

    # minimize IC: p = fraction of matched baselines with IC <= real IC
    p_ic = float(np.mean(matched <= ric))
    return rdl, ric, delta, int(matched.size), p_ic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", choices=["uniform", "projective_like"], default="projective_like")
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

        kept = 0
        used = 0
        pvals = []
        matches = []
        deltas = []

        for idx, tree in enumerate(trees):
            real_order = sorted([n for n in tree.nodes if n != 0])

            if args.only_projective:
                if not is_projective_under_surface(tree, real_order):
                    continue

            used += 1
            rdl, ric, delta, nmatch, p_ic = dl_matched_p_value(
                tree,
                real_order,
                args.baseline,
                args.samples,
                args.min_matches,
                args.delta_start,
                args.delta_max,
            )

            rows.append(
                {
                    "language": lang,
                    "sent_index": idx,
                    "baseline": args.baseline,
                    "length": args.length,
                    "real_dl": rdl,
                    "real_ic": ric,
                    "delta_used": delta,
                    "n_dl_matched": nmatch,
                    "p_ic_given_dl": p_ic,
                }
            )

            if not np.isnan(p_ic):
                kept += 1
                pvals.append(p_ic)
                matches.append(nmatch)
                deltas.append(delta)

        if kept > 0:
            print(
                f"{lang} | baseline={args.baseline} | only_projective={args.only_projective} | "
                f"n_used={used} n_kept={kept} | "
                f"mean_p(IC|DL)={np.mean(pvals):.3f} median={np.median(pvals):.3f} | "
                f"mean_matches={np.mean(matches):.1f} mean_delta={np.mean(deltas):.2f}"
            )
        else:
            print(f"{lang} | baseline={args.baseline} | only_projective={args.only_projective} | n_used={used} n_kept=0")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            w.writeheader()
            w.writerows(rows)

    print(f"\n[SAVED] {args.out}")
    print("Key quantity: p_ic_given_dl. If mean_p is SMALL (<0.5), real tends to have LOWER IC than DL-matched baseline (good).")
    print("If mean_p is LARGE (>0.5), baseline tends to beat real on IC even when DL is matched (bad / metric problem).")


if __name__ == "__main__":
    main()
