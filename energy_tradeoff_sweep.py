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

CONTENT_UPOS = {"NOUN", "PROPN", "VERB", "AUX", "ADJ", "ADV", "NUM"}


def upos_to_cat(upos: str) -> str:
    if upos in ("NOUN", "PROPN"):
        return "N"
    if upos in ("VERB", "AUX"):
        return "V"
    if upos in ("ADJ", "ADV"):
        return "A"
    if upos == "NUM":
        return "D"
    return "O"


CATS = ["N", "V", "A", "D", "O"]


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
    return float(total)


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

        # cue counts only if dependent is BEFORE the verb
        if pos[dep] < pos[head]:
            dep_upos = tree.nodes[dep].get("upostag")
            total += 1.0 if dep_upos in CONTENT_UPOS else 0.0

    return total


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


def sample_candidate_order(tree, baseline):
    if baseline == "uniform":
        return linearize_uniform(tree)
    if baseline == "projective_like":
        return linearize_projective_like(tree)
    raise ValueError(baseline)


def choose_order_energy(tree, baseline, n_candidates, alpha, beta, temperature):
    cand_orders = []
    energies = []

    for _ in range(n_candidates):
        o = sample_candidate_order(tree, baseline)
        dl = get_dep_len(tree, o)
        ic = ic_verb_cues(tree, o)
        E = alpha * dl - beta * ic
        cand_orders.append(o)
        energies.append(E)

    energies = np.asarray(energies, dtype=float)
    Emin = float(np.min(energies))
    # stable softmax
    w = np.exp(-(energies - Emin) / max(temperature, 1e-9))
    wsum = float(np.sum(w))
    if not np.isfinite(wsum) or wsum <= 0:
        idx = random.randrange(len(cand_orders))
        return cand_orders[idx], float("nan"), float("nan"), float("nan")

    r = random.random() * wsum
    acc = 0.0
    idx = 0
    for i, wi in enumerate(w):
        acc += float(wi)
        if acc >= r:
            idx = i
            break

    chosen = cand_orders[idx]
    dl = get_dep_len(tree, chosen)
    ic = ic_verb_cues(tree, chosen)
    E = alpha * dl - beta * ic
    return chosen, dl, ic, E


def orders_to_posseq(tree, order):
    seq = []
    for nid in order:
        upos = tree.nodes[nid].get("upostag")
        seq.append(upos_to_cat(upos if upos is not None else "O"))
    return seq


def positional_probs(seqs, L):
    counts = np.zeros((L, len(CATS)), dtype=float)
    for s in seqs:
        for i, ch in enumerate(s):
            if i >= L:
                break
            j = CATS.index(ch)
            counts[i, j] += 1.0
    denom = counts.sum(axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return counts / denom


def entropy_pos(P):
    eps = 1e-12
    H = -np.sum(P * np.log2(np.clip(P, eps, 1.0)), axis=1)
    return float(np.mean(H))


def js_divergence_pos(P, Q):
    eps = 1e-12
    P = np.clip(P, eps, 1.0)
    Q = np.clip(Q, eps, 1.0)
    P = P / P.sum(axis=1, keepdims=True)
    Q = Q / Q.sum(axis=1, keepdims=True)
    M = 0.5 * (P + Q)

    KL_PM = np.sum(P * (np.log2(P) - np.log2(M)), axis=1)
    KL_QM = np.sum(Q * (np.log2(Q) - np.log2(M)), axis=1)
    JS = 0.5 * (KL_PM + KL_QM)
    return float(np.mean(JS))


def parse_float_list(spec):
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    return [float(x) for x in parts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=12)
    ap.add_argument("--max_sentences", type=int, default=80)
    ap.add_argument("--baseline", choices=["projective_like", "uniform"], default="projective_like")
    ap.add_argument("--only_projective", action="store_true")

    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--betas", type=str, default="0,0.25,0.5,1,2,4,8")
    ap.add_argument("--temperature", type=float, default=1.0)

    ap.add_argument("--candidates", type=int, default=400)
    ap.add_argument("--reps_per_tree", type=int, default=1)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="energy_sweep_len12.csv")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    betas = parse_float_list(args.betas)

    # Load trees per language
    trees_by_lang = {}
    real_posseqs_by_lang = {}
    real_probs_by_lang = {}

    for lang, path in PATHS.items():
        trees = load_trees(path, args.length)
        if len(trees) > args.max_sentences:
            trees = random.sample(trees, args.max_sentences)

        kept = []
        real_seqs = []
        for t in trees:
            surface = sorted([n for n in t.nodes if n != 0])
            if args.only_projective and (not is_projective_under_order(t, surface)):
                continue
            kept.append(t)
            real_seqs.append(orders_to_posseq(t, surface))

        trees_by_lang[lang] = kept
        real_posseqs_by_lang[lang] = real_seqs
        real_probs_by_lang[lang] = positional_probs(real_seqs, args.length)

    rows = []
    targets = list(PATHS.keys())

    for src_lang, trees in trees_by_lang.items():
        if not trees:
            continue

        for beta in betas:
            gen_seqs = []
            dl_list = []
            ic_list = []
            e_list = []

            for t in trees:
                for _ in range(args.reps_per_tree):
                    order, dl, ic, E = choose_order_energy(
                        t,
                        args.baseline,
                        args.candidates,
                        args.alpha,
                        beta,
                        args.temperature,
                    )
                    gen_seqs.append(orders_to_posseq(t, order))
                    dl_list.append(dl)
                    ic_list.append(ic)
                    e_list.append(E)

            Pgen = positional_probs(gen_seqs, args.length)
            Hgen = entropy_pos(Pgen)
            mean_dl = float(np.nanmean(dl_list))
            mean_ic = float(np.nanmean(ic_list))

            for tgt in targets:
                J = js_divergence_pos(Pgen, real_probs_by_lang[tgt])
                rows.append(
                    {
                        "source_lang": src_lang,
                        "target_lang": tgt,
                        "length": args.length,
                        "baseline": args.baseline,
                        "only_projective": bool(args.only_projective),
                        "alpha": args.alpha,
                        "beta": beta,
                        "temperature": args.temperature,
                        "candidates": args.candidates,
                        "reps_per_tree": args.reps_per_tree,
                        "n_trees": len(trees),
                        "n_samples": len(gen_seqs),
                        "mean_gen_dl": mean_dl,
                        "mean_gen_ic": mean_ic,
                        "mean_gen_entropy": Hgen,
                        "js_to_target": J,
                    }
                )

            print(
                f"{src_lang} beta={beta:.3g} | mean_dl={mean_dl:.2f} mean_ic={mean_ic:.2f} mean_H={Hgen:.3f} | "
                f"JS->(En,Hi,Ja)=({js_divergence_pos(Pgen, real_probs_by_lang['English']):.3f},"
                f"{js_divergence_pos(Pgen, real_probs_by_lang['Hindi']):.3f},"
                f"{js_divergence_pos(Pgen, real_probs_by_lang['Japanese']):.3f})"
            )

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            w.writeheader()
            w.writerows(rows)

    print(f"\n[SAVED] {args.out}")
    print("Next: pick beta that MINIMIZES js_to_target when source_lang == target_lang (best fit).")


if __name__ == "__main__":
    main()
