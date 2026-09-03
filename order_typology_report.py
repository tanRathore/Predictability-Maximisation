import os
import csv
import math
import random
import argparse
from collections import defaultdict

import numpy as np
import networkx as nx
from conllu import parse


PATHS = {
    "English": "./SUD/en-sud-train.conllu",
    "Hindi": "./SUD/hi_sud-train.conllu",
    "Japanese": "./SUD/ja_sud-train.conllu",
}


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
                deprel=tok.get("deprel"),
                form=tok.get("form"),
            )
            if tok.get("head") is not None:
                G.add_edge(tok["head"], tok["id"])

        out.append(G)

    return out


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


def verb_cues_score(tree, order):
    pos = {nid: i for i, nid in enumerate(order)}
    nodes = [n for n in tree.nodes if n != 0]

    cues = 0.0
    for h in nodes:
        if tree.nodes[h].get("upostag") not in ("VERB", "AUX"):
            continue
        if h not in pos:
            continue
        hp = pos[h]
        for dep in tree.successors(h):
            if dep == 0 or dep not in pos:
                continue
            if pos[dep] < hp:
                cues += 1.0
    return cues


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


def sample_energy_order(tree, alpha, beta, temperature, candidates):
    cand_orders = []
    dl_list = []
    ic_list = []
    E_list = []

    for _ in range(candidates):
        o = linearize_projective_like(tree)
        dl = get_dep_len(tree, o)
        ic = verb_cues_score(tree, o)  # maximize
        E = alpha * dl - beta * ic
        cand_orders.append(o)
        dl_list.append(dl)
        ic_list.append(ic)
        E_list.append(E)

    E_arr = np.array(E_list, dtype=float)
    Emin = float(np.min(E_arr))
    logits = -(E_arr - Emin) / float(temperature)
    w = np.exp(logits - np.max(logits))
    w = w / np.sum(w)

    idx = int(np.random.choice(len(cand_orders), p=w))
    return cand_orders[idx], dl_list[idx], ic_list[idx]


def coarse_tag(upos):
    if upos in ("NOUN", "PROPN", "PRON"):
        return "N"
    if upos in ("VERB", "AUX"):
        return "V"
    if upos in ("ADJ", "ADV"):
        return "A"
    return "O"


def positional_distribution(seqs, L):
    cats = ["N", "V", "A", "O"]
    ci = {c: i for i, c in enumerate(cats)}
    counts = np.zeros((L, len(cats)), dtype=float)

    for s in seqs:
        if len(s) != L:
            continue
        for i, c in enumerate(s):
            counts[i, ci.get(c, ci["O"])] += 1.0

    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    probs = counts / row_sums
    return probs


def js_divergence(P, Q, eps=1e-12):
    P = np.clip(P, eps, 1.0)
    Q = np.clip(Q, eps, 1.0)
    P = P / P.sum()
    Q = Q / Q.sum()
    M = 0.5 * (P + Q)

    def kl(A, B):
        return float(np.sum(A * np.log2(A / B)))

    return 0.5 * kl(P, M) + 0.5 * kl(Q, M)


def flatten_positional_probs(probs):
    return probs.reshape(-1)


def typology_metrics(tree, order):
    pos = {nid: i for i, nid in enumerate(order)}
    nodes = [n for n in tree.nodes if n != 0]

    head_final = []
    verb_positions = []
    verb_final = []

    for dep in nodes:
        head = tree.nodes[dep].get("head")
        if head is None or head == 0:
            continue
        if head not in pos or dep not in pos:
            continue
        head_final.append(1.0 if pos[head] > pos[dep] else 0.0)

    L = len(order)
    for n in nodes:
        up = tree.nodes[n].get("upostag")
        if up in ("VERB", "AUX") and n in pos:
            vp = pos[n]
            verb_positions.append(vp)
            verb_final.append(1.0 if vp == (L - 1) else 0.0)

    out = {
        "head_final_rate": float(np.mean(head_final)) if head_final else float("nan"),
        "verb_final_rate": float(np.mean(verb_final)) if verb_final else float("nan"),
        "mean_verb_pos": float(np.mean(verb_positions)) if verb_positions else float("nan"),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=12)
    ap.add_argument("--max_sentences", type=int, default=80)
    ap.add_argument("--only_projective", action="store_true")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--betas", type=str, default="0.1,4,8,10")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--candidates", type=int, default=800)
    ap.add_argument("--reps_per_tree", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="typology_report.csv")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    betas = [float(x.strip()) for x in args.betas.split(",") if x.strip()]

    real_posseqs = {}
    for lang, path in PATHS.items():
        trees = load_trees(path, args.length)
        if args.max_sentences and len(trees) > args.max_sentences:
            trees = random.sample(trees, args.max_sentences)

        seqs = []
        for t in trees:
            order = sorted([n for n in t.nodes if n != 0])
            if args.only_projective and not is_projective_under_order(t, order):
                continue
            seqs.append("".join(coarse_tag(t.nodes[n].get("upostag")) for n in order))
        real_posseqs[lang] = seqs

    real_probs = {lang: positional_distribution(seqs, args.length) for lang, seqs in real_posseqs.items()}
    real_flat = {lang: flatten_positional_probs(p) for lang, p in real_probs.items()}

    rows = []

    for source_lang, path in PATHS.items():
        trees = load_trees(path, args.length)
        if args.max_sentences and len(trees) > args.max_sentences:
            trees = random.sample(trees, args.max_sentences)

        filtered = []
        for t in trees:
            order = sorted([n for n in t.nodes if n != 0])
            if args.only_projective and not is_projective_under_order(t, order):
                continue
            filtered.append(t)
        trees = filtered

        for beta in betas:
            gen_posseqs = []
            dls = []
            ics = []
            hf = []
            vf = []
            vpos = []

            for t in trees:
                for _ in range(args.reps_per_tree):
                    o, dl, ic = sample_energy_order(
                        t,
                        alpha=args.alpha,
                        beta=beta,
                        temperature=args.temperature,
                        candidates=args.candidates,
                    )
                    dls.append(dl)
                    ics.append(ic)

                    m = typology_metrics(t, o)
                    hf.append(m["head_final_rate"])
                    vf.append(m["verb_final_rate"])
                    vpos.append(m["mean_verb_pos"])

                    gen_posseqs.append("".join(coarse_tag(t.nodes[n].get("upostag")) for n in o))

            gen_probs = positional_distribution(gen_posseqs, args.length)
            gen_flat = flatten_positional_probs(gen_probs)

            js_en = js_divergence(gen_flat, real_flat["English"])
            js_hi = js_divergence(gen_flat, real_flat["Hindi"])
            js_ja = js_divergence(gen_flat, real_flat["Japanese"])

            row = {
                "source_lang": source_lang,
                "beta": beta,
                "n_orders": len(gen_posseqs),
                "mean_dl": float(np.mean(dls)) if dls else float("nan"),
                "mean_verb_cues": float(np.mean(ics)) if ics else float("nan"),
                "head_final_rate": float(np.nanmean(hf)) if hf else float("nan"),
                "verb_final_rate": float(np.nanmean(vf)) if vf else float("nan"),
                "mean_verb_pos": float(np.nanmean(vpos)) if vpos else float("nan"),
                "js_to_English": float(js_en),
                "js_to_Hindi": float(js_hi),
                "js_to_Japanese": float(js_ja),
            }
            rows.append(row)

            print(
                f"{source_lang} beta={beta:g} | mean_dl={row['mean_dl']:.2f} "
                f"mean_verb_cues={row['mean_verb_cues']:.2f} "
                f"head_final={row['head_final_rate']:.3f} verb_final={row['verb_final_rate']:.3f} "
                f"mean_verb_pos={row['mean_verb_pos']:.2f} | "
                f"JS->(En,Hi,Ja)=({row['js_to_English']:.3f},{row['js_to_Hindi']:.3f},{row['js_to_Japanese']:.3f})"
            )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            w.writeheader()
            w.writerows(rows)

    print(f"\n[SAVED] {args.out}")
    print("What to look for:")
    print("- As beta increases, verb_cues should rise and head_final/verb_final should rise (more SOV-like).")
    print("- English should look SVO-ish at low beta and move toward Hindi/Japanese-ish at high beta.")


if __name__ == "__main__":
    main()
