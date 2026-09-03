import os
import re
import csv
import math
import json
import argparse
from collections import Counter, defaultdict

import numpy as np
import networkx as nx
from conllu import parse
import matplotlib.pyplot as plt


def parse_lengths(spec: str):
    spec = spec.strip()
    if re.fullmatch(r"\d+", spec):
        return [int(spec)]
    if re.fullmatch(r"\d+\s*-\s*\d+", spec):
        a, b = [int(x.strip()) for x in spec.split("-")]
        if a > b:
            a, b = b, a
        return list(range(a, b + 1))
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    out = []
    for p in parts:
        if re.fullmatch(r"\d+", p):
            out.append(int(p))
        else:
            raise ValueError(f"Bad lengths token: {p}")
    return sorted(set(out))


def conllu_to_trees(path):
    with open(path, "r", encoding="utf-8") as f:
        sentences = parse(f.read())

    trees = []
    for sent in sentences:
        g = nx.DiGraph()
        g.add_node(0, upostag="ROOT", head=None)

        tokens = []
        for tok in sent:
            tid = tok.get("id", None)
            if not isinstance(tid, int):
                continue
            upos = tok.get("upostag", None)
            head = tok.get("head", None)
            if upos is None or head is None:
                continue
            tokens.append(tok)

        for tok in tokens:
            tid = tok["id"]
            g.add_node(
                tid,
                upostag=tok.get("upostag", "X"),
                head=tok.get("head", 0),
            )

        for tok in tokens:
            tid = tok["id"]
            head = tok.get("head", 0)
            if head is None:
                head = 0
            if head not in g:
                head = 0
            g.add_edge(head, tid)

        if nx.is_directed_acyclic_graph(g):
            trees.append(g)

    return trees


def surface_order_from_tree(tree, pos_mode="first"):
    nodes = [n for n in tree.nodes() if n != 0]
    nodes.sort()
    tags = []
    for n in nodes:
        upos = tree.nodes[n].get("upostag", "X")
        tags.append(upos[0] if pos_mode == "first" else upos)
    return "".join(tags)


def random_topo_linearization(tree, rng, pos_mode="first"):
    g = tree.copy()
    indeg = {n: 0 for n in g.nodes()}
    for u, v in g.edges():
        indeg[v] += 1

    avail = [n for n in g.nodes() if indeg[n] == 0]
    order = []

    while avail:
        i = rng.integers(0, len(avail))
        n = avail.pop(i)
        if n != 0:
            upos = g.nodes[n].get("upostag", "X")
            order.append(upos[0] if pos_mode == "first" else upos)

        for _, v in list(g.out_edges(n)):
            indeg[v] -= 1
            if indeg[v] == 0:
                avail.append(v)

    return "".join(order)


def dist_from_orders(orders):
    c = Counter(orders)
    total = sum(c.values())
    return {k: v / total for k, v in c.items()} if total > 0 else {}


def entropy_bits(dist):
    h = 0.0
    for p in dist.values():
        if p > 0:
            h -= p * math.log(p, 2)
    return h


def kl_bits(P, Q, eps=1e-12):
    out = 0.0
    for x, p in P.items():
        q = Q.get(x, 0.0)
        q = q if q > 0 else eps
        out += p * math.log(p / q, 2)
    return out


def js_bits(P, Q, eps=1e-12):
    keys = set(P) | set(Q)
    M = {}
    for k in keys:
        M[k] = 0.5 * (P.get(k, 0.0) + Q.get(k, 0.0))
        if M[k] <= 0:
            M[k] = eps
    return 0.5 * kl_bits(P, M, eps=eps) + 0.5 * kl_bits(Q, M, eps=eps)


def positional_dist(orders):
    if not orders:
        return {}

    L = len(orders[0])
    counts = [Counter() for _ in range(L)]
    for o in orders:
        for i, ch in enumerate(o):
            counts[i][ch] += 1

    out = []
    for c in counts:
        tot = sum(c.values())
        out.append({k: v / tot for k, v in c.items()} if tot > 0 else {})
    return out


def positional_kl(Ppos, Qpos, eps=1e-12):
    L = min(len(Ppos), len(Qpos))
    return [kl_bits(Ppos[i], Qpos[i], eps=eps) for i in range(L)]


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def save_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot_entropy(per_len_rows, outpath):
    lengths = sorted(set(r["length"] for r in per_len_rows))
    langs = sorted(set(r["lang"] for r in per_len_rows))

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for lang in langs:
        xs = []
        ys_surface = []
        ys_linear = []
        for L in lengths:
            r = next(rr for rr in per_len_rows if rr["length"] == L and rr["lang"] == lang)
            xs.append(L)
            ys_surface.append(r["surface_H"])
            ys_linear.append(r["linear_H"])
        ax.plot(xs, ys_surface, marker="o", label=f"{lang} surface")
        ax.plot(xs, ys_linear, marker="s", label=f"{lang} linear")

    ax.set_xlabel("Sentence length")
    ax.set_ylabel("Entropy (bits)")
    ax.set_title("Word-order entropy vs sentence length")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_pairwise(metric_rows, outpath, title):
    lengths = sorted(set(r["length"] for r in metric_rows))
    pairs = sorted(set(r["pair"] for r in metric_rows))

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for pair in pairs:
        xs = []
        ys = []
        for L in lengths:
            r = next(rr for rr in metric_rows if rr["length"] == L and rr["pair"] == pair)
            xs.append(L)
            ys.append(r["value"])
        ax.plot(xs, ys, marker="o", label=pair)

    ax.set_xlabel("Sentence length")
    ax.set_ylabel("JS divergence (bits)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", required=True, help="e.g. 5  or 3-15  or 3,4,5,6")
    ap.add_argument("--lin_method", default="topological_random", choices=["topological_random"])
    ap.add_argument("--samples_per_tree", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pos_mode", default="first", choices=["first", "full"])
    ap.add_argument("--balance", action="store_true", help="downsample each language to same #sentences per length")
    ap.add_argument("--outdir", default="results_sweep")

    ap.add_argument("--en", required=True)
    ap.add_argument("--hi", required=True)
    ap.add_argument("--ja", required=True)

    args = ap.parse_args()
    lengths = parse_lengths(args.lengths)
    rng = np.random.default_rng(args.seed)

    ensure_dir(args.outdir)

    lang_paths = {"en": args.en, "hi": args.hi, "ja": args.ja}
    lang_trees_all = {lang: conllu_to_trees(p) for lang, p in lang_paths.items()}

    per_len_rows = []
    pair_surface_rows = []
    pair_linear_rows = []
    poskl_surface_rows = []
    poskl_linear_rows = []

    for L in lengths:
        per_lang_orders_surface = {}
        per_lang_orders_linear = {}

        per_lang_counts = {}

        for lang, trees in lang_trees_all.items():
            trees_L = []
            for t in trees:
                n = len([x for x in t.nodes() if x != 0])
                if n == L:
                    trees_L.append(t)

            per_lang_counts[lang] = len(trees_L)

        if args.balance:
            m = min(per_lang_counts.values())
            if m == 0:
                continue
        else:
            m = None

        for lang, trees in lang_trees_all.items():
            trees_L = [t for t in trees if len([x for x in t.nodes() if x != 0]) == L]
            if not trees_L:
                per_lang_orders_surface[lang] = []
                per_lang_orders_linear[lang] = []
                continue

            if m is not None:
                if len(trees_L) > m:
                    idx = rng.choice(len(trees_L), size=m, replace=False)
                    trees_L = [trees_L[i] for i in idx]

            surface_orders = [surface_order_from_tree(t, pos_mode=args.pos_mode) for t in trees_L]

            linear_orders = []
            if args.lin_method == "topological_random":
                for t in trees_L:
                    for _ in range(args.samples_per_tree):
                        linear_orders.append(random_topo_linearization(t, rng, pos_mode=args.pos_mode))

            per_lang_orders_surface[lang] = surface_orders
            per_lang_orders_linear[lang] = linear_orders

            ds = dist_from_orders(surface_orders)
            dl = dist_from_orders(linear_orders)

            per_len_rows.append({
                "length": L,
                "lang": lang,
                "n_sentences": len(surface_orders),
                "uniq_surface": len(ds),
                "uniq_linear": len(dl),
                "surface_H": entropy_bits(ds),
                "linear_H": entropy_bits(dl),
            })

        langs = sorted(per_lang_orders_surface.keys())
        for i in range(len(langs)):
            for j in range(i + 1, len(langs)):
                a, b = langs[i], langs[j]

                Ps = dist_from_orders(per_lang_orders_surface[a])
                Qs = dist_from_orders(per_lang_orders_surface[b])
                Pl = dist_from_orders(per_lang_orders_linear[a])
                Ql = dist_from_orders(per_lang_orders_linear[b])

                pair_surface_rows.append({
                    "length": L,
                    "pair": f"{a}-{b}",
                    "value": js_bits(Ps, Qs),
                })
                pair_linear_rows.append({
                    "length": L,
                    "pair": f"{a}-{b}",
                    "value": js_bits(Pl, Ql),
                })

                Pspos = positional_dist(per_lang_orders_surface[a])
                Qspos = positional_dist(per_lang_orders_surface[b])
                Plpos = positional_dist(per_lang_orders_linear[a])
                Qlpos = positional_dist(per_lang_orders_linear[b])

                ks = positional_kl(Pspos, Qspos)
                kl = positional_kl(Plpos, Qlpos)

                poskl_surface_rows.append({
                    "length": L,
                    "pair": f"{a}->{b}",
                    "avg_pos_kl": float(np.mean(ks)) if ks else float("nan"),
                    "per_pos": json.dumps(ks),
                })
                poskl_linear_rows.append({
                    "length": L,
                    "pair": f"{a}->{b}",
                    "avg_pos_kl": float(np.mean(kl)) if kl else float("nan"),
                    "per_pos": json.dumps(kl),
                })

    save_csv(
        os.path.join(args.outdir, "per_language.csv"),
        per_len_rows,
        ["length", "lang", "n_sentences", "uniq_surface", "uniq_linear", "surface_H", "linear_H"],
    )
    save_csv(
        os.path.join(args.outdir, "pairwise_js_surface.csv"),
        pair_surface_rows,
        ["length", "pair", "value"],
    )
    save_csv(
        os.path.join(args.outdir, "pairwise_js_linear.csv"),
        pair_linear_rows,
        ["length", "pair", "value"],
    )
    save_csv(
        os.path.join(args.outdir, "poskl_surface.csv"),
        poskl_surface_rows,
        ["length", "pair", "avg_pos_kl", "per_pos"],
    )
    save_csv(
        os.path.join(args.outdir, "poskl_linear.csv"),
        poskl_linear_rows,
        ["length", "pair", "avg_pos_kl", "per_pos"],
    )

    plot_entropy(per_len_rows, os.path.join(args.outdir, "entropy_vs_length.png"))
    if pair_surface_rows:
        plot_pairwise(pair_surface_rows, os.path.join(args.outdir, "js_surface_vs_length.png"),
                      "Pairwise JS divergence (surface orders) vs length")
    if pair_linear_rows:
        plot_pairwise(pair_linear_rows, os.path.join(args.outdir, "js_linear_vs_length.png"),
                      "Pairwise JS divergence (linearized orders) vs length")

    print(f"\nSaved results to: {args.outdir}")
    print("Key outputs:")
    print("  per_language.csv")
    print("  pairwise_js_surface.csv / pairwise_js_linear.csv")
    print("  poskl_surface.csv / poskl_linear.csv")
    print("  entropy_vs_length.png")
    print("  js_surface_vs_length.png / js_linear_vs_length.png")


if __name__ == "__main__":
    main()
