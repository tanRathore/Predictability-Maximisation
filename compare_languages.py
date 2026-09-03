import argparse
from itertools import combinations

from word_order_analysis import (
    extract_orders_from_sud,
    compute_distribution,
    calculate_entropy,
    calculate_cross_entropy,
    kl_divergence,
    compute_positional_distribution,
    calculate_positional_kl_divergence,
)

from word_order_analysis import (
    load_sud_trees,
    get_surface_order_from_tree,
    get_linearized_order_string,
)


def summarize_lang(name, path, length, lin_method, samples_per_tree, seed):
    orders_surface = extract_orders_from_sud(path, length_filter=length)
    dist_surface = compute_distribution(orders_surface)
    ent_surface = calculate_entropy(dist_surface)

    trees = load_sud_trees(path, length_filter=length)

    orders_linear = []
    for t in trees:
        for k in range(samples_per_tree):
            orders_linear.append(
                get_linearized_order_string(t, method=lin_method, seed=seed + k)
            )

    dist_linear = compute_distribution(orders_linear)
    ent_linear = calculate_entropy(dist_linear)

    pos_surface = compute_positional_distribution(orders_surface) if length else None
    pos_linear = compute_positional_distribution(orders_linear) if length else None

    return {
        "name": name,
        "path": path,
        "n_sentences": len(orders_surface),
        "n_trees": len(trees),
        "surface": {
            "entropy_bits": ent_surface,
            "unique_orders": len(dist_surface),
            "dist": dist_surface,
            "positional": pos_surface,
        },
        "linear": {
            "entropy_bits": ent_linear,
            "unique_orders": len(dist_linear),
            "dist": dist_linear,
            "positional": pos_linear,
        },
    }


def pairwise_metrics(A, B, key):
    P = A[key]["dist"]
    Q = B[key]["dist"]
    return {
        "cross_entropy_P_to_Q_bits": calculate_cross_entropy(P, Q),
        "cross_entropy_Q_to_P_bits": calculate_cross_entropy(Q, P),
        "kl_P_to_Q_bits": kl_divergence(P, Q),
        "kl_Q_to_P_bits": kl_divergence(Q, P),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=10)
    ap.add_argument("--lin_method", type=str, default="topological_random")
    ap.add_argument("--samples_per_tree", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--en", type=str, required=True)
    ap.add_argument("--hi", type=str, required=True)
    ap.add_argument("--ja", type=str, required=True)
    args = ap.parse_args()

    langs = [
        ("en", args.en),
        ("hi", args.hi),
        ("ja", args.ja),
    ]

    results = {}
    for name, path in langs:
        results[name] = summarize_lang(
            name=name,
            path=path,
            length=args.length,
            lin_method=args.lin_method,
            samples_per_tree=args.samples_per_tree,
            seed=args.seed,
        )

    print("\n=== Per-language summary ===")
    for name in results:
        r = results[name]
        print(
            f"{name}: n_sentences={r['n_sentences']}  "
            f"surface_H={r['surface']['entropy_bits']:.4f} (uniq={r['surface']['unique_orders']})  "
            f"linear_H={r['linear']['entropy_bits']:.4f} (uniq={r['linear']['unique_orders']})"
        )

    print("\n=== Pairwise comparisons (SURFACE) ===")
    for a, b in combinations(results.keys(), 2):
        m = pairwise_metrics(results[a], results[b], "surface")
        print(f"{a} vs {b}: {m}")

        if args.length:
            kl_pos = calculate_positional_kl_divergence(
                results[a]["surface"]["positional"], results[b]["surface"]["positional"]
            )
            print(f"{a} vs {b} positional KL (surface): {kl_pos}")

    print("\n=== Pairwise comparisons (LINEARIZED) ===")
    for a, b in combinations(results.keys(), 2):
        m = pairwise_metrics(results[a], results[b], "linear")
        print(f"{a} vs {b}: {m}")

        if args.length:
            kl_pos = calculate_positional_kl_divergence(
                results[a]["linear"]["positional"], results[b]["linear"]["positional"]
            )
            print(f"{a} vs {b} positional KL (linear): {kl_pos}")


if __name__ == "__main__":
    main()
