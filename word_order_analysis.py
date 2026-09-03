from __future__ import annotations

from collections import defaultdict
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable, Any

import numpy as np
import networkx as nx
from conllu import parse


# ----------------------------
# Your original functions (kept)
# ----------------------------

def extract_orders_from_sud(file_path, length_filter=None):
    """
    Extracts word orders from a SUD file[cite: 42].
    Each sentence is represented by the sequence of the first letters of the 'upostag' field[cite: 42].

    :param file_path: Path to the SUD file[cite: 43].
    :param length_filter: (Optional) An integer to only include sentences of this length.
    :return: List of word order strings (e.g., ["NNNV", "NNVN", ...])[cite: 43].
    """
    orders = []
    with open(file_path, "r", encoding="utf-8") as f:
        sentences = parse(f.read())
    for sentence in sentences:
        pos_sequence = "".join([word['upostag'][0] for word in sentence if word.get('upostag')])

        if length_filter:
            if len(pos_sequence) == length_filter:
                orders.append(pos_sequence)
        else:
            orders.append(pos_sequence)
    return orders


def get_order_from_tree(tree):
    """
    Converts a dependency tree into a word order string.
    Assumes that each node (except the abstract root) has a 'upostag' field.

    NOTE: This is a dependency-consistent linearization if your edges are head->dep,
    because nx.topological_sort enforces head before dependent.

    :param tree: A networkx.DiGraph representing the dependency tree.
    :return: A string representing the word order (e.g., "NNNV").
    """
    order = "".join([tree.nodes[node].get('upostag', 'X')[0]
                     for node in nx.topological_sort(tree) if node != 0])
    return order


def get_orders_from_trees(trees):
    """
    Converts a list of trees into a list of word order strings.

    :param trees: List of networkx.DiGraph trees.
    :return: List of word order strings.
    """
    return [get_order_from_tree(tree) for tree in trees]


def compute_distribution(orders):
    """
    Computes the probability distribution over word orders.

    :param orders: List of word order strings.
    :return: Dictionary mapping each order to its probability.
    """
    dist = {}
    total = len(orders)
    if total == 0:
        return {}
    for order in orders:
        dist[order] = dist.get(order, 0) + 1
    for order in dist:
        dist[order] /= total
    return dist


def calculate_entropy(probabilities):
    """
    Calculates the entropy of a given probability distribution.

    :param probabilities: Dictionary of probabilities.
    :return: Entropy value (float).
    """
    entropy = 0.0
    for p in probabilities.values():
        if p > 0:
            entropy -= p * math.log(p, 2)
    return entropy


def calculate_cross_entropy(P, Q):
    """
    Calculates the cross-entropy between two probability distributions.

    :param P: Distribution P as a dictionary.
    :param Q: Distribution Q as a dictionary.
    :return: Cross-entropy value (float).
    """
    cross_entropy = 0.0
    for x, p_val in P.items():
        q_val = Q.get(x, 1e-10)  # Use a small epsilon if Q(x) is zero
        cross_entropy -= p_val * math.log(q_val, 2)
    return cross_entropy


def kl_divergence(P, Q):
    """
    Computes the Kullback-Leibler (KL) divergence from Q to P:
    KL(P||Q) = sum_x P(x) * log(P(x) / Q(x))

    :param P: Distribution P as a dictionary.
    :param Q: Distribution Q as a dictionary.
    :return: KL divergence value (float).
    """
    kl = 0.0
    for x, p_val in P.items():
        q_val = Q.get(x, 1e-10)
        kl += p_val * math.log(p_val / q_val, 2)
    return kl


def euclidean_distance(order1, order2):
    """
    Calculates the Euclidean distance between two word order strings.
    Each character is converted to its ASCII value; strings are padded if lengths differ.

    :param order1: First word order string.
    :param order2: Second word order string.
    :return: Euclidean distance (float).
    """
    max_len = max(len(order1), len(order2))
    vec1 = [ord(c) for c in order1.ljust(max_len, ' ')]
    vec2 = [ord(c) for c in order2.ljust(max_len, ' ')]
    return np.linalg.norm(np.array(vec1) - np.array(vec2))


def compute_positional_distribution(orders):
    """
    Computes the probability distribution of POS tags at each position.
    This directly answers: "probability of each category in first position, second position, etc."
    :param orders: A list of word order strings (e.g., ['NNV', 'NVN']).
    :return: A dict: position -> dict(POS -> prob)
    """
    if not orders:
        return {}

    max_len = len(orders[0])
    positional_counts = {i: defaultdict(int) for i in range(max_len)}

    for order in orders:
        for i, pos_tag in enumerate(order):
            positional_counts[i][pos_tag] += 1

    positional_dist = {i: defaultdict(float) for i in range(max_len)}
    for pos, counts in positional_counts.items():
        total = sum(counts.values())
        if total > 0:
            for tag, count in counts.items():
                positional_dist[pos][tag] = count / total

    return positional_dist


def calculate_positional_kl_divergence(dist1, dist2):
    """
    Calculates the KL divergence at each position between two languages / models.
    This gives you "Cross-entropy at all positions".
    :param dist1: Positional distribution for language/model 1.
    :param dist2: Positional distribution for language/model 2.
    :return: position -> KL value
    """
    positional_kl = {}
    positions = sorted(list(set(dist1.keys()) & set(dist2.keys())))

    for pos in positions:
        p_dist = dist1.get(pos, {})
        q_dist = dist2.get(pos, {})
        positional_kl[pos] = kl_divergence(p_dist, q_dist)

    return positional_kl


# ----------------------------
# NEW: SUD -> nx.DiGraph (tree builder you said you might be missing)
# ----------------------------

def load_sud_trees(
    file_path: str,
    length_filter: Optional[int] = None,
    root_id: int = 0,
) -> List[nx.DiGraph]:
    """
    Loads a SUD/UD conllu file and builds one nx.DiGraph per sentence.

    Conventions:
    - node ids are token ids (int)
    - root node id is `root_id` (default 0)
    - directed edges are head -> dependent
    - node attrs include: upostag, xpos, deprel, head (int), form, lemma

    If length_filter is provided, only include sentences with exactly that many tokens (int ids).
    """
    with open(file_path, "r", encoding="utf-8") as f:
        sentences = parse(f.read())

    trees: List[nx.DiGraph] = []

    for sent in sentences:
        tokens = []
        for tok in sent:
            tid = tok.get("id")
            if isinstance(tid, int):
                tokens.append(tok)

        if length_filter is not None and len(tokens) != length_filter:
            continue

        g = nx.DiGraph()
        g.add_node(root_id, upostag="ROOT", xpos="ROOT", deprel="root", head=None, form="ROOT", lemma="ROOT")

        for tok in tokens:
            tid = int(tok["id"])
            head = tok.get("head", root_id)
            if head is None:
                head = root_id
            if not isinstance(head, int):
                head = root_id

            g.add_node(
                tid,
                form=tok.get("form"),
                lemma=tok.get("lemma"),
                upostag=tok.get("upostag", "X"),
                xpos=tok.get("xpos"),
                deprel=tok.get("deprel"),
                head=head,
            )

        # add edges head -> dep
        for tok in tokens:
            dep = int(tok["id"])
            head = tok.get("head", root_id)
            if head is None or not isinstance(head, int):
                head = root_id
            if head == dep:
                head = root_id
            if head not in g:
                head = root_id
            g.add_edge(head, dep)

        # quick sanity: must be a DAG
        if not nx.is_directed_acyclic_graph(g):
            continue

        trees.append(g)

    return trees


# ----------------------------
# NEW: Surface order from tree (true token order, NOT topological)
# ----------------------------

def get_surface_order_from_tree(tree: nx.DiGraph, root_id: int = 0) -> str:
    """
    Surface order = token id order (1..n) for SUD/UD where token id is the surface index.
    """
    nodes = [n for n in tree.nodes() if n != root_id and isinstance(n, int)]
    nodes.sort()
    return "".join([tree.nodes[n].get("upostag", "X")[0] for n in nodes])


def get_surface_orders_from_trees(trees: List[nx.DiGraph], root_id: int = 0) -> List[str]:
    return [get_surface_order_from_tree(t, root_id=root_id) for t in trees]


# ----------------------------
# NEW: Dependency-consistent linearizations
# ----------------------------

def _random_topological_order(tree: nx.DiGraph, rng: random.Random) -> List[Any]:
    """
    Kahn topological sort with random tie-breaking.
    Returns a list of nodes including root; caller can drop root.
    """
    indeg = {n: 0 for n in tree.nodes()}
    for u, v in tree.edges():
        indeg[v] += 1

    zero = [n for n, d in indeg.items() if d == 0]
    rng.shuffle(zero)

    out = []
    while zero:
        n = zero.pop()
        out.append(n)
        for succ in tree.successors(n):
            indeg[succ] -= 1
            if indeg[succ] == 0:
                zero.append(succ)
        rng.shuffle(zero)

    if len(out) != len(indeg):
        raise nx.NetworkXUnfeasible("Graph has a cycle; cannot topologically sort.")
    return out


def _projective_linearize(tree: nx.DiGraph, root_id: int, rng: random.Random) -> List[int]:
    """
    Simple projective linearization:
    - recursively linearize children
    - randomly split children into left/right
    - output: left-subtrees + head + right-subtrees
    """
    children = {n: list(tree.successors(n)) for n in tree.nodes()}

    def rec(node: int) -> List[int]:
        ch = children.get(node, [])
        if not ch:
            return [node]

        rng.shuffle(ch)
        split = rng.randint(0, len(ch))
        left = ch[:split]
        right = ch[split:]

        left_out: List[int] = []
        for c in left:
            left_out.extend(rec(c))

        right_out: List[int] = []
        for c in right:
            right_out.extend(rec(c))

        return left_out + [node] + right_out

    # choose a real root child of abstract root_id
    roots = [n for n in tree.nodes() if n != root_id and tree.in_degree(n) == 1 and root_id in tree.predecessors(n)]
    if not roots:
        # fallback: any node with indeg 1 and not root_id, else smallest non-root
        roots = [n for n in tree.nodes() if n != root_id and tree.in_degree(n) == 1]
    if not roots:
        roots = [n for n in tree.nodes() if n != root_id]

    r = roots[0]
    out = rec(r)
    out = [n for n in out if n != root_id]
    return out


def linearize_tree(
    tree: nx.DiGraph,
    method: str = "topological_random",
    seed: Optional[int] = None,
    root_id: int = 0,
) -> List[int]:
    """
    Returns a list of node ids (excluding root_id) in a linearized order.

    method:
      - "topological_deterministic": nx.topological_sort (deterministic)
      - "topological_random": random Kahn topological sort (dependency-consistent)
      - "projective_random": recursive projective linearization (optional)
    """
    rng = random.Random(seed)

    if method == "topological_deterministic":
        order = list(nx.topological_sort(tree))
    elif method == "topological_random":
        order = _random_topological_order(tree, rng)
    elif method == "projective_random":
        order = [root_id] + _projective_linearize(tree, root_id=root_id, rng=rng)
    else:
        raise ValueError(f"Unknown linearization method: {method}")

    return [n for n in order if n != root_id]


def get_linearized_order_string(
    tree: nx.DiGraph,
    method: str = "topological_random",
    seed: Optional[int] = None,
    root_id: int = 0,
) -> str:
    nodes = linearize_tree(tree, method=method, seed=seed, root_id=root_id)
    return "".join([tree.nodes[n].get("upostag", "X")[0] for n in nodes])


def get_linearized_orders_from_trees(
    trees: List[nx.DiGraph],
    method: str = "topological_random",
    samples_per_tree: int = 1,
    seed: int = 0,
    root_id: int = 0,
) -> List[str]:
    """
    Generate multiple linearizations per tree (useful if method is random).
    """
    rng = random.Random(seed)
    orders: List[str] = []
    for t in trees:
        for _ in range(samples_per_tree):
            s = rng.randint(0, 2**31 - 1)
            orders.append(get_linearized_order_string(t, method=method, seed=s, root_id=root_id))
    return orders


# ----------------------------
# NEW: Distribution distance (JS) + bootstrap CI
# ----------------------------

def jensen_shannon_divergence(P: Dict[str, float], Q: Dict[str, float], eps: float = 1e-12) -> float:
    """
    JS divergence in bits (log base 2).
    """
    keys = set(P.keys()) | set(Q.keys())
    if not keys:
        return 0.0

    Pn = {k: max(P.get(k, 0.0), 0.0) for k in keys}
    Qn = {k: max(Q.get(k, 0.0), 0.0) for k in keys}
    sp = sum(Pn.values())
    sq = sum(Qn.values())
    if sp == 0 or sq == 0:
        return 0.0

    for k in keys:
        Pn[k] /= sp
        Qn[k] /= sq

    M = {k: 0.5 * (Pn[k] + Qn[k]) for k in keys}

    def _kl(A, B):
        out = 0.0
        for k in keys:
            a = A.get(k, 0.0)
            b = B.get(k, 0.0)
            if a > 0:
                out += a * math.log(a / max(b, eps), 2)
        return out

    return 0.5 * _kl(Pn, M) + 0.5 * _kl(Qn, M)


def bootstrap_js_ci(
    surface_orders: List[str],
    linear_orders: List[str],
    n_boot: int = 500,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """
    Bootstraps JS(surface_dist, linear_dist) by resampling orders with replacement.
    Returns: (js_point_estimate, ci_low, ci_high)
    """
    rng = random.Random(seed)
    if not surface_orders or not linear_orders:
        return 0.0, 0.0, 0.0

    P = compute_distribution(surface_orders)
    Q = compute_distribution(linear_orders)
    js0 = jensen_shannon_divergence(P, Q)

    js_samples = []
    for _ in range(n_boot):
        s = [surface_orders[rng.randrange(len(surface_orders))] for _ in range(len(surface_orders))]
        l = [linear_orders[rng.randrange(len(linear_orders))] for _ in range(len(linear_orders))]
        js_samples.append(jensen_shannon_divergence(compute_distribution(s), compute_distribution(l)))

    js_samples.sort()
    lo = js_samples[int(0.025 * (n_boot - 1))]
    hi = js_samples[int(0.975 * (n_boot - 1))]
    return js0, lo, hi


# ----------------------------
# NEW: "Surface word order model" and "Dependency-consistent linearization model"
# ----------------------------

@dataclass
class SurfaceOrderModel:
    """
    Empirical surface word-order distribution model.
    """
    dist: Dict[str, float]

    @classmethod
    def fit(cls, surface_orders: List[str]) -> "SurfaceOrderModel":
        return cls(dist=compute_distribution(surface_orders))

    def logprob(self, order: str, eps: float = 1e-12) -> float:
        return math.log(max(self.dist.get(order, eps), eps))

    def entropy_bits(self) -> float:
        return calculate_entropy(self.dist)


@dataclass
class DependencyLinearizationModel:
    """
    A model that generates dependency-consistent linearizations from trees.
    This is "a linearization model" in the sense of: tree -> order (POS sequence).
    """
    method: str = "topological_random"
    samples_per_tree: int = 1
    seed: int = 0
    root_id: int = 0

    def generate_orders(self, trees: List[nx.DiGraph]) -> List[str]:
        return get_linearized_orders_from_trees(
            trees,
            method=self.method,
            samples_per_tree=self.samples_per_tree,
            seed=self.seed,
            root_id=self.root_id,
        )

    def distribution(self, trees: List[nx.DiGraph]) -> Dict[str, float]:
        return compute_distribution(self.generate_orders(trees))


# ----------------------------
# OPTIONAL: End-to-end runner (so you DO have the “tree list”)
# ----------------------------

def run_surface_vs_linearized(
    sud_path: str,
    length_filter: Optional[int],
    linearization_method: str,
    samples_per_tree: int,
    seed: int,
    n_boot: int,
) -> Dict[str, Any]:
    """
    Loads SUD trees, extracts surface orders, generates linearized orders,
    then reports distribution stats + JS + bootstrap CI.
    """
    trees = load_sud_trees(sud_path, length_filter=length_filter, root_id=0)
    surface_orders = get_surface_orders_from_trees(trees, root_id=0)

    surface_model = SurfaceOrderModel.fit(surface_orders)
    lin_model = DependencyLinearizationModel(
        method=linearization_method,
        samples_per_tree=samples_per_tree,
        seed=seed,
        root_id=0,
    )
    linear_orders = lin_model.generate_orders(trees)

    P = compute_distribution(surface_orders)
    Q = compute_distribution(linear_orders)

    js = jensen_shannon_divergence(P, Q)
    js0, lo, hi = bootstrap_js_ci(surface_orders, linear_orders, n_boot=n_boot, seed=seed)

    # position-wise KL (only meaningful if fixed length filter)
    pos_kl = {}
    if length_filter is not None:
        posP = compute_positional_distribution(surface_orders)
        posQ = compute_positional_distribution(linear_orders)
        pos_kl = calculate_positional_kl_divergence(posP, posQ)

    return {
        "num_trees": len(trees),
        "length_filter": length_filter,
        "linearization_method": linearization_method,
        "samples_per_tree": samples_per_tree,
        "surface_entropy_bits": surface_model.entropy_bits(),
        "linear_entropy_bits": calculate_entropy(Q),
        "js_divergence_bits": js,
        "js_bootstrap_point": js0,
        "js_bootstrap_ci_low": lo,
        "js_bootstrap_ci_high": hi,
        "positional_kl": pos_kl,
        "surface_dist_size": len(P),
        "linear_dist_size": len(Q),
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--sud", type=str, required=True, help="Path to .conllu/.sud file")
    parser.add_argument("--length", type=int, default=None, help="Filter sentences by exact length")
    parser.add_argument("--lin_method", type=str, default="topological_random",
                        choices=["topological_deterministic", "topological_random", "projective_random"])
    parser.add_argument("--samples_per_tree", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_boot", type=int, default=300)
    args = parser.parse_args()

    out = run_surface_vs_linearized(
        sud_path=args.sud,
        length_filter=args.length,
        linearization_method=args.lin_method,
        samples_per_tree=args.samples_per_tree,
        seed=args.seed,
        n_boot=args.n_boot,
    )
    print(json.dumps(out, indent=2))
