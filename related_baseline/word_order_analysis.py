from collections import defaultdict
import math
import numpy as np
from conllu import parse
import networkx as nx

def extract_orders_from_sud(file_path):
    """
    Extracts word orders from a SUD file.
    Each sentence is represented by the sequence of the first letters of the 'upostag' field.
    
    :param file_path: Path to the SUD file.
    :return: List of word order strings (e.g., ["NNNV", "NNVN", ...]).
    """
    orders = []
    with open(file_path, "r", encoding="utf-8") as f:
        sentences = parse(f.read())
    for sentence in sentences:
        pos_sequence = "".join([word['upostag'][0] for word in sentence if word.get('upostag')])
        orders.append(pos_sequence)
    return orders

def get_order_from_tree(tree):
    """
    Converts a dependency tree into a word order string.
    Assumes that each node (except the abstract root) has a 'upostag' field.
    
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