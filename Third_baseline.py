import os
import networkx as nx
import random
import treegen as gen
from Measures import Compute_measures
from Measures_rand import Compute_measures_rand


class Random_base(object):
    def __init__(self, tree):
        self.tree = tree
        self.ls_rand = []
        # Store original node data (like POS tags) for later use
        self.original_nodes = {n: d for n, d in tree.nodes(data=True) if n != 0}
        self.node_ids = sorted(self.original_nodes.keys())
        self.num_nodes = len(self.node_ids)

    def is_similar_distribution(self, randtree, abs_root, weights, similarity_threshold=5.0):
        find = Compute_measures_rand(randtree, abs_root)
        real_measures = Compute_measures(self.tree)

        rand_tree_edges = [e for e in randtree.edges() if e[0] != abs_root]
        real_tree_edges = [e for e in self.tree.edges() if e[0] != 0]
        if not real_tree_edges:
            return True

        # Calculate metrics for the random tree
        random_dd_sample = [find.dependency_distance(edgey) for edgey in rand_tree_edges]
        random_predictability_sample = [find.calculate_predictability(edgey) for edgey in rand_tree_edges]
        random_dd_sample.sort()
        random_predictability_sample.sort()

        # Calculate metrics for the real tree
        real_dd_sample = [real_measures.dependency_distance(edgev) for edgev in real_tree_edges]
        real_predictability_sample = [real_measures.calculate_predictability(edgev) for edgev in real_tree_edges]
        real_dd_sample.sort()
        real_predictability_sample.sort()

        if not real_dd_sample:
            return True

        # Safety if lengths ever diverge (shouldn't, but avoids silent zip truncation)
        m = min(len(real_dd_sample), len(random_dd_sample))
        if m == 0:
            return True

        mse_dd = sum((random_dd_sample[i] - real_dd_sample[i]) ** 2 for i in range(m)) / m

        m2 = min(len(real_predictability_sample), len(random_predictability_sample))
        if m2 == 0:
            return True

        mse_predictability = sum((random_predictability_sample[i] - real_predictability_sample[i]) ** 2 for i in range(m2)) / m2

        combined_score = (weights[0] * mse_dd) + (weights[1] * mse_predictability)
        return combined_score < similarity_threshold

    def rand_tree(self):
        if self.num_nodes < 2:
            return None

        # Use treegen to create a TRULY random directed tree structure
        random_g = gen.random_directed_tree(self.num_nodes)

        treey = nx.DiGraph()
        for i, node_id in enumerate(self.node_ids):
            treey.add_node(node_id, **self.original_nodes[node_id])

        node_map = {i: node_id for i, node_id in enumerate(self.node_ids)}
        for u, v in random_g.edges():
            head, dependent = node_map[u], node_map[v]
            treey.add_edge(head, dependent)

        # Update the 'head' attribute for each node in the new tree
        for node in treey.nodes():
            predecessors = list(treey.predecessors(node))
            treey.nodes[node]['head'] = predecessors[0] if predecessors else 1000  # Abstract root

        return treey

    def gen_random(self, num_to_generate, weights, max_attempts=100000, similarity_threshold=5.0, seed=None):
        self.ls_rand = []
        if seed is not None:
            random.seed(seed)

        for attempt in range(max_attempts):
            if len(self.ls_rand) >= num_to_generate:
                break

            new_tree_structure = self.rand_tree()
            if new_tree_structure is None:
                continue

            abstract_root = 1000
            if abstract_root not in new_tree_structure:
                new_tree_structure.add_node(abstract_root, upostag="ROOT", head=None)

            try:
                real_root = next(nx.topological_sort(new_tree_structure))
                if real_root == abstract_root:
                    real_root = next(n for n in nx.topological_sort(new_tree_structure) if n != abstract_root)

                new_tree_structure.add_edge(abstract_root, real_root)
                new_tree_structure.nodes[real_root]['head'] = abstract_root

            except nx.NetworkXUnfeasible:
                # Handles cases where the random graph has a cycle (rare)
                continue
            except StopIteration:
                continue

            if self.is_similar_distribution(new_tree_structure, abstract_root, weights, similarity_threshold=similarity_threshold):
                self.ls_rand.append(new_tree_structure)
                if len(self.ls_rand) % 50 == 0:
                    print(f"  ... {len(self.ls_rand)} / {num_to_generate} generated ...")

        if len(self.ls_rand) < num_to_generate:
            print(f"\nWarning: Reached max attempts. Generated {len(self.ls_rand)} trees.")

        return self.ls_rand


# Optional helper if you want to quickly get POS-order strings from generated random trees.
# (kept here so you can compare surface vs linearized orders downstream)
def tree_pos_order_topological(tree, root_id=0):
    return "".join([tree.nodes[n].get("upostag", "X")[0] for n in nx.topological_sort(tree) if n != root_id and n != 1000])


def tree_pos_order_surface_by_id(tree, root_id=0):
    nodes = [n for n in tree.nodes() if n not in (root_id, 1000) and isinstance(n, int)]
    nodes.sort()
    return "".join([tree.nodes[n].get("upostag", "X")[0] for n in nodes])
