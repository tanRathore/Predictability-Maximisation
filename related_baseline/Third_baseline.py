import os
from io import open
import networkx as nx
from Measures_rand import *
from Measures import *
import random
import treegen as gen
import depgraph as dep

class Random_base(object):
    def __init__(self, tree):
        self.tree = tree
        self.ls_rand = []

    def num_cross_rand(self, randtree, abs_root):
        comput = Compute_measures_rand(randtree, abs_root)
        ncross_random = 0
        for edgex in randtree.edges:
            if not edgex[0] == abs_root:
                if comput.is_projective(edgex):
                    ncross_random += 0
                else:
                    ncross_random += 1
        return ncross_random

    def is_equal_num_crossings(self, randtree, abs_root, num_cross_real):
        flag = False
        num_cross_random = self.num_cross_rand(randtree, abs_root)
        if num_cross_random == num_cross_real:
            flag = True
        return flag

    def is_same_tree(self, randtree, abs_root):
        rand_tree = nx.DiGraph()
        for edgex in randtree.edges:
            if not edgex[0] == abs_root:
                rand_tree.add_edge(edgex[0], edgex[1])
        real_tree = nx.DiGraph()
        for edgez in self.tree.edges:
            if not edgez[0] == 0:
                real_tree.add_edge(edgez[0], edgez[1])
        mapping_real = dict(zip(real_tree.nodes(), range(1, len(real_tree.nodes()) + 1)))
        mapping_rand = dict(zip(rand_tree.nodes(), range(1, len(rand_tree.nodes()) + 1)))
        REC_real = nx.relabel_nodes(real_tree, mapping_real)
        REC_rand = nx.relabel_nodes(rand_tree, mapping_rand)
        return REC_real.edges == REC_rand.edges

    def calculate_predictability(self, tree, edge):
        head, dependent = edge
        interveners = 0
        for node in tree.nodes:
            if tree.nodes[node]['head'] == head and node < dependent:
                interveners += 1
        return interveners

    # Modified: function signature now accepts a weights parameter
    def is_similar_distribution(self, randtree, abs_root, weights):
        find = Compute_measures_rand(randtree, abs_root)
        rand_tree = nx.DiGraph()
        for edgex in randtree.edges:
            if not edgex[0] == abs_root:
                rand_tree.add_edge(edgex[0], edgex[1])
       
        random_dd_sample = []
        random_predictability_sample = []
        for edgey in rand_tree.edges:
            random_dd_sample.append(find.dependency_distance(edgey))
            random_predictability_sample.append(find.predictability(edgey))
        random_dd_sample.sort()
        random_predictability_sample.sort()
       
        get = Compute_measures(self.tree)
        real_tree = nx.DiGraph()
        for edgez in self.tree.edges:
            if not edgez[0] == 0:
                real_tree.add_edge(edgez[0], edgez[1])
        real_dd_sample = []
        real_predictability_sample = []
        for edgev in real_tree.edges:
            real_dd_sample.append(get.dependency_distance(edgev))
            real_predictability_sample.append(get.predictability(edgev))
        real_dd_sample.sort()
        real_predictability_sample.sort()

        mse_dd = sum((rd - rr)**2 for rd, rr in zip(random_dd_sample, real_dd_sample)) / len(real_dd_sample)
        mse_predictability = sum((rp - rrp)**2 for rp, rrp in zip(random_predictability_sample, real_predictability_sample)) / len(real_predictability_sample)
       
        # Use provided weights instead of fixed values
        combined_score = (weights[0] * mse_dd) + (weights[1] * mse_predictability)
        return combined_score < 0.5  # Adjust threshold as necessary
   
    # Modified: added weights parameter (default value provided)
    def rand_tree(self, num_cross_real, weights=(0.65, 0.35)):
        real_tree = nx.DiGraph()
        for edgez in self.tree.edges:
            if not edgez[0] == 0:
                real_tree.add_edge(edgez[0], edgez[1])
        edge_list = list(real_tree.edges())
        node_list = list(real_tree.nodes())
        random.shuffle(edge_list)
        random.shuffle(node_list)

        treex = nx.DiGraph()
        treex.add_nodes_from(node_list)
        for nodex in treex.nodes:
            if self.tree.has_node(self.tree.nodes[nodex]['head']):
                if not self.tree.nodes[nodex]['head'] == 0:
                    treex.add_edge(self.tree.nodes[nodex]['head'], nodex)
       
        mapping = dict(zip(treex.nodes(), range(1, len(treex.nodes()) + 1)))
        treey = nx.relabel_nodes(treex, mapping)
        abstract_root = 1000
        real_root = next(nx.topological_sort(treey))
        treey.add_edge(abstract_root, real_root)
        for edgex in treey.edges:
            treey.nodes[edgex[1]]['head'] = edgex[0]

        if self.is_equal_num_crossings(treey, abstract_root, num_cross_real):
            # Pass the weights parameter here
            if self.is_similar_distribution(treey, abstract_root, weights):
                if not self.is_same_tree(treey, abstract_root):
                    self.ls_rand.append(treey)

    # Modified: added weights parameter (default value provided)
    def gen_random(self, num_cross_real, weights=(0.65, 0.35)):
        n = len(self.tree.edges)
        rand_out = []
        if n < 30:
            x = 0
            while len(self.ls_rand) == 0 and x < 30000:
                x += 1
                self.rand_tree(num_cross_real, weights)
                rand_out = self.ls_rand
        return rand_out
