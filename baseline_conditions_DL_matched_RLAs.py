# Using NetworkX package and conllu package
# This is baseline conditions module; more conditions can be added as functions
import os
from io import open
import networkx as nx
from operator import itemgetter
import random
from Measures_rand import *
from Measures import *
import treegen as gen
import depgraph as dep

class Random_base(object):
    def __init__(self, tree):                   # tree has an abstract node=0 and real nodes =1,2,... 
        self.tree=tree                          # tree encodes the nodes and edges content in dictionary format. It uses directed graph (DiGraph) feature of networkX package. For example, nodes are encoded like this - tree.nodes={1:{form:'this',POS:'PRN'},2:{...}}   
        self.ls_rand=[]

##    def crossings_in(self,tree):
##        for edge in tree.edges():
##            n1, n2 = sorted(edge)
##            for edge_ in tree.edges():
##                n1_, n2_ = sorted(edge_)
##                if not (n2_ <= n1 or n2 <= n1_ or (n1 <= n1_ and n2_ <= n2) or (n1_ <= n1 and n2 <= n2_)):
##                    yield frozenset({edge, edge_})

    def num_cross_rand(self,randtree,abs_root):               # requires random tree and its abstract root=10000
        comput=Compute_measures_rand(randtree,abs_root)
        ncross_random=0
        for edgex in randtree.edges:
            if not edgex[0]==abs_root: 
                if comput.is_projective(edgex):                   # checks if edge is projective or not
                    ncross_random += 0
                else:
                    ncross_random += 1
        return ncross_random                                  # returns number of crossings in the random tree 

    def is_equal_num_crossings(self,randtree,abs_root,num_cross_real):   # requires random tree, its abstract root and numCrossings in real tree  
        flag=False
        num_cross_random=self.num_cross_rand(randtree,abs_root)       
        if num_cross_random==num_cross_real:                      # checks if number of crossings are equal in real and random tree
            flag=True
        return flag

    def is_same_tree(self,randtree,abs_root):
        rand_tree=nx.DiGraph()
        for edgex in randtree.edges:
            if not edgex[0]==abs_root:
                rand_tree.add_edge(edgex[0],edgex[1])   # regenrates a dummy random tree by removing its abstract root

        real_tree = nx.DiGraph()
        for edgez in self.tree.edges:
            if not edgez[0]==0:
                real_tree.add_edge(edgez[0],edgez[1])   # regenrates a dummy real tree by removing its abstract root
        mapping_real=dict(zip(real_tree.nodes(),range(1,len(real_tree.nodes)+1)))
        mapping_rand=dict(zip(rand_tree.nodes(),range(1,len(rand_tree.nodes)+1)))
        REC_real=nx.relabel_nodes(real_tree,mapping_real)
        REC_rand=nx.relabel_nodes(rand_tree,mapping_rand)
        return REC_real.edges==REC_rand.edges


    def is_similar_DD_distribution(self,randtree,abs_root):
        find=Compute_measures_rand(randtree,abs_root)
        rand_tree=nx.DiGraph()
        for edgex in randtree.edges:
            if not edgex[0]==abs_root:
                rand_tree.add_edge(edgex[0],edgex[1])   # regenrates a dummy random tree by removing its abstract root
        
        random_dd_sample=[]
        for edgey in rand_tree.edges:
            random_dd_sample.append(find.dependency_distance(edgey))
        random_dd_sample.sort()
        
        get=Compute_measures(self.tree)
        real_tree = nx.DiGraph()
        for edgez in self.tree.edges:
            if not edgez[0]==0:
                real_tree.add_edge(edgez[0],edgez[1])   # regenrates a dummy real tree by removing its abstract root
        real_dd_sample=[]
        for edgev in real_tree.edges:
            real_dd_sample.append(get.dependency_distance(edgev))
        real_dd_sample.sort()
        return random_dd_sample==real_dd_sample
    
    def rand_tree(self,num_cross_real):                 # requires number of crossings from the real tree
        real_tree = nx.DiGraph()
        for edgez in self.tree.edges:
            if not edgez[0]==0:
                real_tree.add_edge(edgez[0],edgez[1])   # regenrates a dummy real tree by removing its abstract root
        edge_list=list(real_tree.edges())
        node_list=list(real_tree.nodes())               
        random.shuffle(edge_list)
        random.shuffle(node_list)                      # shuffles the ordering of edges of the dummy real tree 

        treex=nx.DiGraph()                              # generates an empty random tree 
        treex.add_nodes_from(node_list)

        for nodex in treex.nodes:
            if self.tree.has_node(self.tree.nodes[nodex]['head']):                                        # to handle disjoint trees
                if not self.tree.nodes[nodex]['head']==0:
                    treex.add_edge(self.tree.nodes[nodex]['head'],nodex)       # adds edges as relation between nodes
        
        mapping=dict(zip(treex.nodes(),range(1,len(treex.nodes)+1)))
        treey=nx.relabel_nodes(treex,mapping) 

        abstract_root=1000
        real_root=next(nx.topological_sort(treey))
        treey.add_edge(abstract_root,real_root)         # adds an abstract root to the random tree 
        for edgex in treey.edges:
            treey.nodes[edgex[1]]['head']=edgex[0]

        if self.is_equal_num_crossings(treey,abstract_root,num_cross_real):  # matches the no. of crossings in the real and random tree
            if self.is_similar_DD_distribution(treey,abstract_root):
                if not self.is_same_tree(treey,abstract_root):
                    self.ls_rand.append(treey)                  # adds the random tree to the list

    def gen_random(self,num_cross_real):              # requires numbr of crossings from the real tree
        n = len(self.tree.edges)
        rand_out=[]
        if n<30:
            x=0
            while len(self.ls_rand)==0 and x<30000:     # checks if list of random trees is ampty and limits the generating attempts 
                x=x+1
                self.rand_tree(num_cross_real)
                rand_out=self.ls_rand
        return rand_out                           # returns the list of random trees
