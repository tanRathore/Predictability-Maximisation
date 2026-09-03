# Packages used: NetworkX -- for tree representation as directed acyclic graphs

import networkx as nx

class Compute_measures_rand(object):
    def __init__(self, tree, root):
        self.tree=tree                          # tree encodes the nodes and edges content in dictionary format. It uses directed graph (DiGraph) feature of networkX package. For example, nodes are encoded like this - tree.nodes={1:{form:'this',POS:'PRN'},2:{...}}   
        self.root=root                             # ROOT is an abstract node in the tree and is encoded as empty and with name=0    
        
    def dependency_direction(self, edge):       # Computes the direction of an edge (i.e., dependency) according to relative position of dependent and head  
        direction=''
        if edge[0]>edge[1]:                     # edge is a list data type in the format [head,dependent]
            direction='RL'
        else:
            direction='LR'

        return direction                        # return direction as 'LR' (Left-to-Right) or RL ('Right-to-Left')  
            
    def dependency_distance(self, edge):        # Computes the dependency length i.e., no. of nodes between head and its dependent 
        dd=0
        if edge[0]>edge[1]:                      
            for nodex in nx.descendants(self.tree, self.root):        
                if edge[1]<nodex<edge[0]:                             # all the nodes that lies linearly between dependent and head   
                    dd+=1
        else:
            for nodex in nx.descendants(self.tree, self.root):
                if edge[0]<nodex<edge[1]:
                    dd+=1
        return dd                              # returns the dependency distance of the edge
        
    def dependency_depth(self, edge):        # Computes the dependency length i.e., no. of nodes between head and its dependent 
        hd=1
        if edge[0]>edge[1]:                      
            for nodex in nx.descendants(self.tree, self.root):        
                if edge[1]<nodex<edge[0]:                             # all the nodes that lies linearly between dependent and head
                    if nx.descendants(self.tree, nodex):
                        hd+=1
        else:
            for nodex in nx.descendants(self.tree, self.root):
                if edge[0]<nodex<edge[1]:
                    if nx.descendants(self.tree, nodex):
                        hd+=1
        return hd                              # returns the dependency distance of the edge

    def calculate_predictability(self, edge):
        head, dependent = edge
        interveners = 0
        for node in nx.descendants(self.tree, self.root):
            if dependent < head:
                if tree.nodes[node]['head'] == head and node > dependent and node < head:
                    interveners += 1
            else:
                if tree.nodes[node]['head'] == dependent and node > head and node < dependent:
                    interveners += 1
        return interveners
    
    def is_projective(self, edge):             # Checks if an edge is projective or not and returns a boolean value.
        projective=True
        edge_span=[]                 
        if edge[0]>edge[1]:                                      
            for nodex in nx.descendants(self.tree, self.root):   
                if edge[1]<nodex<edge[0]:
                    edge_span.append(nodex)                       
        else:
            for nodex in nx.descendants(self.tree, self.root):
                if edge[0]<nodex<edge[1]:
                    edge_span.append(nodex)

        flag=0
        for nodeI in edge_span:
            if not self.tree.nodes[nodeI]['head'] in edge_span:
                if not nodeI in nx.descendants(self.tree, edge[0]):
                    flag += 1
        if not flag==0:
            projective=False
        return projective                                                       # Returns TRUE is edge is projective otherwise FALSE

    def edge_degree(self, edge):                                 # Computes the number of edges causing non-projectivity              
        eD=0
        edge_span=[]                 
        if edge[0]>edge[1]:                                      
            for nodex in nx.descendants(self.tree, self.root):   
                if edge[1]<nodex<edge[0]:
                    edge_span.append(nodex)                       
        else:
            for nodex in nx.descendants(self.tree, self.root):
                if edge[0]<nodex<edge[1]:
                    edge_span.append(nodex)

        for nodeI in edge_span:
            if not self.tree.nodes[nodeI]['head'] in edge_span:         # if the head of any intervening node exists outside the span of the edge
                if not nodeI in nx.descendants(self.tree, edge[0]):
                    eD += 1    
        return eD                                                        

    def gap_degree(self, node):                     # Computes the gaps in the projection chain containing maximum number of gaps 
        chains_gapD=[]
        terminals=[]
        for nodex in self.tree.nodes:
            if self.tree.out_degree(nodex)==0:
                terminals.append(nodex)

        for nodeT in terminals:                       
            gapD=0
            if nx.has_path(self.tree, node, nodeT):     
                pathx=nx.all_simple_paths(self.tree, node, nodeT, cutoff=None)     # Projection chain from ROOT to each Terminal node is encoded as list of nodes in the chain   
                for item in pathx:
                    pathy=item
                    for nodeP in item:
                        if not nodeP==self.root:
                            if not self.tree.nodes[nodeP]['head']==self.root:
                                if not self.is_projective([self.tree.nodes[nodeP]['head'],nodeP]):   # If any edge in a projection chain is non-projective 
                                    gapD=gapD+1                                                      # No. of non-projective edges in a projection chain = No. of gap degree
            chains_gapD.append(gapD)

        gap_deg=max(chains_gapD)                                                             # Gap degree of a node is the maximum gap degree in its dependents
        return gap_deg

    def gapnodes(self, pathx):                                          # Returns all the nodes which causes GAP in the projection of a node
        nodeM=[]
        for nodex in pathx:
            if not nodex==self.root:
                if not self.tree.nodes[nodex]['head']==self.root:
                    if not self.is_projective([self.tree.nodes[nodex]['head'],nodex]):
                        cross_dep=nodex
                        cross_head=self.tree.nodes[cross_dep]['head']   
                        edge_span=[]                 
                        if cross_head>cross_dep:                                      
                            for nodev in nx.descendants(self.tree, self.root):   
                                if cross_dep<nodev<cross_head:
                                    edge_span.append(nodev)                       
                        else:
                            for nodev in nx.descendants(self.tree, self.root):
                                if cross_head<nodev<cross_dep:
                                    edge_span.append(nodev)
                        for nodeI in edge_span:
                            if not self.tree.nodes[nodeI]['head'] in edge_span:         # if the head of any intervening node exists outside the span of the edge
                                if not nodeI in nx.descendants(self.tree, cross_head):
                                    nodeM.append(nodeI)
        return nodeM

    def illnestedness(self,node,gapD):
        k_illnest=0
        if gapD==0:
            k_illnest=0
        else:
            illnest = []
            all_gapped_chains=[]
            for nodex in nx.descendants(self.tree, self.root):
                if not nodex==self.root:
                    if not self.tree.nodes[nodex]['head']==self.root:
                        if not self.is_projective([self.tree.nodes[nodex]['head'],nodex]):
                            if nx.has_path(self.tree, node, nodex):
                                pathx=nx.all_simple_paths(self.tree, node, nodex, cutoff=None)
                                for item in pathx:
                                    all_gapped_chains.append(item)                             # It has all the projection chains from root to dependents of crossing arcs
                                                                                          # i.e., all possible chains which can interleave with other                  
            chains_with_gaps=[]
            for chainx in all_gapped_chains:
                flag=0
                for chainy in all_gapped_chains:
                    if set(chainx) < set(chainy):
                        flag=flag+1
                if flag==0:
                    chains_with_gaps.append(chainx)
                    
            for pathz in chains_with_gaps:
                #print(pathz)
                num_interL=0                                                        # Variable for number of chains that can interleave with a single chain in question
                nodeM=self.gapnodes(pathz)
                for pathy in chains_with_gaps:
                    if not pathy==pathz:
                        flag=0
                        for nodeC in nodeM:
                            if nodeC in pathy:
                                flag=flag+1
                        if not flag==0:
                            nodeMM=self.gapnodes(pathy)
                            flagg=0
                            for nodeCC in nodeMM:
                                if nodeCC in pathz:
                                    flagg=flagg+1
                            if not flagg==0:
                                num_interL = num_interL + 1
                illnest.append(num_interL)
            k_illnest = max(illnest)
        return k_illnest

    def gapD_hist(self):
        gapd_histogram={}
        for nodex in self.tree.nodes:
            gapD=self.gap_degree(nodex)
            if gapD in gapd_histogram.keys():
                gapd_histogram[gapD]=gapd_histogram[gapD]+1
            else:
                gapd_histogram[gapD]=1
        return gapd_histogram 

    def projection_degree(self, node):
        size_chains=[]
        terminals=[]
        for nodex in self.tree.nodes:
            if self.tree.out_degree(nodex)==0:
                terminals.append(nodex)

        for nodeT in terminals:
            size=0
            if nx.has_path(self.tree, node, nodeT):
                pathx=nx.all_simple_paths(self.tree, node, nodeT, cutoff=None)              # Projection chain from ROOT to each Terminal node is encoded as list of nodes in the chain
                for item in pathx:  
                    size=len(item)-1
            size_chains.append(size)
        proj_degree=max(size_chains)                                    # Projection degree = No. of nodes in the longest projection chain from ROOT to a terminal node 
        return proj_degree

    def projD_hist(self):
        projd_histogram={}
        for nodex in self.tree.nodes:
            projD=self.projection_degree(nodex)
            if projD in projd_histogram.keys():
                projd_histogram[projD]=projd_histogram[projD]+1
            else:
                projd_histogram[projD]=1
        return projd_histogram 

    def arity(self):                                                    # Computes arity of the tree using out-degree of nodes 
        tree_arity=self.tree.out_degree(list(self.tree.nodes))          # returns a dictionary containing nodenames as keys and its out-degree (or arity) as its values
        max_arity=max([x[1] for x in tree_arity])                       # Maximum out-degree = maximum arity in the tree
        avg_arity=sum([x[1] for x in tree_arity])/len([x[1] for x in tree_arity])
        histogram={}
        for arityx in [x[1] for x in tree_arity]:
            if arityx in histogram.keys():
                histogram[arityx]=histogram[arityx]+1                   # Creates arity histogram i.e. frequency of each arity
            else:
                histogram[arityx]=1
        arity_histogram=histogram                                       
        return [max_arity, avg_arity, tree_arity, arity_histogram]                    

    def endpoint_crossing(self,edge):
        edge_span=[]
        if edge[0]>edge[1]:
            for nodex in nx.descendants(self.tree, self.root):
                if edge[1]<nodex<edge[0]:
                    edge_span.append(nodex)
        else:
            for nodex in nx.descendants(self.tree, self.root):
                if edge[0]<nodex<edge[1]:
                    edge_span.append(nodex)

        endpoint={}
        
        for nodeI in edge_span:
            if not self.tree.nodes[nodeI]['head'] in edge_span:       # nodes intervening the edge span which are not dominated by the any node in the edge span
                if not nodeI in nx.descendants(self.tree, edge[0]):
                    endpoint[self.tree.nodes[nodeI]['head']]=1        # creates a dictionary of all nodes having their outside their span. This dictionary has keys as 'heads' of the intervening nodes
                            
        endpoint_cross=len(endpoint)                                   # If the intervening nodes belongs to more than head outside the edge span, then 1-endpoint crossing constraint is voilated
        return endpoint_cross                                          # returns the no. of heads outside the edge span which dominates the intervening nodes   

    def compute_all(self):
        Arity=self.arity()
        Projection_degree=self.projection_degree()
        Gap_degree=self.gap_degree()
        direction={}
        dep_distane={}
        projectivity={}
        Edge_degree={}
        endpoint_cross={}
        for edgex in self.tree.edges:
            direction[edgex]=self.dependency_direction(edgex)
            dep_distance[edgex]=self.dependency_distance(edgex)
            projectivity[edgex]=self.is_projective(edgex)
            Edge_degree[edgex]=self.edge_degree(edgex)
            endpoint_cross[edgex]=self.endpoint_crossing(edgex)

        return [Arity, Projection_degree, Gap_degree, direction, dep_distance, projectivity, Edge_degree, endpoint_cross]

    def all_dependent_constraint(self,edge):
        all_dep_deg=0
        edge_span=[]
        if edge[0]>edge[1]:
            for nodex in nx.descendants(self.tree, self.root):
                if edge[1]<nodex<edge[0]:
                    edge_span.append(nodex)                          
                    
        else:
            for nodex in nx.descendants(self.tree, self.root):
                if edge[0]<nodex<edge[1]:
                    edge_span.append(nodex)

        if not self.is_projective(edge):
            for nodey in edge_span:
                if not (edge[0] in nx.ancestors(self.tree, nodey)):                                                                 # if a node 'x' occurring the edge span is not dominated by the head of the edge
                    int_node=nodey
                    if not int_node==1000:
                        for edgex in self.tree.edges:
                            if edgex[1]==int_node:
                                int_head=edgex[0]
                        dep_int=0
                        for nodeI in edge_span:
                            if self.tree.nodes[nodeI]['head']==int_head:
                                dep_int=dep_int+1
                            else:
                                dep_int=dep_int
                        all_dep=0
                        for nodeJ in nx.descendants(self.tree, self.root):
                            if self.tree.nodes[nodeJ]['head']==int_head:
                                all_dep=all_dep+1
                            else:
                                all_dep=all_dep
                        all_dep_deg=all_dep-dep_int
                    else:
                        all_dep_deg=0            
        else:
            all_dep_deg=100

        return all_dep_deg
            
    def hdd(self,edge):
        HDD=0
        edge_span=[]
        if edge[0]>edge[1]:
            for nodex in nx.descendants(self.tree, self.root):
                if edge[1]<nodex<edge[0]:
                    edge_span.append(nodex)                          
                    
        else:
            for nodex in nx.descendants(self.tree, self.root):
                if edge[0]<nodex<edge[1]:
                    edge_span.append(nodex)

        if not self.is_projective(edge):
            for nodeI in edge_span:
                if not self.tree.nodes[nodeI]['head'] in edge_span:       # nodes intervening the edge span which are not dominated by the any node in the edge span
                    if not nodeI in nx.descendants(self.tree, edge[0]):
                        int_node=nodeI
                        if not int_node==self.root:
                            for edgex in self.tree.edges:
                                if edgex[1]==int_node:
                                    int_head=edgex[0]
                            if nx.has_path(self.tree, int_head, edge[0]):
                                pathx=nx.all_simple_paths(self.tree, int_head, edge[0], cutoff=None)
                                for item in pathx:
                                    HDD=len(item)-1
                            elif nx.has_path(self.tree, edge[0], int_head):
                                pathx=nx.all_simple_paths(self.tree, edge[0], int_head, cutoff=None)
                                for item in pathx:
                                    HDD=len(item)-1
                            else:
                                HDD=1
                        else:
                            HDD=2
                                    
        return HDD
                
