import networkx as nx
 
class Compute_measures(object):
    def __init__(self, tree):
        self.tree = tree
        self.root = 0

    def _orient_edge(self, edge):
        a, b = edge

        if a == self.root and b != self.root:
            return a, b
        if b == self.root and a != self.root:
            return b, a

        try:
            if self.tree.nodes[b].get("head", None) == a:
                return a, b
            if self.tree.nodes[a].get("head", None) == b:
                return b, a
        except KeyError:
            pass

        if self.tree.has_edge(a, b):
            return a, b
        if self.tree.has_edge(b, a):
            return b, a

        return a, b

    def dependency_direction(self, edge):
        head, dep = self._orient_edge(edge)
        return "RL" if head > dep else "LR"

    def dependency_distance(self, edge):
        head, dep = self._orient_edge(edge)
        lo, hi = (dep, head) if dep < head else (head, dep)

        dd = 0
        for nodex in nx.descendants(self.tree, self.root):
            if lo < nodex < hi:
                dd += 1
        return dd

    def dependency_depth(self, edge):
        head, dep = self._orient_edge(edge)
        lo, hi = (dep, head) if dep < head else (head, dep)

        hd = 1
        for nodex in nx.descendants(self.tree, self.root):
            if lo < nodex < hi:
                if nx.descendants(self.tree, nodex):
                    hd += 1
        return hd

    def calculate_predictability(self, edge):
        head, dep = self._orient_edge(edge)
        lo, hi = (dep, head) if dep < head else (head, dep)

        interveners = 0
        for node, data in self.tree.nodes(data=True):
            if node == self.root:
                continue
            if lo < node < hi and data.get("head", None) == head:
                interveners += 1
        return interveners

    def calculate_predictability_linear(self, edge):
        head, dep = self._orient_edge(edge)
        return max(0, abs(head - dep) - 1)

    def calculate_predictability_both(self, edge):
        return {
            "sibling_interveners": self.calculate_predictability(edge),
            "linear_interveners": self.calculate_predictability_linear(edge),
        }

    def is_projective(self, edge):
        head, dep = self._orient_edge(edge)
        projective = True
        edge_span = []

        lo, hi = (dep, head) if dep < head else (head, dep)
        for nodex in nx.descendants(self.tree, self.root):
            if lo < nodex < hi:
                edge_span.append(nodex)

        flag = 0
        for nodeI in edge_span:
            if self.tree.nodes[nodeI]["head"] not in edge_span:
                if nodeI not in nx.descendants(self.tree, head):
                    if self.tree.nodes[nodeI].get("deprel", None) != "punct":
                        flag += 1

        if flag != 0:
            projective = False
        return projective

    def edge_degree(self, edge):
        head, dep = self._orient_edge(edge)
        eD = 0
        edge_span = []

        lo, hi = (dep, head) if dep < head else (head, dep)
        for nodex in nx.descendants(self.tree, self.root):
            if lo < nodex < hi:
                edge_span.append(nodex)

        for nodeI in edge_span:
            if self.tree.nodes[nodeI]["head"] not in edge_span:
                if nodeI not in nx.descendants(self.tree, head):
                    if self.tree.nodes[nodeI].get("deprel", None) != "punct":
                        eD += 1
        return eD

    def gap_degree(self, node):
        chains_gapD = []
        terminals = []
        for nodex in self.tree.nodes:
            if self.tree.out_degree(nodex) == 0:
                terminals.append(nodex)

        for nodeT in terminals:
            gapD = 0
            if nx.has_path(self.tree, node, nodeT):
                pathx = nx.all_simple_paths(self.tree, node, nodeT, cutoff=None)
                for item in pathx:
                    for nodeP in item:
                        if nodeP != self.root:
                            if self.tree.nodes[nodeP]["head"] != self.root:
                                if not self.is_projective([self.tree.nodes[nodeP]["head"], nodeP]):
                                    gapD += 1
            chains_gapD.append(gapD)

        return max(chains_gapD) if chains_gapD else 0

    def gapnodes(self, pathx):
        nodeM = []
        for nodex in pathx:
            if nodex != self.root:
                if self.tree.nodes[nodex]["head"] != self.root:
                    if not self.is_projective([self.tree.nodes[nodex]["head"], nodex]):
                        cross_dep = nodex
                        cross_head = self.tree.nodes[cross_dep]["head"]
                        edge_span = []

                        lo, hi = (cross_dep, cross_head) if cross_dep < cross_head else (cross_head, cross_dep)
                        for nodev in nx.descendants(self.tree, self.root):
                            if lo < nodev < hi:
                                edge_span.append(nodev)

                        for nodeI in edge_span:
                            if self.tree.nodes[nodeI]["head"] not in edge_span:
                                if nodeI not in nx.descendants(self.tree, cross_head):
                                    if self.tree.nodes[nodeI].get("deprel", None) != "punct":
                                        nodeM.append(nodeI)
        return nodeM

    def illnestedness(self, node, gapD):
        if gapD == 0:
            return 0

        illnest = []
        all_gapped_chains = []

        for nodex in nx.descendants(self.tree, self.root):
            if nodex != self.root:
                if self.tree.nodes[nodex]["head"] != self.root:
                    if not self.is_projective([self.tree.nodes[nodex]["head"], nodex]):
                        if nx.has_path(self.tree, node, nodex):
                            pathx = nx.all_simple_paths(self.tree, node, nodex, cutoff=None)
                            for item in pathx:
                                all_gapped_chains.append(item)

        chains_with_gaps = []
        for chainx in all_gapped_chains:
            flag = 0
            for chainy in all_gapped_chains:
                if set(chainx) < set(chainy):
                    flag += 1
            if flag == 0:
                chains_with_gaps.append(chainx)

        for pathz in chains_with_gaps:
            num_interL = 0
            nodeM = self.gapnodes(pathz)
            for pathy in chains_with_gaps:
                if pathy != pathz:
                    flag = 0
                    for nodeC in nodeM:
                        if nodeC in pathy:
                            flag += 1
                    if flag != 0:
                        nodeMM = self.gapnodes(pathy)
                        flagg = 0
                        for nodeCC in nodeMM:
                            if nodeCC in pathz:
                                flagg += 1
                        if flagg != 0:
                            num_interL += 1
            illnest.append(num_interL)

        return max(illnest) if illnest else 0

    def gapD_hist(self):
        gapd_histogram = {}
        for nodex in self.tree.nodes:
            gapD = self.gap_degree(nodex)
            gapd_histogram[gapD] = gapd_histogram.get(gapD, 0) + 1
        return gapd_histogram

    def projection_degree(self, node):
        size_chains = []
        terminals = []
        for nodex in self.tree.nodes:
            if self.tree.out_degree(nodex) == 0:
                terminals.append(nodex)

        for nodeT in terminals:
            size = 0
            if nx.has_path(self.tree, node, nodeT):
                pathx = nx.all_simple_paths(self.tree, node, nodeT, cutoff=None)
                for item in pathx:
                    size = len(item) - 1
            size_chains.append(size)

        return max(size_chains) if size_chains else 0

    def projD_hist(self):
        projd_histogram = {}
        for nodex in self.tree.nodes:
            projD = self.projection_degree(nodex)
            projd_histogram[projD] = projd_histogram.get(projD, 0) + 1
        return projd_histogram

    def arity(self):
        tree_arity = self.tree.out_degree(list(self.tree.nodes))
        vals = [x[1] for x in tree_arity]
        max_arity = max(vals) if vals else 0
        avg_arity = (sum(vals) / len(vals)) if vals else 0.0

        histogram = {}
        for arityx in vals:
            histogram[arityx] = histogram.get(arityx, 0) + 1

        return [max_arity, avg_arity, tree_arity, histogram]

    def endpoint_crossing(self, edge):
        head, dep = self._orient_edge(edge)
        edge_span = []

        lo, hi = (dep, head) if dep < head else (head, dep)
        for nodex in nx.descendants(self.tree, self.root):
            if lo < nodex < hi:
                edge_span.append(nodex)

        endpoint = {}
        for nodeI in edge_span:
            if self.tree.nodes[nodeI]["head"] not in edge_span:
                if nodeI not in nx.descendants(self.tree, head):
                    if self.tree.nodes[nodeI].get("deprel", None) != "punct":
                        endpoint[self.tree.nodes[nodeI]["head"]] = 1

        return len(endpoint)

    def compute_all(self):
        Arity = self.arity()

        Projection_degree = {}
        Gap_degree = {}
        for nodex in self.tree.nodes:
            if nodex == self.root:
                continue
            Projection_degree[nodex] = self.projection_degree(nodex)
            Gap_degree[nodex] = self.gap_degree(nodex)

        direction = {}
        dep_distance = {}
        projectivity = {}
        Edge_degree = {}
        endpoint_cross = {}

        for edgex in self.tree.edges:
            direction[edgex] = self.dependency_direction(edgex)
            dep_distance[edgex] = self.dependency_distance(edgex)
            projectivity[edgex] = self.is_projective(edgex)
            Edge_degree[edgex] = self.edge_degree(edgex)
            endpoint_cross[edgex] = self.endpoint_crossing(edgex)

        return [Arity, Projection_degree, Gap_degree, direction, dep_distance, projectivity, Edge_degree, endpoint_cross]

    def all_dependent_constraint(self, edge):
        head, dep = self._orient_edge(edge)
        all_dep_deg = 0
        edge_span = []

        lo, hi = (dep, head) if dep < head else (head, dep)
        for nodex in nx.descendants(self.tree, self.root):
            if lo < nodex < hi:
                edge_span.append(nodex)

        if not self.is_projective([head, dep]):
            for nodey in edge_span:
                if head not in nx.ancestors(self.tree, nodey):
                    if self.tree.nodes[nodey].get("deprel", None) != "punct":
                        int_node = nodey
                        if int_node != self.root:
                            int_head = self.tree.nodes[int_node].get("head", None)

                            dep_int = 0
                            for nodeI in edge_span:
                                if self.tree.nodes[nodeI].get("head", None) == int_head:
                                    dep_int += 1

                            all_dep = 0
                            for nodeJ in nx.descendants(self.tree, self.root):
                                if self.tree.nodes[nodeJ].get("head", None) == int_head:
                                    all_dep += 1

                            all_dep_deg = all_dep - dep_int
                        else:
                            all_dep_deg = 0
        else:
            all_dep_deg = 100

        return all_dep_deg

    def hdd(self, edge):
        head, dep = self._orient_edge(edge)
        HDD = 0
        edge_span = []

        lo, hi = (dep, head) if dep < head else (head, dep)
        for nodex in nx.descendants(self.tree, self.root):
            if lo < nodex < hi:
                edge_span.append(nodex)

        if not self.is_projective([head, dep]):
            for nodeI in edge_span:
                if self.tree.nodes[nodeI]["head"] not in edge_span:
                    if nodeI not in nx.descendants(self.tree, head):
                        if self.tree.nodes[nodeI].get("deprel", None) != "punct":
                            int_node = nodeI
                            if int_node != self.root:
                                int_head = self.tree.nodes[int_node].get("head", None)

                                if int_head is None:
                                    HDD = 1
                                elif nx.has_path(self.tree, int_head, head):
                                    for item in nx.all_simple_paths(self.tree, int_head, head, cutoff=None):
                                        HDD = len(item) - 1
                                elif nx.has_path(self.tree, head, int_head):
                                    for item in nx.all_simple_paths(self.tree, head, int_head, cutoff=None):
                                        HDD = len(item) - 1
                                else:
                                    HDD = 1
                            else:
                                HDD = 2

        return HDD