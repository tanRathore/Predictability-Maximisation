import random
import numpy as np
import networkx as nx
from conllu import parse

# ==========================================
# METRICS (Reusing your definitions)
# ==========================================
def get_dep_len(tree, order):
    pos_map = {node_id: i for i, node_id in enumerate(order)}
    total_dl = 0
    for node in tree.nodes:
        if node == 0: continue
        head = tree.nodes[node].get('head')
        if head is not None and head in pos_map and node in pos_map:
            total_dl += abs(pos_map[head] - pos_map[node])
    return total_dl

def get_intervener_complexity(tree, order):
    pos_map = {node_id: i for i, node_id in enumerate(order)}
    total_ic = 0
    for node in tree.nodes:
        if node == 0: continue
        head = tree.nodes[node].get('head')
        if head is not None and head in pos_map and node in pos_map:
            p1, p2 = sorted((pos_map[node], pos_map[head]))
            for candidate in tree.nodes:
                if candidate in [0, node, head]: continue
                if candidate in pos_map:
                    if p1 < pos_map[candidate] < p2:
                        # Check if sibling
                        if tree.nodes[candidate].get('head') == head:
                            total_ic += 1
    return total_ic

def linearize_random(tree):
    nodes = [n for n in tree.nodes if n != 0]
    random.shuffle(nodes)
    return nodes

def get_pos_string(tree, order):
    # Extracts the POS tag string (e.g., "NVN")
    return "".join([tree.nodes[n]['upostag'][0] for n in order])

# ==========================================
# ABLATION LOGIC
# ==========================================
def run_ablation(file_path, lang_name):
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()
    sentences = parse(data)
    
    # Pick a few sample sentences of length 10-12
    samples = [s for s in sentences if 10 <= len(s) <= 12][:3] 
    
    print(f"\n=== ABLATION TEST: {lang_name} ===")
    
    for i, sent in enumerate(samples):
        # Build Graph
        G = nx.DiGraph()
        G.add_node(0)
        tokens = [t for t in sent if isinstance(t['id'], int)]
        for tok in tokens:
            G.add_node(tok['id'], head=tok['head'], upostag=tok['upostag'])
            if tok['head'] is not None:
                G.add_edge(tok['head'], tok['id'])
        
        # 1. Get Real Stats
        real_order = sorted([n for n in G.nodes if n != 0])
        real_dl = get_dep_len(G, real_order)
        real_ic = get_intervener_complexity(G, real_order)
        real_str = get_pos_string(G, real_order)
        
        # 2. Generate 500 Random Baselines
        candidates = []
        for _ in range(500):
            r_ord = linearize_random(G)
            r_dl = get_dep_len(G, r_ord)
            r_ic = get_intervener_complexity(G, r_ord)
            r_str = get_pos_string(G, r_ord)
            candidates.append({'order': r_ord, 'dl': r_dl, 'ic': r_ic, 'str': r_str})
            
        # 3. Find Champions
        # Sort by DL to find DL-Only Optimizer
        dl_champion = sorted(candidates, key=lambda x: x['dl'])[0]
        
        # Sort by IC to find Predictability-Only Optimizer
        pred_champion = sorted(candidates, key=lambda x: x['ic'])[0]
        
        print(f"\nSentence {i+1} (Length {len(tokens)}):")
        print(f"  REAL ORDER:   {real_str} | DL: {real_dl} | IC: {real_ic}")
        print(f"  OPTIMAL DL:   {dl_champion['str']} | DL: {dl_champion['dl']} | IC: {dl_champion['ic']}")
        print(f"  OPTIMAL PRED: {pred_champion['str']} | DL: {pred_champion['dl']} | IC: {pred_champion['ic']}")
        
        # 4. The "Similarity" Check
        if abs(real_ic - pred_champion['ic']) < abs(real_ic - dl_champion['ic']):
            print("  -> RESULT: Real sentence is closer to PREDICTABILITY Optimized.")
        elif abs(real_dl - dl_champion['dl']) < abs(real_dl - pred_champion['dl']):
            print("  -> RESULT: Real sentence is closer to DL Optimized.")
        else:
             print("  -> RESULT: Balanced / Mixed strategy.")

# Run for English and Hindi
# Adjust paths if needed
run_ablation("./SUD/en-sud-train.conllu", "English")
run_ablation("./SUD/hi_sud-train.conllu", "Hindi")