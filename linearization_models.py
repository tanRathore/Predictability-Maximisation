from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable
from collections import defaultdict
import math
import random
import networkx as nx


ROOT_ID_DEFAULT = 0


def _sorted_token_nodes(tree: nx.DiGraph, root_id: int = ROOT_ID_DEFAULT) -> List[int]:
    nodes = [n for n in tree.nodes() if n != root_id]
    nodes.sort()
    return nodes


def surface_pos_order(tree: nx.DiGraph, root_id: int = ROOT_ID_DEFAULT) -> str:
    nodes = _sorted_token_nodes(tree, root_id=root_id)
    return "".join(tree.nodes[n].get("upostag", "X")[0] for n in nodes)


def _children(tree: nx.DiGraph, head: int) -> List[int]:
    return list(tree.successors(head))


def _heads(tree: nx.DiGraph, root_id: int = ROOT_ID_DEFAULT) -> Dict[int, int]:
    h = {}
    for n in tree.nodes():
        if n == root_id:
            continue
        preds = list(tree.predecessors(n))
        h[n] = preds[0] if preds else root_id
    return h


def dependency_length_total(order: List[int], head_map: Dict[int, int], root_id: int = ROOT_ID_DEFAULT) -> int:
    pos = {tok: i for i, tok in enumerate(order)}
    total = 0
    for dep, head in head_map.items():
        if dep == root_id or head == root_id:
            continue
        total += abs(pos[dep] - pos[head])
    return total


def interveners_predictability_loss(order: List[int], head_map: Dict[int, int], root_id: int = ROOT_ID_DEFAULT) -> int:
    """
    Proxy loss: number of tokens strictly between head and dependent, summed over dependencies.
    Lower is "more predictable" under your current heuristic.
    """
    pos = {tok: i for i, tok in enumerate(order)}
    loss = 0
    for dep, head in head_map.items():
        if dep == root_id or head == root_id:
            continue
        loss += max(0, abs(pos[dep] - pos[head]) - 1)
    return loss


@dataclass
class LinearizationResult:
    order: List[int]
    objective: float
    dl: int
    pred_loss: int


def _postorder(tree: nx.DiGraph, root: int) -> List[int]:
    out = []
    seen = set()

    def dfs(u: int):
        seen.add(u)
        for v in tree.successors(u):
            if v not in seen:
                dfs(v)
        out.append(u)

    dfs(root)
    return out


def _choose_root(tree: nx.DiGraph, root_id: int = ROOT_ID_DEFAULT) -> int:
    """
    Finds the token root (node whose head is root_id) if present; else smallest node.
    """
    for n in tree.nodes():
        if n == root_id:
            continue
        preds = list(tree.predecessors(n))
        if not preds or preds[0] == root_id:
            return n
    nodes = _sorted_token_nodes(tree, root_id=root_id)
    return nodes[0] if nodes else root_id


def random_projective_like_linearization(
    tree: nx.DiGraph,
    root_id: int = ROOT_ID_DEFAULT,
    rng: Optional[random.Random] = None,
) -> List[int]:
    """
    Produces a tree-consistent linearization by recursively ordering children.
    Not guaranteed projective, but tends to be "tree-respecting" and stable.
    """
    rng = rng or random.Random()
    root = _choose_root(tree, root_id=root_id)

    def linearize(head: int) -> List[int]:
        kids = _children(tree, head)
        rng.shuffle(kids)
        left = []
        right = []
        for k in kids:
            # randomly assign child subtree left or right of head
            (left if rng.random() < 0.5 else right).append(k)
        seq = []
        for k in left:
            seq.extend(linearize(k))
        seq.append(head)
        for k in right:
            seq.extend(linearize(k))
        return seq

    return linearize(root)


def dlm_greedy_linearization(tree: nx.DiGraph, root_id: int = ROOT_ID_DEFAULT) -> List[int]:
    """
    Deterministic-ish greedy: place each child subtree on the side that (locally) reduces DL,
    using token index heuristic (smaller index -> left preference).
    """
    root = _choose_root(tree, root_id=root_id)

    def linearize(head: int) -> List[int]:
        kids = _children(tree, head)
        # heuristic: children with lower token id tend left, higher tend right
        left_kids = sorted([k for k in kids if k < head])
        right_kids = sorted([k for k in kids if k > head])
        seq = []
        for k in left_kids:
            seq.extend(linearize(k))
        seq.append(head)
        for k in right_kids:
            seq.extend(linearize(k))
        return seq

    return linearize(root)


def objective(order: List[int], head_map: Dict[int, int], alpha: float, beta: float, root_id: int = ROOT_ID_DEFAULT) -> Tuple[float, int, int]:
    dl = dependency_length_total(order, head_map, root_id=root_id)
    pred = interveners_predictability_loss(order, head_map, root_id=root_id)
    # both are losses to minimize
    return alpha * dl + beta * pred, dl, pred


def local_swap_optimize(
    tree: nx.DiGraph,
    init_order: List[int],
    alpha: float = 1.0,
    beta: float = 1.0,
    max_iters: int = 2000,
    rng: Optional[random.Random] = None,
    root_id: int = ROOT_ID_DEFAULT,
) -> LinearizationResult:
    """
    Simple hill-climb: randomly swap adjacent tokens if it improves objective.
    Keeps dependency structure fixed; only changes linearization.
    """
    rng = rng or random.Random()
    head_map = _heads(tree, root_id=root_id)

    order = init_order[:]
    best_obj, best_dl, best_pred = objective(order, head_map, alpha, beta, root_id=root_id)

    n = len(order)
    if n <= 2:
        return LinearizationResult(order=order, objective=best_obj, dl=best_dl, pred_loss=best_pred)

    for _ in range(max_iters):
        i = rng.randrange(0, n - 1)
        cand = order[:]
        cand[i], cand[i + 1] = cand[i + 1], cand[i]
        cand_obj, cand_dl, cand_pred = objective(cand, head_map, alpha, beta, root_id=root_id)
        if cand_obj < best_obj:
            order = cand
            best_obj, best_dl, best_pred = cand_obj, cand_dl, cand_pred

    return LinearizationResult(order=order, objective=best_obj, dl=best_dl, pred_loss=best_pred)


def linearize_tree(
    tree: nx.DiGraph,
    mode: str = "surface",
    alpha: float = 1.0,
    beta: float = 1.0,
    optimize: bool = True,
    max_iters: int = 2000,
    seed: int = 0,
    root_id: int = ROOT_ID_DEFAULT,
) -> LinearizationResult:
    """
    mode:
      - "surface": use token index order (gold surface)
      - "random": random tree-respecting linearization
      - "dlm": greedy DLM-ish linearization
      - "joint": start from random and optimize alpha*DL + beta*PredLoss
      - "pred": start from random and optimize beta*PredLoss (alpha=0)
    """
    rng = random.Random(seed)
    head_map = _heads(tree, root_id=root_id)

    if mode == "surface":
        init = _sorted_token_nodes(tree, root_id=root_id)
        obj, dl, pred = objective(init, head_map, alpha, beta, root_id=root_id)
        return LinearizationResult(order=init, objective=obj, dl=dl, pred_loss=pred)

    if mode == "random":
        init = random_projective_like_linearization(tree, root_id=root_id, rng=rng)
        obj, dl, pred = objective(init, head_map, alpha, beta, root_id=root_id)
        return LinearizationResult(order=init, objective=obj, dl=dl, pred_loss=pred)

    if mode == "dlm":
        init = dlm_greedy_linearization(tree, root_id=root_id)
        if not optimize:
            obj, dl, pred = objective(init, head_map, alpha, beta, root_id=root_id)
            return LinearizationResult(order=init, objective=obj, dl=dl, pred_loss=pred)
        return local_swap_optimize(tree, init, alpha=1.0, beta=0.0, max_iters=max_iters, rng=rng, root_id=root_id)

    if mode == "pred":
        init = random_projective_like_linearization(tree, root_id=root_id, rng=rng)
        if not optimize:
            obj, dl, pred = objective(init, head_map, alpha=0.0, beta=1.0, root_id=root_id)
            return LinearizationResult(order=init, objective=obj, dl=dl, pred_loss=pred)
        return local_swap_optimize(tree, init, alpha=0.0, beta=1.0, max_iters=max_iters, rng=rng, root_id=root_id)

    if mode == "joint":
        init = random_projective_like_linearization(tree, root_id=root_id, rng=rng)
        if not optimize:
            obj, dl, pred = objective(init, head_map, alpha=alpha, beta=beta, root_id=root_id)
            return LinearizationResult(order=init, objective=obj, dl=dl, pred_loss=pred)
        return local_swap_optimize(tree, init, alpha=alpha, beta=beta, max_iters=max_iters, rng=rng, root_id=root_id)

    raise ValueError(f"Unknown mode: {mode}")


def order_to_pos_string(tree: nx.DiGraph, order: List[int]) -> str:
    return "".join(tree.nodes[n].get("upostag", "X")[0] for n in order)
