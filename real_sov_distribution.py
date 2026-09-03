import os
import csv
import random
import argparse
from collections import Counter, defaultdict

from conllu import parse


PATHS = {
    "English": "./SUD/en-sud-train.conllu",
    "Hindi": "./SUD/hi_sud-train.conllu",
    "Japanese": "./SUD/ja_sud-train.conllu",
}

NOMINAL_UPOS = {"NOUN", "PROPN", "PRON"}
VERBAL_UPOS = {"VERB", "AUX"}


def is_projective_under_order(tokens, order_ids):
    pos = {tid: i for i, tid in enumerate(order_ids)}
    arcs = []
    for t in tokens:
        tid = t["id"]
        if not isinstance(tid, int):
            continue
        head = t.get("head", None)
        if head is None or head == 0:
            continue
        if head not in pos or tid not in pos:
            continue
        i, j = pos[head], pos[tid]
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        arcs.append((a, b))

    for x in range(len(arcs)):
        a, b = arcs[x]
        for y in range(x + 1, len(arcs)):
            c, d = arcs[y]
            if a < c < b < d or c < a < d < b:
                return False
    return True


def pick_predicate_heads(tokens):
    """
    Robust predicate selection:
    - Prefer root token (head==0) that is VERB/AUX
    - Else root token (head==0) regardless of UPOS
    - Else first VERB/AUX in sentence
    Then add "predicate chain" neighbors (root<->child verb/aux) to allow AUX-root cases.
    """
    by_id = {t["id"]: t for t in tokens if isinstance(t.get("id"), int)}

    root = None
    for t in tokens:
        if not isinstance(t.get("id"), int):
            continue
        if t.get("head") == 0 and t.get("upostag") in VERBAL_UPOS:
            root = t
            break

    if root is None:
        for t in tokens:
            if not isinstance(t.get("id"), int):
                continue
            if t.get("head") == 0:
                root = t
                break

    if root is None:
        for t in tokens:
            if not isinstance(t.get("id"), int):
                continue
            if t.get("upostag") in VERBAL_UPOS:
                root = t
                break

    if root is None:
        return set()

    root_id = root["id"]

    verb_ids = {t["id"] for t in tokens if isinstance(t.get("id"), int) and t.get("upostag") in VERBAL_UPOS}
    heads = {root_id}

    # include immediate verb/aux neighbors in the predicate chain (handles AUX-root vs VERB-root)
    for vid in list(verb_ids):
        h = by_id.get(vid, {}).get("head")
        if h == root_id:
            heads.add(vid)
        if vid == by_id.get(root_id, {}).get("head"):
            heads.add(vid)

    return heads


def is_subject_deprel(rel):
    if not rel:
        return False
    rel = rel.lower()
    # UD: nsubj, csubj, nsubj:pass, etc.
    # SUD often still contains "subj" patterns; safest is substring.
    return "subj" in rel


def is_object_deprel(rel):
    if not rel:
        return False
    rel = rel.lower()
    # UD: obj, iobj, etc.
    # SUD variants often include "obj" inside comp:obj / etc.
    return "obj" in rel


def extract_SOV_pattern(tokens, only_projective=False):
    toks = [t for t in tokens if isinstance(t.get("id"), int)]
    if not toks:
        return None, "no_tokens"

    order = sorted([t["id"] for t in toks])  # surface order by ID

    if only_projective and (not is_projective_under_order(toks, order)):
        return None, "non_projective"

    by_id = {t["id"]: t for t in toks}
    pos = {tid: i for i, tid in enumerate(order)}

    predicate_heads = pick_predicate_heads(toks)
    if not predicate_heads:
        return None, "no_predicate"

    # pick a single V position for S/O/V ordering:
    # choose the earliest predicate head in surface order (stable choice)
    v_id = min(predicate_heads, key=lambda tid: pos.get(tid, 10**9))
    v_pos = pos[v_id]

    subj_cands = []
    obj_cands = []

    for t in toks:
        tid = t["id"]
        head = t.get("head")
        rel = t.get("deprel", "")

        if head not in predicate_heads:
            continue

        upos = t.get("upostag")
        if upos not in NOMINAL_UPOS:
            continue

        if is_subject_deprel(rel):
            subj_cands.append(tid)
        if is_object_deprel(rel):
            obj_cands.append(tid)

    if not subj_cands:
        return None, "no_subject"
    if not obj_cands:
        return None, "no_object"

    # choose leftmost S and leftmost O in surface order (common typology heuristic)
    s_id = min(subj_cands, key=lambda tid: pos[tid])
    o_id = min(obj_cands, key=lambda tid: pos[tid])

    s_pos = pos[s_id]
    o_pos = pos[o_id]

    # determine the permutation of S,O,V by their positions
    items = [("S", s_pos), ("O", o_pos), ("V", v_pos)]
    items.sort(key=lambda x: x[1])
    pattern = "".join([x[0] for x in items])

    return pattern, "ok"


def load_sentences(file_path, length_filter, max_sentences, seed):
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        sents = parse(f.read())

    # filter by length (count integer ids only)
    filtered = []
    for sent in sents:
        toks = [t for t in sent if isinstance(t.get("id"), int)]
        if len(toks) != length_filter:
            continue
        filtered.append(sent)

    if len(filtered) > max_sentences:
        random.seed(seed)
        filtered = random.sample(filtered, max_sentences)

    return filtered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=12)
    ap.add_argument("--only_projective", action="store_true")
    ap.add_argument("--max_sentences", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="real_sov_distribution.csv")
    ap.add_argument("--details", default=None, help="Optional: write per-sentence details CSV.")
    ap.add_argument("--debug_label_counts", action="store_true", help="Print top deprel counts for debugging.")
    args = ap.parse_args()

    rows = []
    detail_rows = []
    patterns = ["SOV", "SVO", "VSO", "VOS", "OSV", "OVS"]

    for lang, path in PATHS.items():
        sents = load_sentences(path, args.length, args.max_sentences, args.seed)

        counts = Counter()
        reasons = Counter()
        used = 0
        kept = 0

        deprel_counter = Counter()

        for i, sent in enumerate(sents):
            toks = [t for t in sent if isinstance(t.get("id"), int)]
            used += 1

            if args.debug_label_counts:
                for t in toks:
                    rel = t.get("deprel")
                    if rel:
                        deprel_counter[rel] += 1

            pat, reason = extract_SOV_pattern(toks, only_projective=args.only_projective)
            reasons[reason] += 1

            if pat is None:
                continue

            if pat in patterns:
                counts[pat] += 1
                kept += 1

            if args.details is not None:
                detail_rows.append(
                    {
                        "language": lang,
                        "sent_index": i,
                        "length": args.length,
                        "only_projective": bool(args.only_projective),
                        "pattern": pat,
                    }
                )

        if args.debug_label_counts:
            top = deprel_counter.most_common(15)
            print(f"\n{lang} top deprel labels (top 15):")
            for k, v in top:
                print(f"  {k:20s} {v}")

        print(
            f"{lang} | used={used} kept(S+O+V)={kept} | "
            + " ".join([f"{p}:{counts[p]}" for p in patterns])
        )
        print(f"{lang} | skip_reasons: " + ", ".join([f"{k}={v}" for k, v in reasons.most_common()]))

        for p in patterns:
            rows.append(
                {
                    "language": lang,
                    "length": args.length,
                    "only_projective": bool(args.only_projective),
                    "n_used_after_filters": used,
                    "n_kept_with_S_and_O": kept,
                    "pattern": p,
                    "count": counts[p],
                    "rate": (counts[p] / kept) if kept > 0 else float("nan"),
                }
            )

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "language",
                "length",
                "only_projective",
                "n_used_after_filters",
                "n_kept_with_S_and_O",
                "pattern",
                "count",
                "rate",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    if args.details is not None:
        with open(args.details, "w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["language", "sent_index", "length", "only_projective", "pattern"],
            )
            w.writeheader()
            w.writerows(detail_rows)

    print(f"\n[SAVED] {args.out}")
    if args.details is not None:
        print(f"[SAVED] {args.details}")


if __name__ == "__main__":
    main()
