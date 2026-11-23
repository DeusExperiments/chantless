"""Pick a representative K-problem set from the integer-answer subset of DeepMath-103K.

Usage: python mvp/tools/deepmath_select.py QA_PARQUET MODEL_DIR OUT [K] [N_EXAM] [N_SELECT]
  QA_PARQUET: question/final_answer/difficulty/topic columns (shard, row kept for provenance)
  MODEL_DIR:  local EmbeddingGemma snapshot (sentence-transformers format)
Filter: plain integer answer, difficulty >= 3, exact-duplicate questions dropped (this pool is embedded and
cached), then answers 0 and 1 dropped because a broken model can guess them.
Select: embed with the "Clustering" prompt, k-means into K clusters, keep the member nearest each centroid.
Cluster size is kept as `weight`, since medoids give coverage rather than proportional sampling.
Split: the K picks are grouped into K // 10 coarse clusters and whole groups go to one split (`exam`, optional
`select`, and remaining `search`), so close variants stay isolated. Groups are chosen greedily, then improved
by swaps, matching topic-area and difficulty-band shares.
"""
import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.cluster import KMeans

GUESSABLE = ("0", "1")
SEED = 0
MAX_TOKENS = 512
INT_RE = re.compile(r"[-+]?(?:\d+|\d{1,3}(?:,\d{3})+)")
_WRAP_RE = re.compile(r"\\(?:text|textbf|textrm|mathrm|mbox|hbox)\s*\{([^{}]*)\}")


def _parse_int_answer(raw) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    for _ in range(8):
        s = re.sub(r"\\displaystyle\b", "", s).strip("$").strip()
        m_box = re.fullmatch(r"\\boxed\s*\{(.*)\}", s, flags=re.DOTALL)
        if m_box:
            s = m_box.group(1).strip()
            continue
        nxt = _WRAP_RE.sub(r"\1", s).strip()
        if nxt == s:
            nxt = re.sub(r"(?<![A-Za-z])\{([^{}]*)\}", r"\1", s).strip()
        if nxt != s:
            s = nxt
            continue
        break
    for sp in ("\\,", "\\!", "\\;", "\\:"):
        s = s.replace(sp, "")
    s = s.strip("$").strip()
    if not INT_RE.fullmatch(s):
        return None
    return str(int(s.replace(",", "")))


def _parse_args():
    p = argparse.ArgumentParser(description="Select representative integer subset from DeepMath-103K.")
    p.add_argument("qa", type=Path)
    p.add_argument("model", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("pos_k", nargs="?", type=int, default=None)
    p.add_argument("pos_n_exam", nargs="?", type=int, default=None)
    p.add_argument("pos_n_select", nargs="?", type=int, default=None)
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--n-exam", type=int, default=None)
    p.add_argument("--n-select", type=int, default=None)
    a = p.parse_args()
    k = a.k if a.k is not None else (a.pos_k if a.pos_k is not None else 3000)
    n_exam = a.n_exam if a.n_exam is not None else a.pos_n_exam
    n_select = a.n_select if a.n_select is not None else a.pos_n_select
    if n_exam is None and n_select is None:
        n_exam = k // 3
        n_select = k // 3
    elif n_exam is not None and n_select is None:
        n_select = 0
    elif n_exam is None and n_select is not None:
        n_exam = k // 3
    if k < 10:
        raise ValueError(f"K must be >= 10, got {k}")
    if not (0 <= n_exam <= k):
        raise ValueError(f"N_EXAM must be in [0, {k}], got {n_exam}")
    if not (0 <= n_select <= k - n_exam):
        raise ValueError(f"N_SELECT must be in [0, {k - n_exam}], got {n_select}")
    return a.qa, a.model, a.out, k, n_exam, n_select


def _pick_groups(X: np.ndarray, n_areas: int, target_n: int, allowed: np.ndarray, order: np.ndarray, K: int):
    n_groups = X.shape[0]
    m = np.zeros(n_groups, dtype=bool)
    if target_n <= 0 or not allowed.any():
        return m, 0.0
    T = X.sum(0) * target_n / K

    def cost(mask: np.ndarray) -> float:
        c = X[mask].sum(0)
        return float(np.abs(c - T).sum() + 2.0 * abs(c[:n_areas].sum() - target_n))

    avail = [int(h) for h in order if allowed[h]]
    while X[m].sum(0)[:n_areas].sum() < target_n:
        cands = [h for h in avail if not m[h]]
        if not cands:
            break
        best = min(cands, key=lambda h: cost(m | (np.arange(n_groups) == h)))
        m[best] = True

    improved = True
    while improved:
        improved = False
        for a in [h for h in avail if m[h]]:
            for b in [h for h in avail if not m[h]]:
                t = m.copy()
                t[a], t[b] = False, True
                if cost(t) < cost(m) - 1e-9:
                    m, improved = t, True
                    break
            if improved:
                break
    return m, cost(m)


def main():
    QA, MODEL, OUT, K, N_EXAM, N_SELECT = _parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    rows = pq.read_table(QA).to_pylist()
    counts = {"source": len(rows)}
    pool, seen = [], set()
    for r in rows:
        raw_ans = r.get("final_answer") if r.get("final_answer") is not None else r.get("answer")
        ans_norm = _parse_int_answer(raw_ans)
        if ans_norm is None:
            continue
        counts["integer"] = counts.get("integer", 0) + 1
        raw_diff = r.get("difficulty")
        if raw_diff is None or (isinstance(raw_diff, str) and not raw_diff.strip()):
            continue
        if isinstance(raw_diff, bool):
            raise ValueError(f"difficulty must be numeric (not bool), got {raw_diff!r}")
        diff = float(raw_diff)
        if not math.isfinite(diff):
            raise ValueError(f"difficulty must be finite, got {raw_diff!r}")
        if diff < 3:
            continue
        key = " ".join(str(r["question"]).split())
        if key in seen:
            continue
        seen.add(key)
        r = dict(r)
        r["answer"] = ans_norm
        r["difficulty"] = diff
        r["id"] = hashlib.sha1(key.encode()).hexdigest()[:12]
        pool.append(r)
    counts["difficulty_ge_3_dedup"] = len(pool)
    if not pool:
        raise ValueError("deepmath_select: empty pool after filtering")

    emb_path = OUT / "pool-embeddings.npy"
    if emb_path.exists() and np.load(emb_path, mmap_mode="r").shape[0] == len(pool):
        E = np.load(emb_path)
    else:
        import torch
        from sentence_transformers import SentenceTransformer
        dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        m = SentenceTransformer(str(MODEL), device=dev, model_kwargs={"dtype": torch.bfloat16})
        m.max_seq_length = MAX_TOKENS
        order = np.argsort([len(r["question"]) for r in pool])
        t0 = time.time()
        e = m.encode([pool[i]["question"] for i in order], prompt_name="Clustering", batch_size=16,
                     normalize_embeddings=True, show_progress_bar=True)
        E = np.empty_like(e)
        E[order] = e
        np.save(emb_path, E)
        counts["embed_seconds"] = round(time.time() - t0)

    keep = np.array([r["answer"] not in GUESSABLE for r in pool])
    pool, E = [r for r, k in zip(pool, keep) if k], E[keep]
    counts["answer_not_0_or_1"] = len(pool)
    if len(pool) < K:
        raise ValueError(f"filtered pool size {len(pool)} < K={K}")

    km = KMeans(K, n_init=1, max_iter=100, random_state=SEED).fit(E)
    d = np.linalg.norm(E - km.cluster_centers_[km.labels_], axis=1)
    picked, pidx = [], []
    for c in range(K):
        idx = np.flatnonzero(km.labels_ == c)
        i = idx[np.argmin(d[idx])]
        pidx.append(i)
        picked.append({"id": pool[i]["id"], "shard": pool[i]["shard"], "row": pool[i]["row"],
                       "question": pool[i]["question"], "answer": pool[i]["answer"],
                       "difficulty": pool[i]["difficulty"], "topic": pool[i]["topic"],
                       "cluster": c, "weight": int(len(idx))})
    S = E[pidx]

    def area(r):
        t = r.get("topic") or "unknown"
        return t.split(" -> ")[1] if " -> " in t else t
    band = lambda r: "3-4.5" if r["difficulty"] < 5 else "5-6.5" if r["difficulty"] < 7 else "7+"
    n_groups = K // 10
    groups = KMeans(n_groups, n_init=1, max_iter=100, random_state=SEED).fit(S).labels_
    areas = sorted({area(p) for p in picked})
    bands = sorted({band(p) for p in picked})
    n_areas = len(areas)
    area_idx = {a: i for i, a in enumerate(areas)}
    band_idx = {b: n_areas + i for i, b in enumerate(bands)}
    X = np.zeros((n_groups, n_areas + len(bands)))
    for p, g in zip(picked, groups):
        X[g, area_idx[area(p)]] += 1
        X[g, band_idx[band(p)]] += 1

    order = np.random.default_rng(SEED).permutation(n_groups)
    m_exam, exam_cost = _pick_groups(X, n_areas, N_EXAM, np.ones(n_groups, dtype=bool), order, K)
    m_select, select_cost = _pick_groups(X, n_areas, N_SELECT, ~m_exam, order, K)
    exam_groups = set(np.flatnonzero(m_exam).tolist())
    select_groups = set(np.flatnonzero(m_select).tolist())

    for p, g in zip(picked, groups):
        gi = int(g)
        p["group"] = gi
        p["split"] = "exam" if gi in exam_groups else ("select" if gi in select_groups else "search")
    split = np.array([p["split"] for p in picked])

    emit_splits = ["search", "exam"] + (["select"] if (N_SELECT > 0 or (split == "select").any()) else [])
    for name in emit_splits:
        with open(OUT / f"deepmath-int-rep3k-{name}.jsonl", "w") as f:
            for p in sorted((p for p in picked if p["split"] == name), key=lambda p: -p["weight"]):
                f.write(json.dumps(p) + "\n")

    sim = S @ S.T
    np.fill_diagonal(sim, -1)
    cross = sim[np.ix_(split == "exam", split == "search")]

    def dist(rs, key, w=None):
        out = {}
        for j, r in enumerate(rs):
            out[key(r)] = out.get(key(r), 0) + (w[j] if w else 1)
        tot = sum(out.values())
        if tot == 0:
            return {}
        return {k: round(v / tot, 4) for k, v in sorted(out.items(), key=lambda kv: -kv[1])}

    area_rep = {
        "pool": dist(pool, area),
        "set": dist(picked, area),
        "search": dist([p for p in picked if p["split"] == "search"], area),
        "exam": dist([p for p in picked if p["split"] == "exam"], area),
    }
    diff_rep = {
        "pool": dist(pool, band),
        "set": dist(picked, band),
        "search": dist([p for p in picked if p["split"] == "search"], band),
        "exam": dist([p for p in picked if p["split"] == "exam"], band),
    }
    if "select" in emit_splits:
        area_rep["select"] = dist([p for p in picked if p["split"] == "select"], area)
        diff_rep["select"] = dist([p for p in picked if p["split"] == "select"], band)

    report = {
        "dataset": "zwhe99/DeepMath-103K", "dataset_revision": "5cf055d1fe3d7a2eb19719ac020211469736ae44",
        "license": "mit", "qa_parquet_sha256": hashlib.sha256(QA.read_bytes()).hexdigest(),
        "embedding_model": str(MODEL), "prompt": "Clustering", "max_tokens": MAX_TOKENS, "k": K, "seed": SEED,
        "kmeans_inertia": float(km.inertia_), "counts": counts,
        "cluster_size": {"min": int(min(p["weight"] for p in picked)), "median": float(np.median([p["weight"] for p in picked])),
                         "max": int(max(p["weight"] for p in picked))},
        "max_pairwise_cosine_in_set": float(sim.max()), "pairs_cosine_ge_0.95": int((sim >= 0.95).sum() // 2),
        "split": {"n_search": int((split == "search").sum()), "n_select": int((split == "select").sum()),
                  "n_exam": int((split == "exam").sum()),
                  "groups": n_groups, "select_groups": len(select_groups), "exam_groups": len(exam_groups),
                  "balance_cost": float(exam_cost), "select_balance_cost": float(select_cost),
                  "max_cosine_exam_to_search": float(cross.max()) if cross.size else 0.0,
                  "exam_items_with_search_neighbour_ge_0.95": int((cross.max(1) >= 0.95).sum()) if cross.size else 0,
                  "exam_items_with_search_neighbour_ge_0.98": int((cross.max(1) >= 0.98).sum()) if cross.size else 0},
        "area": area_rep,
        "difficulty": diff_rep,
        "answer_0_or_1_in_set": sum(p["answer"] in GUESSABLE for p in picked),
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
