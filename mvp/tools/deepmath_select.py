"""Pick a representative K-problem set from the integer-answer subset of DeepMath-103K.

Usage: python mvp/tools/deepmath_select.py QA_PARQUET MODEL_DIR OUT [K] [N_EXAM]
  QA_PARQUET: question/final_answer/difficulty/topic columns (shard, row kept for provenance)
  MODEL_DIR:  local EmbeddingGemma snapshot (sentence-transformers format)
Filter: plain integer answer, difficulty >= 3, exact-duplicate questions dropped (this pool is embedded and
cached), then answers 0 and 1 dropped because a broken model can guess them.
Select: embed with the "Clustering" prompt, k-means into K clusters, keep the member nearest each centroid.
Cluster size is kept as `weight`, since medoids give coverage rather than proportional sampling.
Split: the K picks are grouped into K // 10 coarse clusters and whole groups go to one side, so close variants
of an exam problem stay out of the search set. Groups are chosen greedily, then improved by swaps, so the exam
holds about N_EXAM problems with the same topic-area and difficulty-band shares as the whole set.
"""
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.cluster import KMeans

QA, MODEL, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
K = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
N_EXAM = int(sys.argv[5]) if len(sys.argv) > 5 else 1000
GUESSABLE = ("0", "1")
SEED = 0
MAX_TOKENS = 512
INT = re.compile(r"-?\d+")
OUT.mkdir(parents=True, exist_ok=True)

rows = pq.read_table(QA).to_pylist()
counts = {"source": len(rows)}
pool, seen = [], set()
for r in rows:
    a = r["final_answer"].strip().strip("$").strip()
    if not INT.fullmatch(a):
        continue
    counts["integer"] = counts.get("integer", 0) + 1
    if r["difficulty"] < 3:
        continue
    key = " ".join(r["question"].split())
    if key in seen:
        continue
    seen.add(key)
    r["answer"] = str(int(a))
    r["id"] = hashlib.sha1(key.encode()).hexdigest()[:12]
    pool.append(r)
counts["difficulty_ge_3_dedup"] = len(pool)

emb_path = OUT / "pool-embeddings.npy"
if emb_path.exists() and np.load(emb_path, mmap_mode="r").shape[0] == len(pool):
    E = np.load(emb_path)
else:
    import torch
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    m = SentenceTransformer(str(MODEL), device=dev, model_kwargs={"dtype": torch.bfloat16})
    m.max_seq_length = MAX_TOKENS  # cap embedding input length to bound memory
    order = np.argsort([len(r["question"]) for r in pool])  # length-sorted batches pad less
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

area = lambda r: r["topic"].split(" -> ")[1] if " -> " in r["topic"] else r["topic"]
band = lambda r: "3-4.5" if r["difficulty"] < 5 else "5-6.5" if r["difficulty"] < 7 else "7+"
groups = KMeans(K // 10, n_init=1, max_iter=100, random_state=SEED).fit(S).labels_
cells = sorted({area(p) for p in picked}) + sorted({band(p) for p in picked})
X = np.zeros((K // 10, len(cells)))  # per group: count in each area, then in each band
for p, g in zip(picked, groups):
    X[g, cells.index(area(p))] += 1
    X[g, cells.index(band(p))] += 1
T = X.sum(0) * N_EXAM / K  # exam counts the whole set's shares imply


def cost(m):
    c = X[m].sum(0)
    return np.abs(c - T).sum() + 2 * abs(c[:len(cells) - 3].sum() - N_EXAM)


m = np.zeros(K // 10, bool)
order = np.random.default_rng(SEED).permutation(K // 10)  # seeded order breaks ties
while X[m].sum(0)[:len(cells) - 3].sum() < N_EXAM:
    best = min((h for h in order if not m[h]), key=lambda h: cost(m | (np.arange(K // 10) == h)))
    m[best] = True
improved = True
while improved:
    improved = False
    for a in (h for h in order if m[h]):
        for b in (h for h in order if not m[h]):
            t = m.copy(); t[a], t[b] = False, True
            if cost(t) < cost(m) - 1e-9:
                m, improved = t, True
                break
        if improved:
            break
exam_groups = set(np.flatnonzero(m).tolist())
for p, g in zip(picked, groups):
    p["group"], p["split"] = int(g), "exam" if int(g) in exam_groups else "search"
split = np.array([p["split"] for p in picked])
for name in ("search", "exam"):
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
    return {k: round(v / tot, 4) for k, v in sorted(out.items(), key=lambda kv: -kv[1])}


report = {
    "dataset": "zwhe99/DeepMath-103K", "dataset_revision": "5cf055d1fe3d7a2eb19719ac020211469736ae44",
    "license": "mit", "qa_parquet_sha256": hashlib.sha256(QA.read_bytes()).hexdigest(),
    "embedding_model": str(MODEL), "prompt": "Clustering", "max_tokens": MAX_TOKENS, "k": K, "seed": SEED,
    "kmeans_inertia": float(km.inertia_), "counts": counts,
    "cluster_size": {"min": int(min(p["weight"] for p in picked)), "median": float(np.median([p["weight"] for p in picked])),
                     "max": int(max(p["weight"] for p in picked))},
    "max_pairwise_cosine_in_set": float(sim.max()), "pairs_cosine_ge_0.95": int((sim >= 0.95).sum() // 2),
    "split": {"n_search": int((split == "search").sum()), "n_exam": int((split == "exam").sum()),
              "groups": K // 10, "exam_groups": len(exam_groups), "balance_cost": float(cost(m)),
              "max_cosine_exam_to_search": float(cross.max()),
              "exam_items_with_search_neighbour_ge_0.95": int((cross.max(1) >= 0.95).sum()),
              "exam_items_with_search_neighbour_ge_0.98": int((cross.max(1) >= 0.98).sum())},
    "area": {"pool": dist(pool, area), "set": dist(picked, area), "search": dist([p for p in picked if p["split"] == "search"], area),
             "exam": dist([p for p in picked if p["split"] == "exam"], area)},
    "difficulty": {"pool": dist(pool, band), "set": dist(picked, band), "search": dist([p for p in picked if p["split"] == "search"], band),
                   "exam": dist([p for p in picked if p["split"] == "exam"], band)},
    "answer_0_or_1_in_set": sum(p["answer"] in GUESSABLE for p in picked),
}
(OUT / "report.json").write_text(json.dumps(report, indent=1))
print(json.dumps(report, indent=1))
