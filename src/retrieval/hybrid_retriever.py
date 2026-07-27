"""Hybrid retriever: dense (bge-small FAISS) + BM25, fused with RRF.

Reciprocal Rank Fusion combines the two rankings by rank, not score, so no
score normalization is needed and one retriever can't dominate by scale:
    rrf(chunk) = sum_r 1 / (c + rank_r(chunk))          (c = 60)
Each retriever contributes its top-`pool` candidates; the fused top-k is
returned. Same .search()/.cfg interface as Retriever, so eval_retrieval.py and
run_rag.py can use it unchanged.

CLI:
  python src/retrieval/hybrid_retriever.py --dense data/index/bge-small__s256_o50 \
      --bm25 data/index/bm25__s256_o50 --k 5 --query "..."
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from retriever import Retriever
from build_bm25 import tokenize


class BM25:
    def __init__(self, index_dir: str):
        d = Path(index_dir)
        with open(d / 'bm25.pkl', 'rb') as f:
            s = pickle.load(f)
        self.postings, self.idf = s['postings'], s['idf']
        self.doc_len = s['doc_len'].astype(np.float32)
        self.avgdl, self.k1, self.b = s['avgdl'], s['k1'], s['b']
        self.meta = pd.read_parquet(d / 'meta.parquet')

    def search(self, query: str, k: int) -> list[dict]:
        scores = np.zeros(len(self.doc_len), dtype=np.float32)
        for term in set(tokenize(query)):
            post = self.postings.get(term)
            if post is None:
                continue
            ids, tf = post
            dl = self.doc_len[ids]
            denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            scores[ids] += self.idf[term] * (tf * (self.k1 + 1)) / denom
        if not scores.any():
            return []
        top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        top = top[np.argsort(-scores[top])]
        out = []
        for rank, i in enumerate(top, 1):
            if scores[i] <= 0:
                break
            r = self.meta.iloc[int(i)]
            out.append({'rank': rank, 'score': float(scores[i]), 'chunk_id': r['chunk_id'],
                        'doc_name': r['doc_name'], 'page_number': int(r['page_number']),
                        'text': r['text']})
        return out


class HybridRetriever:
    def __init__(self, dense_dir: str, bm25_dir: str, pool: int = 100,
                 rrf_c: int = 60, dense_weight: float = 0.5,
                 device: str | None = None):
        if not 0.0 <= dense_weight <= 1.0:
            raise ValueError('dense_weight must be between 0 and 1')
        self.dense = Retriever(dense_dir, device=device)
        self.bm25 = BM25(bm25_dir)
        self.pool, self.rrf_c = pool, rrf_c
        self.dense_weight = dense_weight
        weight_tag = f'd{dense_weight:.2f}'.replace('.', 'p')
        self.cfg = {'name': f"hybrid-{weight_tag}__{self.dense.cfg['name']}",
                    'embed_model': self.dense.cfg['embed_model'],
                    'dense': self.dense.cfg['name'], 'pool': pool,
                    'rrf_c': rrf_c, 'dense_weight': dense_weight}
        self.ntotal = self.dense.index.ntotal

    def search(self, query: str, k: int = 5) -> list[dict]:
        dense_hits = self.dense.search(query, self.pool)
        bm25_hits = self.bm25.search(query, self.pool)
        fused: dict[str, dict] = {}
        for hits, weight in ((dense_hits, self.dense_weight),
                             (bm25_hits, 1.0 - self.dense_weight)):
            if weight == 0:
                continue
            for h in hits:
                cid = h['chunk_id']
                slot = fused.setdefault(cid, {'hit': h, 'rrf': 0.0})
                slot['rrf'] += weight / (self.rrf_c + h['rank'])
                # keep the chunk record (identical across retrievers)
                slot['hit'] = h
        ranked = sorted(fused.values(), key=lambda s: s['rrf'], reverse=True)[:k]
        return [{**s['hit'], 'rank': i, 'score': s['rrf']}
                for i, s in enumerate(ranked, 1)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dense', required=True)
    ap.add_argument('--bm25', required=True)
    ap.add_argument('--query', required=True)
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--dense-weight', type=float, default=0.5)
    args = ap.parse_args()
    r = HybridRetriever(args.dense, args.bm25, dense_weight=args.dense_weight)
    print(f"hybrid: {r.cfg['name']}  ({r.ntotal:,} chunks)\n")
    for h in r.search(args.query, args.k):
        print(f"#{h['rank']}  rrf={h['score']:.4f}  {h['doc_name']} p{h['page_number']}")
        print(f"    {' '.join(h['text'].split())[:200]}\n")


if __name__ == '__main__':
    main()
