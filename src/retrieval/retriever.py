"""Dense retriever over a prebuilt FAISS index.

Loads an index directory produced by build_index.py (index.faiss + meta.parquet
+ config.json) and answers top-k similarity queries. The query instruction from
the build config (bge-v1.5) is applied here so queries and passages are encoded
consistently with how the index was built.

Used both interactively (CLI below) and as a library by the eval harness.

CLI:
  python src/retrieval/retriever.py --index data/index/bge-small__s256_o50 \
      --k 5 --query "What was Apple's FY2022 capital expenditure?"
"""
import argparse
import json
from pathlib import Path

import faiss
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


def _pick_device() -> str:
    if torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


class Retriever:
    def __init__(self, index_dir: str, device: str | None = None):
        self.dir = Path(index_dir)
        with open(self.dir / 'config.json') as f:
            self.cfg = json.load(f)
        self.index = faiss.read_index(str(self.dir / 'index.faiss'))
        self.meta = pd.read_parquet(self.dir / 'meta.parquet')
        self.query_instruction = self.cfg.get('query_instruction', '')
        self.normalize = self.cfg.get('normalize', True)
        selected_device = device or _pick_device()
        # Python 3.14 / torch 2.13 on this Mac can segfault in the optimized
        # attention/threading path. Eager attention is numerically equivalent
        # for this encoder and single-thread CPU is stable for evaluation.
        if selected_device == 'cpu':
            torch.set_num_threads(1)
        self.model = SentenceTransformer(
            self.cfg['embed_model'],
            device=selected_device,
            model_kwargs={'attn_implementation': 'eager'},
        )

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Return the top-k chunks for a query, each with its similarity score."""
        vec = self.model.encode(
            [self.query_instruction + query],
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        ).astype('float32')
        scores, ids = self.index.search(vec, k)
        hits = []
        for rank, (i, s) in enumerate(zip(ids[0], scores[0]), start=1):
            if i < 0:
                continue
            row = self.meta.iloc[int(i)]
            hits.append({
                'rank': rank,
                'score': float(s),
                'chunk_id': row['chunk_id'],
                'doc_name': row['doc_name'],
                'page_number': int(row['page_number']),
                'text': row['text'],
            })
        return hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--index', required=True, help='index directory from build_index.py')
    ap.add_argument('--query', required=True)
    ap.add_argument('--k', type=int, default=5)
    args = ap.parse_args()

    r = Retriever(args.index)
    print(f"index: {r.cfg['name']}  ({r.index.ntotal:,} chunks, {r.cfg['embed_model']})\n")
    for h in r.search(args.query, args.k):
        print(f"#{h['rank']}  score={h['score']:.3f}  {h['doc_name']} p{h['page_number']}  [{h['chunk_id']}]")
        snippet = ' '.join(h['text'].split())[:280]
        print(f"    {snippet}\n")


if __name__ == '__main__':
    main()
