"""Build a BM25 (lexical) index over a chunk set, as the sparse half of hybrid
retrieval.

Motivation (docs/decisions.md sec.8/sec.10): the dense baseline's dominant
failure is fiscal-year / document disambiguation -- it matches topic but blurs
the literal year token ("2018") across near-duplicate filings. BM25 scores exact
token overlap, so it recovers that signal. Fused with dense via RRF (see
hybrid_retriever.py) this directly targets the bottleneck.

Implementation: a plain Okapi BM25 over an inverted index (no external dep).
Tokenizer emits both the whole alphanumeric token AND its letter/digit sub-runs,
so "FY2018" indexes as {fy2018, fy, 2018} -- this lets a query for the bare year
match table cells that print "2018" and prose that writes "FY2018" alike.

Reads  data/chunks/{name}.parquet
Writes data/index/bm25__{name}/  (bm25.pkl + meta.parquet, chunk-order aligned)

Usage:
  python src/retrieval/build_bm25.py --chunks data/chunks/s256_o50.parquet --name s256_o50
"""
import argparse
import pickle
import re
import time
from array import array
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

INDEX_ROOT = Path('data/index')
META_COLS = ['chunk_id', 'doc_name', 'page_number', 'text']
K1, B = 1.5, 0.75

_ALNUM = re.compile(r'[a-z0-9]+')
_SUBRUN = re.compile(r'[a-z]+|\d+')


def tokenize(text: str) -> list[str]:
    toks: list[str] = []
    for t in _ALNUM.findall(text.lower()):
        toks.append(t)
        if any(c.isalpha() for c in t) and any(c.isdigit() for c in t):
            toks.extend(_SUBRUN.findall(t))
    return toks


def build(chunks_path: str, name: str) -> None:
    df = pd.read_parquet(chunks_path).reset_index(drop=True)
    texts = df['text'].fillna('').tolist()
    n = len(texts)
    print(f'{n:,} chunks -> tokenizing')

    post_ids: dict[str, array] = defaultdict(lambda: array('I'))
    post_tf: dict[str, array] = defaultdict(lambda: array('H'))
    doc_len = np.zeros(n, dtype=np.int32)

    t0 = time.perf_counter()
    for i, text in enumerate(texts):
        tf = Counter(tokenize(text))
        doc_len[i] = sum(tf.values())
        for term, f in tf.items():
            post_ids[term].append(i)
            post_tf[term].append(min(f, 65535))

    avgdl = float(doc_len.mean())
    # idf(t) = ln(1 + (N - df + 0.5)/(df + 0.5))   [BM25+ style, always positive]
    postings = {}
    idf = {}
    for term, ids in post_ids.items():
        df_t = len(ids)
        idf[term] = float(np.log(1 + (n - df_t + 0.5) / (df_t + 0.5)))
        postings[term] = (np.frombuffer(ids, dtype=np.uint32),
                          np.frombuffer(post_tf[term], dtype=np.uint16).astype(np.int32))

    out_dir = INDEX_ROOT / f'bm25__{name}'
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'bm25.pkl', 'wb') as f:
        pickle.dump({'postings': postings, 'idf': idf, 'doc_len': doc_len,
                     'avgdl': avgdl, 'n': n, 'k1': K1, 'b': B, 'name': name}, f)
    df[META_COLS].to_parquet(out_dir / 'meta.parquet', index=False)

    print(f'built BM25: {len(postings):,} unique terms, avgdl={avgdl:.0f}, '
          f'{time.perf_counter()-t0:.0f}s -> {out_dir}')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--chunks', required=True)
    ap.add_argument('--name', required=True, help='chunk-set name, e.g. s256_o50')
    args = ap.parse_args()
    build(args.chunks, args.name)


if __name__ == '__main__':
    main()
