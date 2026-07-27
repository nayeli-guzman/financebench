"""Evaluate BM25/dense RRF weights from one retrieval pass.

Loads both retrievers once, collects each one's top-100 candidates for every
FinanceBench question, then evaluates several fusion weights without repeating
model inference. A dense weight of 0 is BM25-only; 1 is dense-only.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'retrieval'))
from hybrid_retriever import HybridRetriever  # noqa: E402

EVAL_JSONL = Path('data/financebench/financebench_open_source.jsonl')
RUNS = Path('runs')
K_LIST = [1, 3, 5, 10, 20]
PAGE_OFFSET = 1


def gold_pages(rec: dict) -> set[tuple[str, int]]:
    return {
        (ev.get('doc_name') or rec['doc_name'],
         int(ev['evidence_page_num']) + PAGE_OFFSET)
        for ev in rec.get('evidence', [])
        if ev.get('evidence_page_num') is not None
    }


def fuse(dense_hits: list[dict], bm25_hits: list[dict], dense_weight: float,
         rrf_c: int, k: int) -> list[dict]:
    fused: dict[str, dict] = {}
    for hits, weight in ((dense_hits, dense_weight),
                         (bm25_hits, 1.0 - dense_weight)):
        if weight == 0:
            continue
        for hit in hits:
            slot = fused.setdefault(hit['chunk_id'], {'hit': hit, 'score': 0.0})
            slot['score'] += weight / (rrf_c + hit['rank'])
    ranked = sorted(fused.values(), key=lambda x: x['score'], reverse=True)[:k]
    return [x['hit'] for x in ranked]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dense', required=True)
    ap.add_argument('--bm25', required=True)
    ap.add_argument('--weights', default='0,0.5,0.7,0.8,0.9,1')
    ap.add_argument('--pool', type=int, default=100)
    ap.add_argument('--rrf-c', type=int, default=60)
    args = ap.parse_args()
    weights = [float(x) for x in args.weights.split(',')]

    retr = HybridRetriever(args.dense, args.bm25, pool=args.pool,
                            rrf_c=args.rrf_c)
    recs = [json.loads(line) for line in open(EVAL_JSONL)]

    cached = []
    for rec in tqdm(recs, desc='retrieving once'):
        cached.append((
            rec,
            retr.dense.search(rec['question'], args.pool),
            retr.bm25.search(rec['question'], args.pool),
        ))

    summaries = []
    RUNS.mkdir(exist_ok=True)
    for weight in weights:
        rows = []
        for rec, dense_hits, bm25_hits in cached:
            gold = gold_pages(rec)
            gold_docs = {doc for doc, _ in gold}
            hits = fuse(dense_hits, bm25_hits, weight, args.rrf_c, max(K_LIST))
            retrieved = [(h['doc_name'], h['page_number']) for h in hits]
            first = next((i + 1 for i, page in enumerate(retrieved)
                          if page in gold), None)
            row = {
                'financebench_id': rec['financebench_id'],
                'question_type': rec['question_type'],
                'doc_name': rec['doc_name'],
                'dense_weight': weight,
                'first_page_hit_rank': first,
                'mrr': 1.0 / first if first else 0.0,
            }
            for k in K_LIST:
                top = retrieved[:k]
                row[f'doc_hit@{k}'] = int(any(d in gold_docs for d, _ in top))
                row[f'page_hit@{k}'] = int(any(p in gold for p in top))
            rows.append(row)

        df = pd.DataFrame(rows)
        tag = f'd{weight:.2f}'.replace('.', 'p')
        df.to_json(RUNS / f'retrieval_hybrid-{tag}__bge-small__s256_o50.jsonl',
                   orient='records', lines=True)
        summary = {
            'dense_weight': weight,
            'bm25_weight': 1.0 - weight,
            'mrr': df['mrr'].mean(),
        }
        for k in K_LIST:
            summary[f'doc_hit@{k}'] = df[f'doc_hit@{k}'].mean()
            summary[f'page_hit@{k}'] = df[f'page_hit@{k}'].mean()
        summaries.append(summary)

    out = pd.DataFrame(summaries)
    out.to_csv(RUNS / 'retrieval_hybrid_weight_sweep.csv', index=False)
    print('\n=== hybrid RRF weight sweep ===')
    cols = ['dense_weight', 'bm25_weight', 'mrr']
    cols += [f'page_hit@{k}' for k in K_LIST]
    print(out[cols].round(3).to_string(index=False))
    print('\nsaved -> runs/retrieval_hybrid_weight_sweep.csv')


if __name__ == '__main__':
    main()
