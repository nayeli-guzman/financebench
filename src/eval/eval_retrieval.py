"""Retrieval-only evaluation against FinanceBench gold evidence pages.

For each of the 150 questions we retrieve top-K chunks from a FAISS index and
score whether the gold evidence page(s) were found. This needs no LLM, so it is
cheap and objective -- ideal for sweeping k and comparing indexes.

Page-number convention (calibrated empirically, see docs/decisions.md): the
dataset's `evidence_page_num` is 0-indexed; our pdfplumber `page_number` is
1-indexed. So a gold page P maps to our page P + PAGE_OFFSET (default 1).

Metrics (computed at every k in K_LIST from a single K=max retrieval):
  doc_hit@k     - any retrieved chunk is from the correct document
  page_hit@k    - any retrieved chunk is on a gold (doc, page)
  page_recall@k - fraction of the question's gold pages that were retrieved
  evidence_hit@k - any gold evidence span has >=30% normalized trigram recall
  evidence_all_hit@k - every gold evidence span reaches that threshold
  evidence_token_recall@k - mean unique-token recall across evidence spans
  evidence_ngram_recall@k - mean unique-trigram recall across evidence spans
  mrr           - 1 / rank of the first page hit (0 if none in top K)

Outputs:
  runs/retrieval_{index}.jsonl  - per-question detail
  runs/retrieval_{index}.csv    - summary (overall + by question_type)

Usage:
  python src/eval/eval_retrieval.py --index data/index/bge-small__s256_o50
"""
import argparse
import json
from pathlib import Path
import re

import pandas as pd
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'retrieval'))
from retriever import Retriever  # noqa: E402
from hybrid_retriever import HybridRetriever  # noqa: E402

EVAL_JSONL = Path('data/financebench/financebench_open_source.jsonl')
RUNS = Path('runs')
K_LIST = [1, 3, 5, 10, 20]
PAGE_OFFSET = 1   # our_page = evidence_page_num + 1
EVIDENCE_NGRAM = 3
EVIDENCE_HIT_THRESHOLD = 0.30
TOKEN_RE = re.compile(r'[a-z0-9]+')


def gold_pages(rec: dict) -> set[tuple[str, int]]:
    """Set of (doc_name, our_page_number) the evidence lives on."""
    out = set()
    for ev in rec.get('evidence', []):
        p = ev.get('evidence_page_num')
        d = ev.get('doc_name') or rec['doc_name']
        if p is not None:
            out.add((d, int(p) + PAGE_OFFSET))
    return out


def normalized_tokens(text: str) -> list[str]:
    """Extraction-tolerant lowercase alphanumeric tokenization."""
    return TOKEN_RE.findall((text or '').lower())


def ngrams(tokens: list[str], n: int = EVIDENCE_NGRAM) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def evidence_spans(rec: dict) -> list[dict]:
    """Pre-tokenized gold evidence spans, preserving their source document."""
    spans = []
    for ev in rec.get('evidence', []):
        tokens = normalized_tokens(ev.get('evidence_text', ''))
        if not tokens:
            continue
        spans.append({
            'doc_name': ev.get('doc_name') or rec['doc_name'],
            'tokens': set(tokens),
            'ngrams': ngrams(tokens),
        })
    return spans


def evidence_scores(spans: list[dict], hits: list[dict]) -> dict:
    """Recall of each gold span in the union of retrieved same-doc chunks.

    Same-document restriction prevents generic filing vocabulary in an
    unrelated company/year from creating a false evidence match.
    """
    token_recalls = []
    ngram_recalls = []
    for span in spans:
        text = ' '.join(h['text'] for h in hits
                        if h['doc_name'] == span['doc_name'])
        retrieved_tokens = normalized_tokens(text)
        retrieved_token_set = set(retrieved_tokens)
        retrieved_ngrams = ngrams(retrieved_tokens)
        token_recalls.append(
            len(span['tokens'] & retrieved_token_set) / len(span['tokens'])
            if span['tokens'] else 0.0
        )
        ngram_recalls.append(
            len(span['ngrams'] & retrieved_ngrams) / len(span['ngrams'])
            if span['ngrams'] else 0.0
        )
    if not spans:
        return {'token_recall': 0.0, 'ngram_recall': 0.0,
                'any_hit': 0, 'all_hit': 0}
    hits_at_threshold = [r >= EVIDENCE_HIT_THRESHOLD for r in ngram_recalls]
    return {
        'token_recall': sum(token_recalls) / len(token_recalls),
        'ngram_recall': sum(ngram_recalls) / len(ngram_recalls),
        'any_hit': int(any(hits_at_threshold)),
        'all_hit': int(all(hits_at_threshold)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--index', required=True)
    ap.add_argument('--bm25', help='BM25 index dir -> evaluate HYBRID (dense+BM25) instead of dense')
    ap.add_argument('--dense-weight', type=float, default=0.5,
                    help='RRF dense weight (0=BM25 only, 1=dense only)')
    ap.add_argument('--device', choices=('cpu', 'mps', 'cuda'),
                    help='query-embedding device; CPU is most reliable on macOS')
    ap.add_argument('--eval-jsonl', default=str(EVAL_JSONL),
                    help='questions/evidence JSONL (default: full FinanceBench set)')
    ap.add_argument('--max-k', type=int, default=max(K_LIST))
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.eval_jsonl)]
    retr = (HybridRetriever(args.index, args.bm25,
                            dense_weight=args.dense_weight,
                            device=args.device)
            if args.bm25 else Retriever(args.index, device=args.device))
    name = retr.cfg['name']
    ntotal = getattr(retr, 'ntotal', None) or retr.index.ntotal
    print(f"evaluating retrieval on {len(recs)} questions | index {name} "
          f"({ntotal:,} chunks)\n")

    details = []
    for rec in tqdm(recs):
        gold = gold_pages(rec)
        spans = evidence_spans(rec)
        gold_docs = {d for d, _ in gold}
        hits = retr.search(rec['question'], k=args.max_k)
        retrieved = [(h['doc_name'], h['page_number']) for h in hits]

        # rank (1-based) of first page hit
        first_hit = next((i + 1 for i, rp in enumerate(retrieved) if rp in gold), None)
        row = {
            'financebench_id': rec['financebench_id'],
            'question_type': rec['question_type'],
            'doc_name': rec['doc_name'],
            'n_gold_pages': len(gold),
            'first_page_hit_rank': first_hit,
            'mrr': (1.0 / first_hit) if first_hit else 0.0,
        }
        for k in K_LIST:
            topk = retrieved[:k]
            ev = evidence_scores(spans, hits[:k])
            row[f'doc_hit@{k}'] = int(any(d in gold_docs for d, _ in topk))
            row[f'page_hit@{k}'] = int(any(rp in gold for rp in topk))
            row[f'page_recall@{k}'] = (len(gold & set(topk)) / len(gold)) if gold else 0.0
            row[f'evidence_hit@{k}'] = ev['any_hit']
            row[f'evidence_all_hit@{k}'] = ev['all_hit']
            row[f'evidence_token_recall@{k}'] = ev['token_recall']
            row[f'evidence_ngram_recall@{k}'] = ev['ngram_recall']
        details.append(row)

    df = pd.DataFrame(details)
    RUNS.mkdir(exist_ok=True)
    df.to_json(RUNS / f'retrieval_{name}.jsonl', orient='records', lines=True)

    metric_names = (
        'doc_hit', 'page_hit', 'page_recall', 'evidence_hit',
        'evidence_all_hit', 'evidence_token_recall',
        'evidence_ngram_recall',
    )
    metric_cols = ['mrr'] + [f'{m}@{k}' for m in metric_names for k in K_LIST]
    overall = df[metric_cols].mean().to_frame('overall').T
    by_type = df.groupby('question_type')[metric_cols].mean()
    summary = pd.concat([overall, by_type])
    summary.to_csv(RUNS / f'retrieval_{name}.csv')

    pd.set_option('display.width', 200, 'display.max_columns', 40)
    print('\n=== page_hit@k (found the right page) ===')
    print(summary[[f'page_hit@{k}' for k in K_LIST]].round(3).to_string())
    print('\n=== doc_hit@k (found the right document) ===')
    print(summary[[f'doc_hit@{k}' for k in K_LIST]].round(3).to_string())
    print('\n=== mrr (page-level) ===')
    print(summary[['mrr']].round(3).to_string())
    print(f'\n=== evidence_hit@k (>={EVIDENCE_HIT_THRESHOLD:.0%} '
          f'{EVIDENCE_NGRAM}-gram recall) ===')
    print(summary[[f'evidence_hit@{k}' for k in K_LIST]].round(3).to_string())
    print('\n=== evidence_ngram_recall@k ===')
    print(summary[[f'evidence_ngram_recall@{k}' for k in K_LIST]]
          .round(3).to_string())
    print(f'\nsaved -> runs/retrieval_{name}.jsonl , runs/retrieval_{name}.csv')


if __name__ == '__main__':
    main()
