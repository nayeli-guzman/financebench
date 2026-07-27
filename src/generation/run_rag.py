"""Run the RAG pipeline over the full FinanceBench eval set and record answers.

Separated from scoring so answers are generated once and can be re-judged with
different judges/rubrics without re-running the (slow) LLM. For each question we
store the generated answer, which contexts were retrieved (with page-hit flags
computed against gold evidence pages), and citation/refusal signals.

Output: runs/answers_{gen}__{index}.jsonl  (one row per question)

Usage:
  python src/generation/run_rag.py --index data/index/bge-small__s256_o50 \
      --gen-config configs/gen_baseline.yaml
"""
import argparse
import json
import time
from pathlib import Path

from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'retrieval'))
from rag import RAG  # noqa: E402

EVAL_JSONL = Path('data/financebench/financebench_open_source.jsonl')
RUNS = Path('runs')
PAGE_OFFSET = 1   # our_page = evidence_page_num + 1 (see docs/decisions.md §8)


def gold_pages(rec: dict) -> list[list]:
    out = []
    for ev in rec.get('evidence', []):
        p = ev.get('evidence_page_num')
        d = ev.get('doc_name') or rec['doc_name']
        if p is not None:
            out.append([d, int(p) + PAGE_OFFSET])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--index', required=True)
    ap.add_argument('--gen-config', required=True)
    ap.add_argument('--limit', type=int, help='only first N questions (smoke test)')
    ap.add_argument('--overwrite', action='store_true',
                    help='regenerate from scratch instead of resuming')
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(EVAL_JSONL)]
    if args.limit:
        recs = recs[:args.limit]
    rag = RAG(args.index, args.gen_config)
    gen_name = rag.generator.cfg['name']
    idx_name = rag.retriever.cfg['name']
    out_path = RUNS / f'answers_{gen_name}__{idx_name}.jsonl'
    RUNS.mkdir(exist_ok=True)

    # resume: skip questions already generated (this run is ~11s/question)
    done: set[str] = set()
    if out_path.exists() and not args.overwrite:
        with open(out_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)['financebench_id'])
                except json.JSONDecodeError:
                    pass   # tolerate a truncated final line from a killed run
    todo = [r for r in recs if r['financebench_id'] not in done]
    mode = 'w' if (args.overwrite or not done) else 'a'
    print(f'generating {len(todo)} answers ({len(done)} already done) | '
          f'index {idx_name} | gen {gen_name} | k={rag.k}')

    t0 = time.time()
    with open(out_path, mode) as f:
        for rec in tqdm(todo):
            gold = gold_pages(rec)
            gold_set = {tuple(g) for g in gold}
            res = rag.answer(rec['question'])
            ctx = [{'rank': c['rank'], 'doc_name': c['doc_name'],
                    'page_number': c['page_number'], 'chunk_id': c['chunk_id'],
                    'score': c['score'], 'text': c['text']} for c in res['contexts']]
            hit_ranks = [c['rank'] for c in ctx if (c['doc_name'], c['page_number']) in gold_set]
            # did the model cite a context that is actually a gold evidence page?
            cited_pages_valid = any(
                (ctx[i - 1]['doc_name'], ctx[i - 1]['page_number']) in gold_set
                for i in res['cited'] if 1 <= i <= len(ctx))
            f.write(json.dumps({
                'financebench_id': rec['financebench_id'],
                'question_type': rec['question_type'],
                'doc_name': rec['doc_name'],
                'question': rec['question'],
                'gold_answer': rec['answer'],
                'gold_pages': gold,
                'answer': res['answer'],
                'cited': res['cited'],
                'refused': res['refused'],
                'page_hit': bool(hit_ranks),
                'first_hit_rank': min(hit_ranks) if hit_ranks else None,
                'doc_hit': any(c['doc_name'] == rec['doc_name'] for c in ctx),
                'cited_pages_valid': cited_pages_valid,
                'contexts': ctx,
            }) + '\n')
            f.flush()   # survive interruption; resume picks up from here

    dt = time.time() - t0
    n = max(len(todo), 1)
    print(f'\nwrote {len(todo)} answers -> {out_path}  ({dt/60:.1f} min, {dt/n:.1f}s/q)')


if __name__ == '__main__':
    main()
