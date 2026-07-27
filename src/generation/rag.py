"""End-to-end RAG pipeline: retrieve -> generate a cited answer.

Combines a Retriever (FAISS index) with a Generator (local LLM). Used both from
the CLI (ask one question, or replay a FinanceBench item by id) and as a library
by the answer-eval harness.

CLI:
  python src/generation/rag.py --index data/index/bge-small__s256_o50 \
      --gen-config configs/gen_baseline.yaml --query "..."
  python src/generation/rag.py --index ... --gen-config ... --fb-id financebench_id_03029
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'retrieval'))
from retriever import Retriever  # noqa: E402
from generator import Generator  # noqa: E402

EVAL_JSONL = Path('data/financebench/financebench_open_source.jsonl')


class RAG:
    def __init__(self, index_dir: str, gen_config: str):
        self.retriever = Retriever(index_dir)
        self.generator = Generator(gen_config)
        self.k = self.generator.cfg.get('retrieval_k', 10)

    def answer(self, question: str, k: int | None = None) -> dict:
        contexts = self.retriever.search(question, k=k or self.k)
        result = self.generator.answer(question, contexts)
        result['contexts'] = contexts
        return result


def _load_fb(fb_id: str) -> dict:
    for line in open(EVAL_JSONL):
        rec = json.loads(line)
        if rec['financebench_id'] == fb_id:
            return rec
    raise SystemExit(f'no such financebench_id: {fb_id}')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--index', required=True)
    ap.add_argument('--gen-config', required=True)
    ap.add_argument('--query')
    ap.add_argument('--fb-id', help='replay a FinanceBench question by id')
    ap.add_argument('--k', type=int)
    args = ap.parse_args()

    gold = None
    if args.fb_id:
        rec = _load_fb(args.fb_id)
        question, gold = rec['question'], rec
    elif args.query:
        question = args.query
    else:
        raise SystemExit('provide --query or --fb-id')

    rag = RAG(args.index, args.gen_config)
    res = rag.answer(question, k=args.k)

    print(f"\nQ: {question}\n")
    print(f"A: {res['answer']}\n")
    print(f"cited passages: {res['cited']}   refused: {res['refused']}")
    if gold:
        print(f"\nGOLD answer: {gold['answer']}")
        print(f"GOLD evidence page(s): "
              f"{sorted({ev['evidence_page_num'] + 1 for ev in gold['evidence'] if ev.get('evidence_page_num') is not None})}")
    print('\nretrieved contexts:')
    for i, c in enumerate(res['contexts'], 1):
        marker = '*' if i in res['cited'] else ' '
        print(f" {marker}[{i}] {c['doc_name']} p{c['page_number']}  (score {c['score']:.3f})")


if __name__ == '__main__':
    main()
