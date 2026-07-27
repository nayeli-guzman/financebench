"""Generate with a new LLM while reusing previously retrieved contexts.

This is the correct runner for generator-only experiments: both generators see
the exact same top-k chunks, so model size is the only independent variable.
It also avoids keeping the embedding model resident in unified GPU memory while
Ollama serves a larger generator.

Usage:
  python src/generation/regenerate_from_contexts.py \
    --source runs/answers_qwen2.5-3b__bge-small__s256_o50.jsonl \
    --gen-config configs/gen_qwen7b.yaml
"""
import argparse
import json
import time
from pathlib import Path

from tqdm import tqdm

from generator import Generator

RUNS = Path('runs')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True,
                    help='completed answers JSONL whose contexts are reused')
    ap.add_argument('--gen-config', required=True)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--page-hit-only', action='store_true',
                    help='generate only questions whose gold page was retrieved')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    source = [json.loads(line) for line in open(args.source)]
    if args.limit:
        source = source[:args.limit]
    if args.page_hit_only:
        source = [row for row in source if row['page_hit']]
    if not source:
        raise SystemExit('source file is empty')

    generator = Generator(args.gen_config)
    source_tag = Path(args.source).stem.replace('answers_', '')
    index_tag = source_tag.split('__', 1)[1]
    out_path = RUNS / f"answers_{generator.cfg['name']}__{index_tag}.jsonl"
    RUNS.mkdir(exist_ok=True)

    done: set[str] = set()
    if out_path.exists() and not args.overwrite:
        for line in open(out_path):
            try:
                done.add(json.loads(line)['financebench_id'])
            except json.JSONDecodeError:
                pass
    todo = [row for row in source if row['financebench_id'] not in done]
    mode = 'w' if args.overwrite or not done else 'a'

    print(f"generating {len(todo)} answers ({len(done)} already done) | "
          f"contexts from {Path(args.source).name} | gen {generator.cfg['name']}")
    t0 = time.time()
    with open(out_path, mode) as out:
        for row in tqdm(todo):
            result = generator.answer(row['question'], row['contexts'])
            copied = {k: v for k, v in row.items()
                      if k not in ('answer', 'cited', 'refused',
                                   'cited_pages_valid')}
            gold = {tuple(page) for page in row['gold_pages']}
            contexts = row['contexts']
            cited_pages_valid = any(
                (contexts[i - 1]['doc_name'], contexts[i - 1]['page_number']) in gold
                for i in result['cited'] if 1 <= i <= len(contexts)
            )
            copied.update({
                'answer': result['answer'],
                'cited': result['cited'],
                'refused': result['refused'],
                'cited_pages_valid': cited_pages_valid,
            })
            out.write(json.dumps(copied) + '\n')
            out.flush()

    elapsed = time.time() - t0
    n = max(len(todo), 1)
    print(f"\nwrote {len(todo)} answers -> {out_path} "
          f"({elapsed / 60:.1f} min, {elapsed / n:.1f}s/q)")


if __name__ == '__main__':
    main()
