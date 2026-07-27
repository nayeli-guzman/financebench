"""Chunk cleaned pages into retrieval passages, one config at a time.

Reads  data/clean/{doc}.parquet   (one row per cleaned page)
Writes data/chunks/{config.name}.parquet   (one row per chunk, whole corpus)

Chunking is the primary experimental knob (see the assignment). This script is
config-driven: a YAML file fixes the strategy and parameters, so the Phase-E
sweep is just "run again with a different YAML".

Strategy = recursive (page-bounded):
  * Each page is split independently, so every chunk carries exactly one
    page_number -> clean page-level retrieval scoring against evidence_page_num.
  * Text is broken on natural boundaries in priority order
    (blank line -> newline -> sentence -> word) so no atom exceeds chunk_size;
    atoms are then greedily packed to ~chunk_size tokens with chunk_overlap
    tokens carried over. Table rows (single lines) stay intact.
  * Size is measured in TOKENS using the configured tokenizer, so a chunk never
    overflows the embedding model's context window.

Strategy = fixed (page-bounded structure ablation):
  * Each page is split into exact token windows with the configured overlap.
  * Sentence, paragraph, and table-row boundaries are deliberately ignored.
    Comparing this with recursive splitting isolates structure preservation
    while keeping page attribution, size, overlap, and corpus fixed.

Output columns:
  chunk_id (doc__p{page}__c{idx}), doc_name, page_number, chunk_index (global
  order within doc), text, n_tokens, n_chars.

Usage:
  python src/chunk/build_chunks.py --config configs/chunk_baseline.yaml
  python src/chunk/build_chunks.py --config ... --sample DOC   # print samples, write nothing
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer

CLEAN_DIR = Path('data/clean')
OUT_DIR = Path('data/chunks')
EVAL_JSONL = Path('data/financebench/financebench_open_source.jsonl')
SEPARATORS = ['\n\n', '\n', '. ', ' ']   # priority order for recursive splitting


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key in ('name', 'chunk_size', 'chunk_overlap', 'tokenizer'):
        if key not in cfg:
            raise ValueError(f'config missing required key: {key}')
    cfg.setdefault('strategy', 'recursive')
    cfg.setdefault('cross_page', False)
    cfg.setdefault('eval_docs_only', False)
    cfg.setdefault('eval_jsonl', str(EVAL_JSONL))
    if cfg['strategy'] not in ('recursive', 'fixed'):
        raise ValueError("strategy must be 'recursive' or 'fixed'")
    if cfg['chunk_overlap'] >= cfg['chunk_size']:
        raise ValueError('chunk_overlap must be smaller than chunk_size')
    return cfg


def split_to_atoms(text: str, tok, max_tokens: int, level: int = 0) -> list[str]:
    """Recursively break text on natural boundaries so no atom exceeds max_tokens."""
    if len(tok.encode(text, add_special_tokens=False)) <= max_tokens:
        return [text]
    if level >= len(SEPARATORS):
        # last resort: hard split by token windows
        ids = tok.encode(text, add_special_tokens=False)
        return [tok.decode(ids[i:i + max_tokens]) for i in range(0, len(ids), max_tokens)]
    sep = SEPARATORS[level]
    parts = text.split(sep)
    atoms: list[str] = []
    for i, p in enumerate(parts):
        piece = p + (sep if i < len(parts) - 1 else '')
        if not piece.strip():
            continue
        if len(tok.encode(piece, add_special_tokens=False)) <= max_tokens:
            atoms.append(piece)
        else:
            atoms.extend(split_to_atoms(piece, tok, max_tokens, level + 1))
    return atoms


def pack(atoms: list[str], tok, size: int, overlap: int) -> list[tuple[str, int]]:
    """Greedily pack atoms into ~size-token chunks with `overlap`-token backstep."""
    if not atoms:
        return []
    lens = [len(tok.encode(a, add_special_tokens=False)) for a in atoms]
    chunks: list[tuple[str, int]] = []
    start = 0
    n = len(atoms)
    while start < n:
        end, tok_sum = start, 0
        while end < n and (tok_sum + lens[end] <= size or end == start):
            tok_sum += lens[end]
            end += 1
        text = ''.join(atoms[start:end]).strip()
        if text:
            chunks.append((text, tok_sum))
        if end >= n:
            break
        # step back to build the overlap tail, then guarantee forward progress
        ov, ns = 0, end
        while ns > start and ov < overlap:
            ns -= 1
            ov += lens[ns]
        start = max(ns, start + 1)
    return chunks


def fixed_windows(text: str, tok, size: int,
                  overlap: int) -> list[tuple[str, int]]:
    """Exact token windows that intentionally ignore document structure."""
    ids = tok.encode(text, add_special_tokens=False)
    if not ids:
        return []
    step = size - overlap
    chunks = []
    for start in range(0, len(ids), step):
        window = ids[start:start + size]
        decoded = tok.decode(window, skip_special_tokens=True).strip()
        if decoded:
            chunks.append((decoded, len(window)))
        if start + size >= len(ids):
            break
    return chunks


def chunk_doc(df: pd.DataFrame, tok, cfg: dict) -> list[dict]:
    rows: list[dict] = []
    idx = 0
    for _, page in df.sort_values('page_number').iterrows():
        text = page['text'] or ''
        if not text.strip():
            continue
        if cfg['strategy'] == 'fixed':
            page_chunks = fixed_windows(
                text, tok, cfg['chunk_size'], cfg['chunk_overlap'])
        else:
            atoms = split_to_atoms(text, tok, cfg['chunk_size'])
            page_chunks = pack(
                atoms, tok, cfg['chunk_size'], cfg['chunk_overlap'])
        for ctext, ntok in page_chunks:
            rows.append({
                'chunk_id':    f"{page['doc_name']}__p{page['page_number']}__c{idx}",
                'doc_name':    page['doc_name'],
                'page_number': int(page['page_number']),
                'chunk_index': idx,
                'text':        ctext,
                'n_tokens':    ntok,
                'n_chars':     len(ctext),
            })
            idx += 1
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--sample', help='doc_name: print a few chunks and exit, write nothing')
    args = ap.parse_args()

    cfg = load_config(args.config)
    scope = 'eval-docs' if cfg['eval_docs_only'] else 'full-corpus'
    print(f"config '{cfg['name']}': strategy={cfg['strategy']} "
          f"size={cfg['chunk_size']} overlap={cfg['chunk_overlap']} "
          f"scope={scope} tokenizer={cfg['tokenizer']}")
    tok = AutoTokenizer.from_pretrained(cfg['tokenizer'])

    if args.sample:
        df = pd.read_parquet(CLEAN_DIR / f"{args.sample}.parquet")
        rows = chunk_doc(df, tok, cfg)
        print(f'\n{args.sample}: {len(df)} pages -> {len(rows)} chunks')
        for r in rows[:4]:
            print(f"\n--- {r['chunk_id']}  (page {r['page_number']}, {r['n_tokens']} tok, {r['n_chars']} ch) ---")
            print(r['text'][:600])
        return

    docs = sorted(p.stem for p in CLEAN_DIR.glob('*.parquet'))
    if cfg['eval_docs_only']:
        eval_docs = {
            json.loads(line)['doc_name']
            for line in open(cfg['eval_jsonl'])
        }
        docs = [doc for doc in docs if doc in eval_docs]
    all_rows: list[dict] = []
    for doc in tqdm(docs):
        df = pd.read_parquet(CLEAN_DIR / f'{doc}.parquet')
        all_rows.extend(chunk_doc(df, tok, cfg))

    out = pd.DataFrame(all_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{cfg['name']}.parquet"
    out.to_parquet(out_path, index=False)

    print(f'\nwrote {len(out):,} chunks from {out["doc_name"].nunique()} docs -> {out_path}')
    print(f"tokens/chunk: mean {out['n_tokens'].mean():.0f}  median {out['n_tokens'].median():.0f}  "
          f"p95 {out['n_tokens'].quantile(0.95):.0f}  max {out['n_tokens'].max()}")
    print(f"chunks/doc:   mean {len(out)/out['doc_name'].nunique():.0f}")


if __name__ == '__main__':
    main()
