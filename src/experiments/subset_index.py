"""Create a document-filtered FAISS index by reusing existing vectors."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-index', required=True)
    ap.add_argument('--eval-jsonl', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--output-root', default='data/index')
    args = ap.parse_args()

    source_dir = Path(args.source_index)
    out_dir = Path(args.output_root) / args.name
    if (out_dir / 'index.faiss').exists():
        print(f'index already exists at {out_dir}')
        return

    docs = {
        json.loads(line)['doc_name']
        for line in open(args.eval_jsonl)
    }
    meta = pd.read_parquet(source_dir / 'meta.parquet')
    positions = np.flatnonzero(meta['doc_name'].isin(docs).to_numpy()).astype(np.int64)
    subset_meta = meta.iloc[positions].reset_index(drop=True)

    source = faiss.read_index(str(source_dir / 'index.faiss'))
    vectors = source.reconstruct_batch(positions)
    subset = faiss.IndexFlatIP(source.d)
    subset.add(np.asarray(vectors, dtype=np.float32))

    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(subset, str(out_dir / 'index.faiss'))
    subset_meta.to_parquet(out_dir / 'meta.parquet', index=False)
    source_cfg = json.loads((source_dir / 'config.json').read_text())
    (out_dir / 'config.json').write_text(json.dumps({
        **source_cfg,
        'name': args.name,
        'n_vectors': int(subset.ntotal),
        'subset_of': str(source_dir),
        'eval_jsonl': args.eval_jsonl,
        'subset_docs': len(docs),
        'built_at': datetime.now(timezone.utc).isoformat(),
    }, indent=2))
    print(
        f'reused {subset.ntotal:,} vectors for {len(docs)} documents '
        f'-> {out_dir}'
    )


if __name__ == '__main__':
    main()
