"""Embed a chunk set and build a FAISS index for retrieval.

Reads  data/chunks/{...}.parquet  (chunk set named by an embed config)
Writes data/index/{config.name}/
         index.faiss   - IndexFlatIP over unit-normalized chunk vectors (exact
                         cosine search)
         meta.parquet  - chunk metadata in the SAME row order as the index, so
                         a FAISS result id maps straight to meta.iloc[id]
         config.json   - provenance (model, dim, chunk set, counts, timings)

Design: exact flat index (not approximate/HNSW) so retrieval error never
confounds the experiments; 211k x 384 fp32 is only ~325 MB. Passages are
embedded with no instruction prefix; the query instruction (bge-v1.5) is stored
in config for the retriever to apply at query time.

Usage:
  python src/retrieval/build_index.py --config configs/embed_baseline.yaml
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
import yaml
from sentence_transformers import SentenceTransformer

INDEX_ROOT = Path('data/index')
META_COLS = ['chunk_id', 'doc_name', 'page_number', 'text', 'n_tokens']


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument(
        '--max-new-shards', type=int,
        help='encode at most this many new shards, then exit resumably',
    )
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = INDEX_ROOT / cfg['name']
    if (out_dir / 'index.faiss').exists():
        print(f"index already exists at {out_dir} - delete it to rebuild")
        return

    chunks = pd.read_parquet(cfg['chunks']).reset_index(drop=True)
    passages = (cfg.get('passage_prefix', '') + chunks['text']).tolist()
    print(f"{len(chunks):,} chunks from {cfg['chunks']}")

    device = cfg.get('device') or pick_device()
    fp16 = cfg.get('fp16', False)
    shard_size = int(cfg.get('shard_size', len(passages)))
    ranges = [
        (start, min(start + shard_size, len(passages)))
        for start in range(0, len(passages), shard_size)
    ]
    shard_dir = out_dir / '_embedding_shards'
    shard_dir.mkdir(parents=True, exist_ok=True)
    progress_path = shard_dir / 'progress.json'
    progress = (
        json.loads(progress_path.read_text())
        if progress_path.exists()
        else {'encode_secs': 0.0}
    )
    shard_paths = [
        shard_dir / f'{start:09d}_{end:09d}.npy'
        for start, end in ranges
    ]
    missing = [
        (start, end, path)
        for (start, end), path in zip(ranges, shard_paths)
        if not path.exists()
    ]

    reuse_positions = None
    reuse_index = None
    if missing and cfg.get('reuse_index'):
        reuse_dir = Path(cfg['reuse_index'])
        reuse_meta = pd.read_parquet(reuse_dir / 'meta.parquet')
        # Embeddings depend only on passage text, but include doc_name in the
        # key to avoid accidentally reusing boilerplate across filings.
        key_to_position = {
            (doc, text): pos
            for pos, (doc, text) in enumerate(
                zip(reuse_meta['doc_name'], reuse_meta['text'])
            )
        }
        reuse_positions = np.array([
            key_to_position.get((doc, text), -1)
            for doc, text in zip(chunks['doc_name'], passages)
        ], dtype=np.int64)
        reused = int((reuse_positions >= 0).sum())
        print(f'reusing {reused:,}/{len(chunks):,} exact-text vectors '
              f"from {cfg['reuse_index']}")
        reuse_index = faiss.read_index(str(reuse_dir / 'index.faiss'))

    if missing:
        print(f"loading {cfg['embed_model']} on {device}"
              f"{' (fp16)' if fp16 else ''}")
        if device == 'cpu':
            torch.set_num_threads(cfg.get('cpu_threads', 1))
        model = SentenceTransformer(
            cfg['embed_model'],
            device=device,
            model_kwargs={'attn_implementation': 'eager'},
        )
        if fp16:
            model = model.half()   # faster on MPS; negligible retrieval impact

        made = 0
        for start, end, path in missing:
            print(f'encoding shard {start:,}:{end:,} '
                  f'({len(passages) - end:,} chunks remain after it)')
            t0 = time.perf_counter()
            if reuse_positions is not None:
                source_ids = reuse_positions[start:end]
                reused_mask = source_ids >= 0
                shard = np.empty((end - start, reuse_index.d), dtype=np.float32)
                if reused_mask.any():
                    shard[reused_mask] = reuse_index.reconstruct_batch(
                        source_ids[reused_mask])
                new_local = np.flatnonzero(~reused_mask)
                new_passages = [passages[start + i] for i in new_local]
                if new_passages:
                    shard[new_local] = model.encode(
                        new_passages,
                        batch_size=cfg.get('batch_size', 64),
                        normalize_embeddings=cfg.get('normalize', True),
                        show_progress_bar=True,
                        convert_to_numpy=True,
                    ).astype(np.float32)
                print(f'  reused {int(reused_mask.sum()):,}; '
                      f'encoded {len(new_passages):,}')
            else:
                shard = model.encode(
                    passages[start:end],
                    batch_size=cfg.get('batch_size', 64),
                    normalize_embeddings=cfg.get('normalize', True),
                    show_progress_bar=True,
                    convert_to_numpy=True,
                ).astype(np.float32)
            elapsed = time.perf_counter() - t0
            np.save(path, shard)
            progress['encode_secs'] += elapsed
            progress_path.write_text(json.dumps(progress, indent=2))
            made += 1
            remaining = sum(not p.exists() for p in shard_paths)
            print(f'saved {path.name} ({remaining} shards remain)')
            if (args.max_new_shards and made >= args.max_new_shards
                    and remaining):
                print('resume with the same command to encode the next shard')
                return

    emb = np.concatenate(
        [np.load(path) for path in shard_paths],
        axis=0,
    ).astype(np.float32, copy=False)
    dim = emb.shape[1]
    encode_secs = round(progress['encode_secs'], 1)
    print(f"encoded {emb.shape} in {encode_secs}s ({len(chunks)/encode_secs:.0f} chunks/s)")

    if cfg.get('normalize', True):
        faiss.normalize_L2(emb)   # exact unit norm in float32 (fp16 encode drifts slightly)

    index = faiss.IndexFlatIP(dim)   # normalized vectors -> inner product == cosine
    index.add(emb)

    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / 'index.faiss'))
    chunks[META_COLS].to_parquet(out_dir / 'meta.parquet', index=False)
    with open(out_dir / 'config.json', 'w') as f:
        json.dump({
            **cfg,
            'dim': dim,
            'n_vectors': int(index.ntotal),
            'device': device,
            'encode_secs': encode_secs,
            'built_at': datetime.now(timezone.utc).isoformat(),
        }, f, indent=2)

    print(f"\nwrote index ({index.ntotal:,} x {dim}) -> {out_dir}")


if __name__ == '__main__':
    main()
