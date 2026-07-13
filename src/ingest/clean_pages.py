"""Clean parsed page text conservatively, in preparation for chunking.

Reads  data/parsed/{doc}.parquet  (one row per page, raw pdfplumber text)
Writes data/clean/{doc}.parquet   (same rows, cleaned `text`)

Design goals: fix extraction artifacts and strip running boilerplate WITHOUT
destroying the financial-table structure that pdfplumber preserves. Every
removal is auditable.

Cleaning steps (see docs/decisions.md):
  1. Unicode NFKC normalization  -> ligatures (fi/fl), NBSP->space, fancy
     quotes/dashes folded to ASCII-ish equivalents.
  2. Per-line whitespace         -> strip ends; collapse internal runs of
     spaces/tabs to a single space.
  3. Running header/footer removal -> a line is treated as boilerplate only if
     it sits in the top-3 or bottom-3 lines of a page, is short (<= MAX_HF_LEN),
     and its digit-masked signature recurs on >= HF_FREQ of the document's
     pages. This targets true running heads/page numbers, not content.
  4. Blank-line collapse         -> 2+ consecutive blank lines -> 1.

Idempotent: skips docs whose clean parquet already exists.
Audit ledger: data/clean_manifest.csv  (per-doc removed-line signatures + counts).

Usage:
  python src/ingest/clean_pages.py                # full corpus
  python src/ingest/clean_pages.py DOC1 DOC2 ...  # only named docs (by doc_name)
  python src/ingest/clean_pages.py --dry-run DOC  # print before/after, write nothing
"""
import argparse
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

PARSED_DIR = Path('data/parsed')
CLEAN_DIR = Path('data/clean')
MANIFEST = Path('data/clean_manifest.csv')

# header/footer detection tuning
MAX_HF_LEN = 60      # candidate boilerplate lines must be at most this many chars
HF_EDGE = 3          # only the top-N and bottom-N lines of a page are candidates
HF_FREQ = 0.50       # signature must appear on >= this fraction of pages
HF_MIN_PAGES = 4     # don't run header/footer removal on very short docs (8-Ks)

_DIGITS = re.compile(r'\d+')
_WS_RUN = re.compile(r'[ \t]+')
_BLANKS = re.compile(r'\n{3,}')


def signature(line: str) -> str:
    """Digit-masked, lowercased signature so 'Page 12' and 'Page 13' collide."""
    return _DIGITS.sub('#', line.strip().lower())


def detect_boilerplate(pages: list[str]) -> set[str]:
    """Return the set of digit-masked signatures that are running headers/footers."""
    n = len(pages)
    if n < HF_MIN_PAGES:
        return set()
    counts: Counter[str] = Counter()
    for text in pages:
        lines = text.split('\n')
        edge = lines[:HF_EDGE] + lines[-HF_EDGE:]
        seen = set()
        for ln in edge:
            s = ln.strip()
            if s and len(s) <= MAX_HF_LEN:
                sig = signature(s)
                if sig not in seen:      # count each signature once per page
                    seen.add(sig)
                    counts[sig] += 1
    threshold = HF_FREQ * n
    return {sig for sig, c in counts.items() if c >= threshold}


def clean_page(text: str, boilerplate: set[str]) -> str:
    text = unicodedata.normalize('NFKC', text)
    lines = text.split('\n')
    n = len(lines)
    out = []
    for i, ln in enumerate(lines):
        ln = _WS_RUN.sub(' ', ln).strip()
        # only strip boilerplate when it appears at a page edge
        is_edge = i < HF_EDGE or i >= n - HF_EDGE
        if is_edge and ln and signature(ln) in boilerplate:
            continue
        out.append(ln)
    cleaned = '\n'.join(out)
    cleaned = _BLANKS.sub('\n\n', cleaned).strip()
    return cleaned


def process_doc(doc: str, dry_run: bool = False) -> dict:
    df = pd.read_parquet(PARSED_DIR / f'{doc}.parquet').sort_values('page_number')
    raw_pages = df['text'].fillna('').tolist()
    boiler = detect_boilerplate(raw_pages)

    clean_pages = [clean_page(t, boiler) for t in raw_pages]
    raw_chars = sum(len(t) for t in raw_pages)
    clean_chars = sum(len(t) for t in clean_pages)

    if not dry_run:
        out = df.copy()
        out['text'] = clean_pages
        out['char_count_raw'] = [len(t) for t in raw_pages]
        out['char_count'] = [len(t) for t in clean_pages]
        CLEAN_DIR.mkdir(parents=True, exist_ok=True)
        out[['doc_name', 'page_number', 'text', 'char_count_raw', 'char_count']].to_parquet(
            CLEAN_DIR / f'{doc}.parquet', index=False)

    return {
        'doc_name': doc,
        'pages': len(raw_pages),
        'chars_raw': raw_chars,
        'chars_clean': clean_chars,
        'pct_removed': round(100 * (raw_chars - clean_chars) / raw_chars, 2) if raw_chars else 0.0,
        'boilerplate_sigs': ' | '.join(sorted(boiler)) if boiler else '',
        'n_boilerplate': len(boiler),
    }


def show_diff(doc: str) -> None:
    """Print a before/after sample for one doc so a human can eyeball cleaning."""
    df = pd.read_parquet(PARSED_DIR / f'{doc}.parquet').sort_values('page_number')
    raw_pages = df['text'].fillna('').tolist()
    boiler = detect_boilerplate(raw_pages)
    print(f'\n{"="*70}\n{doc}   ({len(raw_pages)} pages)')
    print(f'boilerplate signatures detected ({len(boiler)}):')
    for s in sorted(boiler):
        print(f'    - {s!r}')
    # pick a middle page (more likely to have a running head than page 1)
    idx = min(len(raw_pages) - 1, max(1, len(raw_pages) // 2))
    raw = raw_pages[idx]
    print(f'\n--- RAW  page {idx+1} (first 900 chars) ---')
    print(raw[:900])
    print(f'\n--- CLEAN page {idx+1} (first 900 chars) ---')
    print(clean_page(raw, boiler)[:900])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('docs', nargs='*', help='doc_names to process (default: all)')
    ap.add_argument('--dry-run', action='store_true',
                    help='print before/after for the given docs, write nothing')
    args = ap.parse_args()

    all_docs = sorted(p.stem for p in PARSED_DIR.glob('*.parquet'))
    targets = args.docs if args.docs else all_docs

    if args.dry_run:
        for doc in targets:
            show_diff(doc)
        return

    done = {p.stem for p in CLEAN_DIR.glob('*.parquet')} if CLEAN_DIR.exists() else set()
    todo = [d for d in targets if d not in done]
    print(f'{len(targets)} target docs, {len(done)} already clean, {len(todo)} to do')

    records = []
    for doc in todo:
        records.append(process_doc(doc))

    if records:
        new = pd.DataFrame(records)
        if MANIFEST.exists():
            old = pd.read_csv(MANIFEST)
            new = pd.concat([old[~old['doc_name'].isin(new['doc_name'])], new], ignore_index=True)
        new.sort_values('doc_name').to_csv(MANIFEST, index=False)
        print(f'wrote {len(records)} docs; total removed '
              f'{new["chars_raw"].sum() - new["chars_clean"].sum():,} chars '
              f'({100*(new["chars_raw"].sum()-new["chars_clean"].sum())/new["chars_raw"].sum():.2f}% of corpus)')


if __name__ == '__main__':
    main()
