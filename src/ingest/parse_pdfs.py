"""Parse every PDF in data/raw_pdfs/ with pdfplumber.
Output: one parquet per document at data/parsed/{doc_name}.parquet with
columns (doc_name, page_number, text, char_count). One row per page.

Idempotent: skips docs whose parquet already exists.
Records outcomes in data/parse_manifest.csv."""
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pdfplumber
from tqdm import tqdm

PDF_DIR = Path('data/raw_pdfs')
OUT_DIR = Path('data/parsed')
OUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST = Path('data/parse_manifest.csv')

pdfs = sorted(PDF_DIR.glob('*.pdf'))
print(f'{len(pdfs)} PDFs on disk')

# resume support: build todo list from existing outputs
done = {p.stem for p in OUT_DIR.glob('*.parquet')}
todo = [p for p in pdfs if p.stem not in done]
print(f'{len(done)} already parsed, {len(todo)} to do\n')

# load or initialise the manifest
if MANIFEST.exists():
    mdf = pd.read_csv(MANIFEST)
    mdf['status']        = mdf['status'].astype('object')
    mdf['error_message'] = mdf['error_message'].astype('object')
    mdf['parsed_at']     = mdf['parsed_at'].astype('object')
    for col in ('pages', 'chars'):
        mdf[col] = mdf[col].astype('Int64')
else:
    mdf = pd.DataFrame({
        'doc_name': [p.stem for p in pdfs],
        'status':   'pending',
        'pages':    pd.array([pd.NA]*len(pdfs), dtype='Int64'),
        'chars':    pd.array([pd.NA]*len(pdfs), dtype='Int64'),
        'secs':     0.0,
        'error_message': '',
        'parsed_at':     '',
    })

records = mdf.set_index('doc_name').to_dict('index')

for src in tqdm(todo):
    doc = src.stem
    t0 = time.perf_counter()
    try:
        pages = []
        with pdfplumber.open(str(src)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                txt = page.extract_text() or ''
                pages.append({
                    'doc_name':    doc,
                    'page_number': i,
                    'text':        txt,
                    'char_count':  len(txt),
                })
        out_df = pd.DataFrame(pages)
        out_path = OUT_DIR / f'{doc}.parquet'
        out_df.to_parquet(out_path, index=False)

        records[doc] = {
            'status':        'parsed',
            'pages':         len(pages),
            'chars':         int(out_df['char_count'].sum()),
            'secs':          round(time.perf_counter() - t0, 2),
            'error_message': '',
            'parsed_at':     datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        records[doc] = {
            'status':        'failed',
            'pages':         pd.NA,
            'chars':         pd.NA,
            'secs':          round(time.perf_counter() - t0, 2),
            'error_message': str(e)[:200],
            'parsed_at':     datetime.now(timezone.utc).isoformat(),
        }

    # checkpoint every 10 docs
    if len(todo) and (todo.index(src) + 1) % 10 == 0:
        pd.DataFrame([{'doc_name': k, **v} for k, v in records.items()]).to_csv(MANIFEST, index=False)

pd.DataFrame([{'doc_name': k, **v} for k, v in records.items()]).to_csv(MANIFEST, index=False)

# summary
summary = pd.Series([v['status'] for v in records.values()]).value_counts()
print('\n' + '=' * 40)
print('final status:')
print(summary)
