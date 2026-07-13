import pandas as pd
from pathlib import Path

docs = pd.read_json('data/financebench/financebench_document_information.jsonl', lines=True)
manifest = docs[['doc_name', 'doc_link']].drop_duplicates('doc_name').copy()
manifest['status']          = 'pending'
manifest['http_status']     = pd.NA
manifest['file_size_bytes'] = pd.NA
manifest['error_message']   = ''
manifest['downloaded_at']   = pd.NaT

manifest.to_csv('data/manifest.csv', index=False)
print(f'wrote {len(manifest)} rows to data/manifest.csv')
