import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

MANIFEST = Path('data/manifest.csv')
OUT_DIR  = Path('data/raw_pdfs')
OUT_DIR.mkdir(parents=True, exist_ok=True)

UA = 'nayeli.guzman@epita.fr'

session = requests.Session()
retry = Retry(total=3, backoff_factor=1.5,
              status_forcelist=[500, 502, 503, 504],
              allowed_methods=['GET'])
session.mount('https://', HTTPAdapter(max_retries=retry))
session.headers.update({'User-Agent': UA, 'Accept': 'application/pdf,*/*'})

df = pd.read_csv(MANIFEST)


for col in ('status', 'error_message', 'downloaded_at', 'doc_name', 'doc_link'):
    if col in df.columns:
        df[col] = df[col].astype('object')
for col in ('http_status', 'file_size_bytes'):
    if col in df.columns:
        df[col] = df[col].astype('Int64')  # nullable int

for i, row in tqdm(df.iterrows(), total=len(df)):
    doc, url = row['doc_name'], row['doc_link']
    out = OUT_DIR / f'{doc}.pdf'

    # skip 
    if out.exists() and out.stat().st_size > 50_000:
        df.at[i, 'status']          = 'downloaded'
        df.at[i, 'file_size_bytes'] = out.stat().st_size
        continue

    try:
        r = session.get(url, timeout=(10, 60), stream=True)
        df.at[i, 'http_status'] = r.status_code
        r.raise_for_status()
        with open(out, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1 << 15):
                f.write(chunk)
        df.at[i, 'status']          = 'downloaded'
        df.at[i, 'file_size_bytes'] = out.stat().st_size
        df.at[i, 'downloaded_at']   = datetime.now(timezone.utc).isoformat()
    except Exception as e:
        df.at[i, 'status']        = 'failed'
        df.at[i, 'error_message'] = str(e)[:200]

    time.sleep(0.15)
    if i % 20 == 0:
        df.to_csv(MANIFEST, index=False)

df.to_csv(MANIFEST, index=False)
print(df['status'].value_counts())