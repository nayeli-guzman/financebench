
import time
from pathlib import Path

import requests
from tqdm import tqdm

API = 'https://api.github.com/repos/patronus-ai/financebench/contents/pdfs'
OUT_DIR = Path('data/raw_pdfs')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. list what's in the repo
r = requests.get(API, params={'per_page': 500}, timeout=30)
r.raise_for_status()
entries = [e for e in r.json() if e['name'].endswith('.pdf')]
print(f'repo has {len(entries)} PDFs\n')

# 2. figure out what to skip
on_disk = {p.name for p in OUT_DIR.glob('*.pdf')}
todo = [e for e in entries if e['name'] not in on_disk]
print(f'{len(on_disk)} already on disk, {len(todo)} to download\n')

# 3. download the rest
session = requests.Session()
session.headers.update({'User-Agent': 'financebench-mirror research/edu'})

ok, bad = 0, []
for entry in tqdm(todo):
    name = entry['name']
    url  = entry['download_url']  # points at raw.githubusercontent.com
    out  = OUT_DIR / name
    try:
        resp = session.get(url, timeout=60)
        if resp.status_code == 200 and resp.content[:4] == b'%PDF':
            out.write_bytes(resp.content)
            ok += 1
        else:
            bad.append((name, f'status={resp.status_code}, magic={resp.content[:4]!r}'))
    except Exception as e:
        bad.append((name, str(e)[:120]))
    time.sleep(0.05)  # pause

print(f'\ndownloaded : {ok}')
print(f'failed     : {len(bad)}')
for name, reason in bad:
    print(f'  - {name}  ({reason})')
