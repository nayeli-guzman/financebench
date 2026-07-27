"""Create a compact dense-vs-hybrid evidence retrieval comparison."""
import argparse
from pathlib import Path

import pandas as pd

RUNS = Path('runs')
K_LIST = [1, 3, 5, 10, 20]


def overall(path: str, name: str) -> dict:
    df = pd.read_csv(path, index_col=0)
    row = df.loc['overall']
    out = {'configuration': name}
    for k in K_LIST:
        for metric in ('page_hit', 'evidence_hit',
                       'evidence_token_recall', 'evidence_ngram_recall'):
            out[f'{metric}@{k}'] = row[f'{metric}@{k}']
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dense', required=True)
    ap.add_argument('--hybrid', required=True)
    args = ap.parse_args()

    out = pd.DataFrame([
        overall(args.dense, 'dense'),
        overall(args.hybrid, 'hybrid_dense0.9_bm250.1'),
    ])
    path = RUNS / 'evidence_retrieval_summary.csv'
    out.to_csv(path, index=False)
    display = ['configuration']
    for k in K_LIST:
        display += [f'page_hit@{k}', f'evidence_hit@{k}',
                    f'evidence_ngram_recall@{k}']
    print(out[display].round(3).to_string(index=False))
    print(f'\nsaved -> {path}')


if __name__ == '__main__':
    main()
