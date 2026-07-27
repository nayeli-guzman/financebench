"""Summarize paired retrieval results for the controlled chunking pilot."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = [
    'mrr',
    'page_hit@5',
    'page_hit@10',
    'evidence_hit@5',
    'evidence_hit@10',
    'evidence_ngram_recall@10',
]


def paired_interval(delta: np.ndarray, rng: np.random.Generator,
                    samples: int) -> tuple[float, float]:
    n = len(delta)
    means = np.empty(samples)
    for i in range(samples):
        means[i] = delta[rng.integers(0, n, n)].mean()
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', required=True)
    ap.add_argument('--variant', action='append', nargs=2,
                    metavar=('LABEL', 'JSONL'), required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--bootstrap-samples', type=int, default=5000)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    baseline = pd.read_json(args.baseline, lines=True).set_index('financebench_id')
    frames = [('baseline', baseline)] + [
        (label, pd.read_json(path, lines=True).set_index('financebench_id'))
        for label, path in args.variant
    ]
    rng = np.random.default_rng(args.seed)
    rows = []
    for label, frame in frames:
        common = baseline.index.intersection(frame.index)
        for metric in METRICS:
            value = frame.loc[common, metric].mean()
            delta = (
                frame.loc[common, metric].to_numpy()
                - baseline.loc[common, metric].to_numpy()
            )
            lo, hi = paired_interval(delta, rng, args.bootstrap_samples)
            rows.append({
                'variant': label,
                'metric': metric,
                'value': value,
                'delta_vs_baseline': delta.mean(),
                'delta_ci95_low': lo,
                'delta_ci95_high': hi,
                'n_questions': len(common),
            })
    out = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(out.round(3).to_string(index=False))
    print(f'\nsaved -> {args.output}')


if __name__ == '__main__':
    main()
