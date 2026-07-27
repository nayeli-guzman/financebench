"""Summarize the matched 3B-vs-7B generator experiment."""
import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path

import pandas as pd

RUNS = Path('runs')


def load(path: str) -> dict[str, dict]:
    return {row['financebench_id']: row
            for row in (json.loads(line) for line in open(path))}


def sign_test(wins_a: int, wins_b: int) -> float:
    n = wins_a + wins_b
    if n == 0:
        return 1.0
    low = min(wins_a, wins_b)
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(low + 1)) / 2 ** n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--answers-3b', required=True)
    ap.add_argument('--answers-7b', required=True)
    ap.add_argument('--judge-3b', required=True)
    ap.add_argument('--judge-7b', required=True)
    ap.add_argument('--judge-3b-swapped', required=True)
    ap.add_argument('--judge-7b-swapped', required=True)
    args = ap.parse_args()

    a3, a7 = load(args.answers_3b), load(args.answers_7b)
    ids = sorted(i for i in set(a3) & set(a7) if a3[i]['page_hit'])
    if not ids:
        raise SystemExit('no matched page-hit rows')

    operational = []
    for name, answers in [('qwen2.5:3b', a3), ('qwen2.5:7b', a7)]:
        operational.append({
            'model': name,
            'n': len(ids),
            'refusal_rate': sum(answers[i]['refused'] for i in ids) / len(ids),
            'citation_rate': sum(bool(answers[i]['cited']) for i in ids) / len(ids),
            'gold_page_citation_rate':
                sum(answers[i]['cited_pages_valid'] for i in ids) / len(ids),
            'mean_answer_words':
                statistics.mean(len(answers[i]['answer'].split()) for i in ids),
            'median_answer_words':
                statistics.median(len(answers[i]['answer'].split()) for i in ids),
        })
    operational_df = pd.DataFrame(operational)
    operational_df.to_csv(RUNS / 'generator_size_operational_summary.csv',
                          index=False)

    j3, j7 = load(args.judge_3b), load(args.judge_7b)
    j3s, j7s = load(args.judge_3b_swapped), load(args.judge_7b_swapped)
    judge_rows = []
    for label, judged in [('qwen2.5:3b', j3), ('qwen2.5:7b', j7)]:
        counts = Counter(judged[i]['preferred'] for i in ids)
        judge_rows.append({
            'judge': label,
            'n': len(ids),
            'prefers_7b': counts['7b'],
            'prefers_3b': counts['3b'],
            'ties': counts['tie'],
            'prefers_7b_rate': counts['7b'] / len(ids),
            'prefers_3b_rate': counts['3b'] / len(ids),
            'tie_rate': counts['tie'] / len(ids),
            'two_sided_sign_test_p':
                sign_test(counts['7b'], counts['3b']),
        })
    judge_df = pd.DataFrame(judge_rows)
    judge_df.to_csv(RUNS / 'generator_size_pairwise_summary.csv', index=False)

    agreement = sum(j3[i]['preferred'] == j7[i]['preferred']
                    for i in ids) / len(ids)
    robust_rows = []
    robust_by_judge = {}
    for label, original, swapped in [
        ('qwen2.5:3b', j3, j3s),
        ('qwen2.5:7b', j7, j7s),
    ]:
        robust = {
            i: original[i]['preferred']
            if original[i]['preferred'] == swapped[i]['preferred']
            else 'unstable'
            for i in ids
        }
        robust_by_judge[label] = robust
        counts = Counter(robust.values())
        robust_rows.append({
            'judge': label,
            'n': len(ids),
            'robust_7b': counts['7b'],
            'robust_3b': counts['3b'],
            'robust_tie': counts['tie'],
            'order_unstable': counts['unstable'],
            'order_unstable_rate': counts['unstable'] / len(ids),
        })
    robust_df = pd.DataFrame(robust_rows)
    robust_df.to_csv(RUNS / 'generator_size_order_robust_summary.csv',
                     index=False)

    both_robust = Counter()
    for i in ids:
        p3 = robust_by_judge['qwen2.5:3b'][i]
        p7 = robust_by_judge['qwen2.5:7b'][i]
        both_robust[p3 if p3 == p7 else 'not_agree'] += 1

    print('=== objective behavior on matched page-hit questions ===')
    print(operational_df.round(3).to_string(index=False))
    print('\n=== blinded pairwise preferences ===')
    print(judge_df.round(3).to_string(index=False))
    print(f'\nexact judge agreement: {agreement:.1%} ({round(agreement * len(ids))}/{len(ids)})')
    print('\n=== order-robust preferences (same result in both A/B orders) ===')
    print(robust_df.round(3).to_string(index=False))
    print(f'\nboth judges robustly agree: {dict(both_robust)}')
    print('\nsaved -> runs/generator_size_*_summary.csv')


if __name__ == '__main__':
    main()
