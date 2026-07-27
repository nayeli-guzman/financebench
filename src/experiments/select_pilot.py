"""Create a deterministic, balanced FinanceBench pilot set.

The open-source set contains 50 questions for each of three question types.
Sampling 20% within each type produces 30 questions: 10 per type.
"""
import argparse

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--fraction', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    df = pd.read_json(args.input, lines=True)
    pilot = (
        df.groupby('question_type', group_keys=False)
        .sample(frac=args.fraction, random_state=args.seed)
        .sort_index()
    )
    pilot.to_json(args.output, orient='records', lines=True, force_ascii=False)
    print(
        f'wrote {len(pilot)} questions, {pilot.doc_name.nunique()} documents, '
        f'{pilot.company.nunique()} companies -> {args.output}'
    )
    print(pilot.question_type.value_counts().sort_index().to_string())


if __name__ == '__main__':
    main()
