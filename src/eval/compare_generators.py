"""Blinded pairwise comparison of two generators on matched RAG contexts.

Only rows present in both files are compared. By default, the main analysis is
conditioned on `page_hit=True`, which isolates generation quality: both models
receive identical contexts containing the gold evidence page.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

RUNS = Path('runs')

SYSTEM = (
    "You are a strict financial QA evaluator. Compare two candidate answers to "
    "the same question using the reference answer. Prefer the candidate that "
    "states the correct key fact or figure, performs required arithmetic "
    "correctly, stays grounded, and answers concisely. A refusal loses to a "
    "correct supported answer. If both are equally correct or equally wrong, "
    "choose tie. Ignore whether the candidate is labeled A or B. "
    'Return strict JSON only: {"winner":"A|B|tie","reason":"<short>"}'
)


def load(path: str) -> dict[str, dict]:
    return {row['financebench_id']: row
            for row in (json.loads(line) for line in open(path))}


class PairJudge:
    def __init__(self, model: str, host: str = 'http://localhost:11434'):
        self.model = model
        self.host = host.rstrip('/')

    def compare(self, question: str, gold: str, answer_a: str,
                answer_b: str) -> dict:
        user = (
            f"Question: {question}\nReference answer: {gold}\n"
            f"Candidate A: {answer_a}\nCandidate B: {answer_b}"
        )
        response = requests.post(f'{self.host}/api/chat', json={
            'model': self.model,
            'messages': [{'role': 'system', 'content': SYSTEM},
                         {'role': 'user', 'content': user}],
            'stream': False,
            'format': 'json',
            'options': {'temperature': 0.0, 'seed': 0, 'num_ctx': 4096},
        }, timeout=600)
        response.raise_for_status()
        try:
            parsed = json.loads(response.json()['message']['content'])
            winner = str(parsed.get('winner', 'tie')).strip()
            if winner.lower() == 'tie':
                winner = 'tie'
            elif winner.upper() in ('A', 'B'):
                winner = winner.upper()
            else:
                winner = 'tie'
            return {'winner': winner,
                    'reason': str(parsed.get('reason', ''))[:250]}
        except (json.JSONDecodeError, KeyError):
            return {'winner': 'tie', 'reason': 'unparseable judge output'}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--answers-3b', required=True)
    ap.add_argument('--answers-7b', required=True)
    ap.add_argument('--judge', required=True)
    ap.add_argument('--all-common', action='store_true',
                    help='include retrieval misses; default is page_hit only')
    ap.add_argument('--swap-order', action='store_true',
                    help='invert the deterministic A/B order for bias control')
    args = ap.parse_args()

    answers_3b = load(args.answers_3b)
    answers_7b = load(args.answers_7b)
    ids = sorted(set(answers_3b) & set(answers_7b))
    if not args.all_common:
        ids = [fid for fid in ids if answers_3b[fid]['page_hit']]

    judge_tag = args.judge.replace(':', '-')
    scope = 'all-common' if args.all_common else 'page-hit'
    order_tag = '-swapped' if args.swap_order else ''
    out_path = RUNS / f'generator_pairwise_{judge_tag}_{scope}{order_tag}.jsonl'
    done = load(str(out_path)) if out_path.exists() else {}
    todo = [fid for fid in ids if fid not in done]
    print(f'judging {len(todo)} pairs ({len(done)} already done) | '
          f'judge={args.judge} | scope={scope}')

    judge = PairJudge(args.judge)
    with open(out_path, 'a' if done else 'w') as out:
        for fid in tqdm(todo):
            r3, r7 = answers_3b[fid], answers_7b[fid]
            # Stable alternating order prevents a systematic A/B position bias.
            seven_is_a = int(hashlib.sha256(fid.encode()).hexdigest(), 16) % 2 == 0
            if args.swap_order:
                seven_is_a = not seven_is_a
            answer_a = r7['answer'] if seven_is_a else r3['answer']
            answer_b = r3['answer'] if seven_is_a else r7['answer']
            result = judge.compare(r3['question'], r3['gold_answer'],
                                   answer_a, answer_b)
            if result['winner'] == 'tie':
                preferred = 'tie'
            elif (result['winner'] == 'A') == seven_is_a:
                preferred = '7b'
            else:
                preferred = '3b'
            row = {
                'financebench_id': fid,
                'question_type': r3['question_type'],
                'page_hit': r3['page_hit'],
                'judge': args.judge,
                'seven_is_a': seven_is_a,
                'preferred': preferred,
                'reason': result['reason'],
            }
            out.write(json.dumps(row) + '\n')
            out.flush()

    rows = [json.loads(line) for line in open(out_path)]
    df = pd.DataFrame(rows)
    print('\n=== preference ===')
    print(df['preferred'].value_counts().reindex(['7b', '3b', 'tie'],
                                                 fill_value=0).to_string())
    print('\n=== by question type ===')
    print(pd.crosstab(df['question_type'], df['preferred'])
          .reindex(columns=['7b', '3b', 'tie'], fill_value=0).to_string())
    print(f'\nsaved -> {out_path}')


if __name__ == '__main__':
    main()
