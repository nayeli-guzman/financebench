"""Score generated answers: LLM-as-judge correctness + objective numeric match.

Reads a runs/answers_*.jsonl produced by run_rag.py and produces per-question
grades plus an aggregate summary. Two complementary correctness signals:

  * numeric_match  - for questions whose gold answer is a number, extract the
    value from the model answer and compare within tolerance. Objective, no LLM,
    but only defined for numeric golds. Used to validate the judge.
  * judge_verdict  - a separate, larger local model (default qwen2.5:7b via
    Ollama, forced JSON) grades correct / partial / incorrect against the gold
    answer. Works for numeric AND open-ended questions. A different model from
    the generator to avoid self-grading bias (see docs/decisions.md).

Cross-tabs by question_type and by retrieval page_hit separate retrieval
failures from generation failures.

Outputs: runs/answers_eval_{tag}.jsonl (per-question) + .csv (summary).

Usage:
  python src/eval/eval_answers.py --answers runs/answers_qwen2.5-3b__bge-small__s256_o50.jsonl
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

RUNS = Path('runs')
SCALE = {'thousand': 1e3, 'million': 1e6, 'billion': 1e9, 'trillion': 1e12,
         'k': 1e3, 'm': 1e6, 'bn': 1e9, 'b': 1e9}

JUDGE_SYSTEM = (
    "You grade a financial question-answering system. You are given a question, "
    "the reference (correct) answer, and a candidate answer. Decide whether the "
    "candidate conveys the same key fact or figure as the reference.\n"
    "Rules: allow paraphrase, rounding, and extra correct detail. Numbers must "
    "match the reference (after unit/scale normalization). A refusal or a "
    "candidate missing the key fact is 'incorrect'. Partly-right or "
    "right-direction-but-imprecise is 'partial'.\n"
    'Respond with strict JSON only: {"verdict": "correct|partial|incorrect", '
    '"reason": "<short>"}'
)


def parse_number(text: str) -> float | None:
    """Best-effort: first monetary/numeric value with optional scale word."""
    if text is None:
        return None
    t = text.replace(',', '')
    m = re.search(r'\(?\$?\s*(-?\d+(?:\.\d+)?)\s*(thousand|million|billion|trillion|bn|k|m|b)?', t, re.I)
    if not m:
        return None
    val = float(m.group(1))
    if m.group(2):
        val *= SCALE[m.group(2).lower()]
    if '(' in text[:m.start() + 1] and ')' in text[m.end() - 1:]:
        val = -abs(val)
    return val


def numeric_match(gold: str, answer: str, rel_tol: float = 0.01) -> bool | None:
    """None if gold isn't numeric; else whether any number in `answer` matches."""
    g = parse_number(gold)
    if g is None:
        return None
    for token in re.findall(r'\(?\$?\s*-?\d[\d,]*(?:\.\d+)?\s*(?:thousand|million|billion|trillion|bn|k|m|b)?', answer, re.I):
        a = parse_number(token)
        if a is None:
            continue
        if abs(a - g) <= max(rel_tol * abs(g), 1e-6) or (g and abs(a - g) / abs(g) <= rel_tol):
            return True
    return False


class Judge:
    def __init__(self, model: str, host: str = 'http://localhost:11434'):
        self.model, self.host = model, host.rstrip('/')

    def grade(self, question: str, gold: str, answer: str) -> dict:
        user = (f"Question: {question}\nReference answer: {gold}\n"
                f"Candidate answer: {answer}")
        resp = requests.post(f'{self.host}/api/chat', json={
            'model': self.model,
            'messages': [{'role': 'system', 'content': JUDGE_SYSTEM},
                         {'role': 'user', 'content': user}],
            'stream': False, 'format': 'json',
            'options': {'temperature': 0.0, 'seed': 0, 'num_ctx': 4096},
        }, timeout=600)
        resp.raise_for_status()
        try:
            j = json.loads(resp.json()['message']['content'])
            v = str(j.get('verdict', 'incorrect')).lower()
            return {'verdict': v if v in ('correct', 'partial', 'incorrect') else 'incorrect',
                    'reason': str(j.get('reason', ''))[:200]}
        except (json.JSONDecodeError, KeyError):
            return {'verdict': 'incorrect', 'reason': 'unparseable judge output'}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--answers', required=True)
    ap.add_argument('--judge', default='qwen2.5:7b')
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.answers)]
    judge = Judge(args.judge)
    tag = Path(args.answers).stem.replace('answers_', '')
    print(f'scoring {len(rows)} answers | judge {args.judge}')

    out = []
    for r in tqdm(rows):
        nm = numeric_match(r['gold_answer'], r['answer'])
        jv = judge.grade(r['question'], r['gold_answer'], r['answer'])
        out.append({
            'financebench_id': r['financebench_id'],
            'question_type': r['question_type'],
            'page_hit': r['page_hit'],
            'refused': r['refused'],
            'has_citation': len(r['cited']) > 0,
            'cited_pages_valid': r['cited_pages_valid'],
            'numeric_match': nm,
            'verdict': jv['verdict'],
            'correct': jv['verdict'] == 'correct',
            'correct_or_partial': jv['verdict'] in ('correct', 'partial'),
            'reason': jv['reason'],
        })

    df = pd.DataFrame(out)
    df.to_json(RUNS / f'answers_eval_{tag}.jsonl', orient='records', lines=True)

    def block(sub: pd.DataFrame) -> dict:
        answered = sub[~sub['refused']]   # citations only meaningful when it answered
        d = {
            'n': len(sub),
            'accuracy': round(sub['correct'].mean(), 3),
            'acc_or_partial': round(sub['correct_or_partial'].mean(), 3),
            'refusal_rate': round(sub['refused'].mean(), 3),
            'citation_rate': round(answered['has_citation'].mean(), 3) if len(answered) else None,
            'cited_valid_rate': round(sub['cited_pages_valid'].mean(), 3),
        }
        nm = sub['numeric_match'].dropna()
        d['numeric_match_acc'] = round(nm.mean(), 3) if len(nm) else None
        return d

    summary = {'overall': block(df)}
    for qt, sub in df.groupby('question_type'):
        summary[qt] = block(sub)
    # retrieval-vs-generation cross-tab
    for hit, sub in df.groupby('page_hit'):
        summary[f'page_hit={hit}'] = {'n': len(sub), 'accuracy': round(sub['correct'].mean(), 3)}

    sdf = pd.DataFrame(summary).T
    sdf.to_csv(RUNS / f'answers_eval_{tag}_summary.csv')
    pd.set_option('display.width', 200, 'display.max_columns', 20)
    print('\n' + sdf.to_string())

    # agreement between the two correctness signals (validates the judge)
    both = df[df['numeric_match'].notna()]
    if len(both):
        agree = (both['numeric_match'] == both['correct']).mean()
        print(f"\njudge vs numeric-match agreement on {len(both)} numeric Qs: {agree:.1%}")
    print(f"\nsaved -> runs/answers_eval_{tag}.jsonl , runs/answers_eval_{tag}_summary.csv")


if __name__ == '__main__':
    main()
