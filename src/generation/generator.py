"""Grounded answer generation with a local instruct LLM served by Ollama.

Takes a question and a list of retrieved context passages and produces an
answer that (a) uses only the provided context and (b) cites the passages it
relied on by number -- the citation requirement from the assignment. Generation
runs fully locally via the Ollama server (http://localhost:11434); we switched
to Ollama because transformers+MPS segfaults on long (~2.8k-token) RAG prompts
on this machine and CPU is too slow.

Prompted to reply with a fixed refusal string when the context lacks the
answer, so we can distinguish "retrieval failed" from "model hallucinated" in
error analysis.

Used as a library by the RAG pipeline and the answer-eval harness.
"""
import re

import requests
import yaml

REFUSAL = "I cannot answer from the provided context."

SYSTEM = (
    "You are a financial analyst assistant. Answer the question using ONLY the "
    "numbered context passages provided.\n"
    "Rules:\n"
    "- Place a citation in square brackets immediately after each claim, e.g. "
    "\"revenue was $8.2B [1]\". EVERY answer must contain at least one [n] "
    "citation naming the passage(s) you used.\n"
    "- If the answer is a numeric value, state it with units (USD millions, %, etc.).\n"
    "- Be concise: one or two sentences.\n"
    f"- If the context does not contain the answer, reply exactly: \"{REFUSAL}\" "
    "(no citation needed in that case)."
)


def format_context(contexts: list[dict]) -> str:
    """Render retrieved chunks as a numbered, source-tagged block."""
    lines = []
    for i, c in enumerate(contexts, start=1):
        tag = f"{c['doc_name']} p.{c['page_number']}"
        body = ' '.join(c['text'].split())
        lines.append(f"[{i}] ({tag}) {body}")
    return '\n'.join(lines)


class Generator:
    def __init__(self, config_path: str, host: str = 'http://localhost:11434'):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        self.host = host.rstrip('/')
        self.model = self.cfg['model']
        # num_ctx must exceed prompt tokens (~2.8k for k=10) + max_new_tokens,
        # else Ollama silently truncates the retrieved passages.
        self.options = {
            'temperature': self.cfg.get('temperature', 0.0),
            'num_predict': self.cfg.get('max_new_tokens', 384),
            'num_ctx': self.cfg.get('num_ctx', 8192),
            'seed': self.cfg.get('seed', 0),
        }

    def build_messages(self, question: str, contexts: list[dict]) -> list[dict]:
        user = (f"Context:\n{format_context(contexts)}\n\n"
                f"Question: {question}\n\nAnswer:")
        return [{'role': 'system', 'content': SYSTEM},
                {'role': 'user', 'content': user}]

    def answer(self, question: str, contexts: list[dict]) -> dict:
        messages = self.build_messages(question, contexts)
        resp = requests.post(
            f'{self.host}/api/chat',
            json={'model': self.model, 'messages': messages,
                  'stream': False, 'options': self.options},
            timeout=600,
        )
        resp.raise_for_status()
        text = resp.json()['message']['content'].strip()
        # handles [3], [1][2], and [1, 2] citation styles
        cited = sorted({int(n) for grp in re.findall(r'\[([\d,\s]+)\]', text)
                        for n in re.findall(r'\d+', grp)})
        return {
            'answer': text,
            'cited': cited,                       # 1-based context indices cited
            'refused': REFUSAL.lower() in text.lower(),
        }
