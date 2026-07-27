# Design decisions log

Running record of methodology choices, with the evidence and reasoning
behind each. Feeds into the *Methodology* and *Corpus Description*
sections of the final report.

---

## 1. Dataset choice

**Decision:** FinanceBench (patronus-ai/financebench, open-source 150-question subset).

**Reasoning:** Assignment recommends it as an industry-standard benchmark
for LLM evaluation on financial documents. The paired structure
(`financebench_document_information.jsonl` + `financebench_open_source.jsonl`
linked by `doc_name`) provides ground-truth Q&A with cited evidence spans,
enabling both retrieval and answer-quality evaluation without any manual
annotation.

---

## 2. Corpus acquisition

**Decision:** Automated download from `doc_link` for all 360 catalog
entries, with fallback to the FinanceBench repo's `/pdfs/` folder for
failures.

**Reasoning and observations:**
- The `doc_link` column is a mix of SEC EDGAR URLs (reliable) and
  company investor-relations URLs (unreliable — link rot, timeouts,
  anti-bot measures).
- First-pass automated download from `doc_link` succeeded for 263/360
  documents (~73%). The 97 failures split as:
  - **61 read timeouts** on company IR hosts (investor.activision.com,
    johnsonandjohnson.gcs-web.com, investors.lockheedmartin.com,
    pepsico.gcs-web.com, microsoft.gcs-web.com, investors.footlocker-inc.com)
  - **9 × 404s** on Adobe wrapper URLs of the form
    `www.adobe.com/pdf-page.html?pdfTarget=...`
  - **8 × 403s** on hosts rejecting anonymous requests
  - **0 failures** on `sec.gov`, confirming the User-Agent policy is not
    the cause
- A subtler failure mode: some hosts returned HTTP 200 with HTML error
  pages saved as `.pdf` files (silent failures). Detected via `%PDF`
  magic-byte verification; 19 such files were flagged and removed.
- All 97 failures were recovered by mirroring the FinanceBench
  maintainers' pre-committed `/pdfs/` folder via the GitHub Contents API.

**Final corpus:** 368 valid PDFs on disk (all `%PDF` verified), fully
covering the 150-question open-source eval set. The 8 additional PDFs
beyond the 360 catalog entries are extras present in the repo but not
listed in the catalog; they are retained for potential future use.

**Reproducibility:** Full acquisition ledger is in `data/manifest.csv`
with `doc_name`, `doc_link`, `status`, `http_status`, `file_size_bytes`,
`downloaded_at`. Final counts:
- `downloaded`           : 263 (from `doc_link`)
- `downloaded_fallback`  : 97  (from repo mirror)
- `failed`               : 0

**Report takeaway:** The unreliability of `doc_link` for enterprise
financial documents is itself an observation about corpus construction
for financial NLP — enterprise IR pages are not durable primary sources.

---

## 3. PDF parser choice

**Decision:** `pdfplumber` for all corpus parsing.

**Reasoning:** Ran `pypdf`, `pdfplumber`, and `PyMuPDF` (fitz) over four
representative samples:
- `APPLE_2022_10K` (recent large 10-K)
- `Pfizer_2023Q2_10Q` (quarterly, table-heavy)
- `PEPSICO_2015_10K` (older 10-K, pre-2018)
- `ULTABEAUTY_2023Q2_EARNINGS` (earnings release, table-dense)

**Decisive evidence — table row coherence.** On `APPLE_2022_10K` page 24
(Products and Services Performance table), the row for Mac 2022 revenue
was rendered as:

- `pdfplumber` : `"Mac (1)   40,177   14 %   35,190   23 %   28,622"`  (one line)
- `pypdf`      : `"Mac (1)   40,177   14 %   35,190   23 %   28,622"`  (one line)
- `pymupdf`    : `"Mac (1)"` / `"40,177"` / `"14 %"` / `"35,190"` / … (~7 lines)

For a RAG system answering numerical questions about financial tables,
row-coherent extraction is critical: a chunk boundary falling inside
`pymupdf`'s shredded output would separate row labels from their values
and make retrieval unreliable. This is exactly the failure mode that
causes hallucinated numbers even when the correct page is retrieved.

**Narrative-page sanity check.** All three parsers produced clean,
readable prose on `APPLE_2022_10K` page 10 (Risk Factors). Differences
were limited to trailing whitespace (removable in cleaning). No parser
was uniquely better on narrative-only pages.

**Why pdfplumber over pypdf** (both were table-coherent): pdfplumber is
built on `pdfminer.six` and has genuine layout awareness (bounding-box
detection, whitespace normalization), giving fewer cleanup artifacts.
The tradeoff is speed — see the smoke-test timing table in
`data/parser_smoke/`.

---

## 4. Open decisions (to be filled)

- [x] Text cleaning strategy (header/footer detection, ligature fixes) — see §5
- [x] Chunking strategy (size, overlap, structure-awareness) — see §6
- [x] Embedding model choice — see §7
- [x] Vector store (FAISS flat vs. HNSW; metadata storage) — see §7
- [x] Retrieval configuration (k and dense/BM25 fusion) — baseline in §8 and sweep in §11
- [x] Generation model + runtime — see §9
- [x] Evaluation metrics — retrieval hit@k (§8); answer quality: LLM-judge + numeric match (§10)

---

## 4. Full-corpus parsing outcome

Ran `pdfplumber` over all 368 PDFs. Output stored as one Parquet file
per document at `data/parsed/{doc_name}.parquet`, one row per page with
columns `(doc_name, page_number, text, char_count)`.

**Results:**
- 368/368 documents parsed successfully (0 failures)
- 54,120 total pages
- 178,532,301 total characters
- Wall time: 91 minutes (avg 15s per document)
- Avg chars per page: ~3,300 (consistent with well-extracted business prose)

**Warning noise.** pdfminer.six (via pdfplumber) emitted repeated
`FontBBox` warnings on many documents — malformed font metadata is
common in SEC filings and company IR PDFs. These are non-fatal;
extraction proceeded normally with default fallbacks.

**Shortest parses (< 5000 chars) audit.** Five documents flagged for
brevity — all are 8-K filings, which are inherently short by document
type (2–5 pages announcing specific material events). No scanned or
otherwise-broken documents in the corpus.

**Reproducibility:** Full parse ledger in `data/parse_manifest.csv`.

---

## 5. Text cleaning

**Decision:** A conservative, auditable cleaning pass applied once over the
parsed pages. `data/parsed/{doc}.parquet` -> `data/clean/{doc}.parquet`
(same one-row-per-page schema, cleaned `text`). Script:
`src/ingest/clean_pages.py`.

**Steps:**
1. **Unicode NFKC normalization** — folds ligatures (ﬁ→fi, ﬂ→fl),
   non-breaking spaces → regular spaces, and full-width/compatibility
   characters to canonical forms.
2. **Per-line whitespace** — strip line ends; collapse internal runs of
   spaces/tabs to a single space.
3. **Running header/footer removal** — a line is stripped only if it (a)
   sits in the top-3 or bottom-3 lines of a page, (b) is ≤ 60 chars, and
   (c) its digit-masked signature (`\d+`→`#`, lowercased) recurs on ≥ 50%
   of the document's pages. Documents with < 4 pages are exempt (protects
   short 8-Ks). This positional + frequency gate targets true running
   heads and page numbers, never table content.
4. **Blank-line collapse** — 2+ consecutive blank lines → one.

**Why conservative over aggressive:** the decisive risk for a financial
RAG is destroying table-row coherence (see §3). By constraining
header/footer removal to page *edges* and requiring cross-page recurrence,
mid-page table rows — where the numeric evidence lives — are provably
untouched. Verified on `APPLE_2022_10K` p41 (marketable-securities table)
and `NETFLIX_2023Q2_10Q` p20 (results-of-operations table): both are
byte-identical before/after.

**Outcome (full corpus, 368 docs / 54,120 pages, ~9s):**
- Only **0.28%** of corpus characters removed (mean 0.33%/doc, max 1.91%).
- Top removals are all legitimate boilerplate: Corning's
  `© 2015 Corning Incorporated. All rights reserved.` footer, Footlocker's
  `Third Quarter 2023 Form 10-Q Page 5` footer, Oracle's
  `Index to Financial Statements` header.
- 59/368 docs had no detectable running head and were left untouched.

**Known tradeoff:** the bare `#` signature (a page-edge line that is only a
number) also removes standalone numeric lines. In SEC filings such
edge-isolated numbers are page numbers, not data (data always appears in
multi-value rows), so the risk is negligible and bounded to page edges.

**Reproducibility:** every removed signature is logged per-doc in
`data/clean_manifest.csv` (`pct_removed`, `n_boilerplate`,
`boilerplate_sigs`).

---

## 6. Chunking

**Decision:** Config-driven recursive, page-bounded chunking.
`data/clean/{doc}.parquet` -> `data/chunks/{config.name}.parquet` (one row
per chunk). Script: `src/chunk/build_chunks.py`; baseline config
`configs/chunk_baseline.yaml`. This is the primary experimental knob, so the
strategy and parameters live entirely in YAML — the Phase-E sweep copies the
baseline file and varies only `chunk_size` / `chunk_overlap`.

**Baseline parameters:** `chunk_size=256` tokens, `chunk_overlap=50` (~20%),
`strategy=recursive`, `cross_page=false`, token ruler = the baseline embedder
`BAAI/bge-small-en-v1.5`.

**Three design choices and why:**
1. **Page-bounded (chunks never cross a page).** Each chunk therefore carries
   exactly one `page_number`, which maps directly onto the eval set's
   `evidence_page_num`. This buys a cheap, objective, LLM-free retrieval metric
   (hit@k / recall@k at the page level). Tradeoff: a table spanning a page
   break is split — accepted for the baseline; cross-page chunking is reserved
   as a Phase-E variant.
2. **Recursive natural-boundary splitting** (blank line -> newline -> sentence
   -> word, then a hard token split only as a last resort). No atom exceeds
   `chunk_size`, so single-line table rows are never cut mid-row — the same
   row-coherence property that motivated the parser choice (§3) is preserved
   through to the chunk.
3. **Size measured in TOKENS with the embedder's own tokenizer.** Guarantees a
   chunk never overflows the model's context window (silent tail truncation at
   embed time). Because the sweep's candidate embedders (MiniLM, bge-small,
   bge-base) all use near-identical BERT WordPiece vocabularies, token counts
   are comparable across models, keeping `chunk_size` a clean independent
   variable.

**Overlap** is implemented by backstepping whole atoms from a chunk's end until
>= `chunk_overlap` tokens are carried into the next chunk (verified: 52 shared
boundary tokens for a 50-token target — the +2 is the snap to the atom
boundary), with a guaranteed forward-progress guard.

**Outcome (baseline s256_o50, full corpus):**
- **211,671 chunks** from 368 docs; mean **575 chunks/doc**.
- tokens/chunk: mean 222, median 240, p95 255, max 256 (hard cap holds). The
  mean sitting below the median is the expected short-tail-chunk signature of
  page-bounded chunking. Build time ~5.5 min (one-off per config).

**Reproducibility:** `python src/chunk/build_chunks.py --config
configs/chunk_baseline.yaml`. Output columns: `chunk_id`
(`{doc}__p{page}__c{idx}`), `doc_name`, `page_number`, `chunk_index`, `text`,
`n_tokens`, `n_chars`.

---

## 7. Embedding + vector index

**Decision:** Baseline embedder `BAAI/bge-small-en-v1.5` (384-dim, 512-token
context), exact FAISS `IndexFlatIP` over unit-normalized vectors (= cosine).
Script: `src/retrieval/build_index.py`; config `configs/embed_baseline.yaml`;
output `data/index/{name}/` (`index.faiss` + `meta.parquet` aligned to index
order + `config.json`).

**Why bge-small baseline:** strong retrieval quality per parameter, 512-token
context (covers the full chunk-size sweep without truncation, unlike MiniLM's
256), fast enough to iterate. bge-base and MiniLM are reserved as sweep points.
bge-v1.5 query instruction ("Represent this sentence for searching relevant
passages: ") is prepended to queries only, never to passages.

**Why flat (exact) index over HNSW:** at 211,671 x 384 the flat index is only
325 MB and searches in milliseconds; exact search means approximation error
never confounds the experimental comparisons. HNSW is unnecessary at this scale.

**Hardware note (shapes the whole project).** Embedding is the binding
constraint on a 16 GB Apple-Silicon laptop (M-series, MPS). Sustained encode
throughput thermally throttles to ~110 chunks/s; the full corpus took 1,919 s
(~32 min). fp16 on MPS measured ~26% faster than fp32 (151 vs 120 chunks/s on a
warm sample) with negligible retrieval impact, so `fp16: true` is the default;
vectors are re-normalized in float32 (`faiss.normalize_L2`) after encoding to
keep cosine exact. This cost is why each sweep index is a deliberate,
budgeted run rather than a free knob.

**Corpus scope.** Full 368-document corpus indexed (not just the 84
eval-referenced docs) — the realistic "large heterogeneous closed collection"
setting, where retrieval must disambiguate among many near-duplicate filings.

**Reproducibility:** `python src/retrieval/build_index.py --config
configs/embed_baseline.yaml`.

---

## 8. Retrieval evaluation — baseline

**Page-number calibration (critical).** The dataset's `evidence_page_num` is
**0-indexed**; our pdfplumber `page_number` is **1-indexed**. Verified by
locating each question's `evidence_text` in the parsed pages: offset
`our_page - evidence_page_num == +1` in 23/25 locatable cases (the two outliers
are evidence strings that also occur in a table of contents). The scorer uses
`gold_page = evidence_page_num + 1`. Missing this would have silently zeroed the
page-level metric.

**Harness:** `src/eval/eval_retrieval.py` retrieves top-20 once per question and
scores every k in {1,3,5,10,20}: `doc_hit@k` (right document retrieved),
`page_hit@k` (a gold (doc,page) retrieved), `page_recall@k`, and page-level
`mrr`. No LLM involved, so it is cheap and objective. Results ->
`runs/retrieval_{index}.{jsonl,csv}`.

**Baseline result (bge-small, s256_o50, full corpus, 150 questions):**

| metric | k=1 | k=5 | k=10 | k=20 |
|---|---|---|---|---|
| page_hit | 0.10 | 0.22 | 0.31 | 0.41 |
| doc_hit  | 0.25 | 0.51 | 0.69 | 0.83 |

- **Dominant failure mode: document/year disambiguation.** Large gap between
  doc_hit (0.83 @20) and page_hit (0.41 @20), and doc_hit@1 only 0.25 — dense
  similarity captures topic but not the specific fiscal year among near-
  duplicate filings. Confirmed qualitatively (a "3M FY2018 capex" query
  retrieved 3M 2020/2019/2023 narrative pages, not the 2018 cash-flow
  statement).
- **By question type (page_hit@20):** metrics-generated 0.52 (rewards large k;
  answer sits on one statement page), novel-generated 0.44, domain-relevant
  0.26 (hardest; diffuse evidence).

**Implications for the experiment plan (levers this motivates):** hybrid
BM25+dense (keyword match recovers the year/company tokens dense misses),
stronger embedder (bge-base), metadata filtering by company/fiscal year,
larger k, and optionally a cross-encoder reranker. The weak, well-diagnosed
baseline is deliberate headroom for the experimental-analysis section.

---

## 9. Generation model + runtime

**Decision:** `Qwen2.5-3B-Instruct` (Q4_K_M) served locally by **Ollama**;
answers are grounded in the top-k retrieved chunks and must cite them.
Code: `src/generation/generator.py` (LLM), `src/generation/rag.py`
(retrieve->generate pipeline); config `configs/gen_baseline.yaml`.

**Runtime — why Ollama, not transformers.** The first implementation used
HuggingFace `transformers` on MPS. It works for short prompts (~7 tok/s) but
**segfaults (exit 139) on the full ~2,800-token RAG prompt** (k=10 contexts) --
a known class of MPS backend crash with long sequences on this stack (Python
3.14 / torch 2.13 / 16 GB). CPU generation was reliable but far too slow (a
64-token completion on a 2,841-token prompt did not finish in 9 min). Ollama
(llama.cpp + Metal, Q4_K_M quantization) is reliable and fast: **~11-12 s per
grounded answer** (~30 min for the full 150-question set), which also makes the
generation sweep feasible. Free and fully local, satisfying the assignment
constraint.

**Model choice.** Qwen2.5-3B-Instruct chosen for strong extraction/QA on
numeric financial text at a size that fits 16 GB comfortably; a 7B fp16 model
(~14 GB) would contend with the OS and embedder for memory. Other suggested
models (Llama-3.2-3B, Phi-3.5-mini, Mistral) are reserved as generation-sweep
points -- swapping is a one-line `ollama pull` + config change.

**Prompt / grounding.** System prompt restricts the model to the numbered
context passages, requires a bracketed `[n]` citation after each claim (the
assignment's evidence-reference requirement), and mandates a fixed refusal
string ("I cannot answer from the provided context.") when the answer is
absent. `num_ctx=8192` so the ~2.8k-token prompt is never silently truncated;
`temperature=0`, `seed=0` for reproducibility.

**Early qualitative check (3 questions, one per behavior):**
- retrieval-hit question -> correct grounded answer with a valid citation to the
  gold page;
- calculation question with no directly stated value -> honest refusal (no
  fabricated ratio);
- retrieval-miss question (wrong-year documents) -> recognized the value was
  absent, did NOT hallucinate the gold number.
This refuse-when-unsupported behavior is what lets error analysis separate
retrieval failures from generation failures.

---

## 10. End-to-end baseline result (answer quality)

**Setup:** index `bge-small__s256_o50` (full 368-doc corpus), generator
`qwen2.5:3b` via Ollama at k=10, judged by a **separate, larger** model
`qwen2.5:7b` (avoids self-grading bias) plus an objective numeric-match check.
Scripts: `src/generation/run_rag.py` (150 answers, 40.6 min, ~17 s/q, resumable)
-> `src/eval/eval_answers.py`. Artifacts: `runs/answers_qwen2.5-3b__bge-small__s256_o50.jsonl`,
`runs/answers_eval_qwen2.5-3b__bge-small__s256_o50{.jsonl,_summary.csv}`.

**Headline numbers (150 questions):**

| metric | value |
|---|---|
| accuracy (judge, all questions) | **0.120** |
| accuracy among answered (non-refused) | 0.279 |
| correct-or-partial among answered | 0.574 |
| refusal rate | 0.593 |
| citation compliance (among answered) | **0.852** |
| exact numeric match (metrics-generated) | **0.020** |
| evidence page retrieved (page_hit @k=10) | 0.31 |

**Finding 1 — retrieval is the dominant bottleneck.** Cross-tab on whether the
gold evidence page was retrieved:

| | n | refusal | accuracy | accuracy if answered |
|---|---|---|---|---|
| page NOT retrieved | 104 | 0.663 | 0.077 | 0.200 |
| page retrieved | 46 | 0.435 | **0.217** | **0.385** |

Retrieving the evidence page ~triples accuracy and cuts refusals by a third.

**Finding 2 — retrieval is necessary but NOT sufficient.** Even with the gold
page retrieved *and* the model electing to answer, accuracy is only **0.385**.
A second, independent bottleneck is the 3B generator's ability to convert
correct evidence into a correct answer. `metrics-generated` accuracy (judge) is
**0.14** -- many of those questions require multi-step arithmetic (e.g.
fixed-asset turnover = revenue / net PP&E), which a 3B model does poorly. This
decomposition is only visible because of the page_hit cross-tab.

*Caveat on the numeric-match metric.* The objective `numeric_match` reads 0.02
on `metrics-generated`, but manual inspection shows this is a **brittle lower
bound**, not the true rate: gold answers use an implied-millions convention
(`$12645.00`) while the model writes `$12,645 million`, so the parser's
scale-word handling produces false negatives (verified: a Boeing PP&E answer the
judge correctly marked *correct* was scored `numeric_match=False`). The LLM
judge's 0.14 is the trustworthy figure; `numeric_match` is retained only as a
noisy secondary signal and a judge cross-check, not a headline number.

**Finding 3 — failures are honest, not hallucinated.** 59% refusal, and refusal
rate tracks retrieval failure (0.663 vs 0.435). The system declines rather than
inventing figures; qualitatively it did not fabricate 3M's $1,577 capex when
retrieval missed. Low accuracy here reflects abstention, not fabrication --
a meaningfully different (and safer) failure profile for a financial assistant.

**Judge validation.** The LLM judge agrees with objective numeric matching on
**86.5%** of the 126 numeric-gold questions, supporting the LLM-as-judge
methodology rather than asserting it. (Caveat: agreement is partly inflated by
both signals concurring on negatives.)

**Metric note.** `citation_rate` is conditioned on non-refused answers --
refusals are instructed not to cite, so including them understates compliance
(0.347 unconditioned vs 0.852 conditioned).

**Experimental agenda this motivates (each with a directional hypothesis):**
1. **Hybrid BM25 + dense retrieval** -> BM25 matches the literal fiscal-year
   token ("2018") that dense embeddings blur across near-duplicate filings;
   predicted to raise page_hit and therefore everything downstream.
2. **Larger generator (`qwen2.5:7b`, already local)** -> predicted to raise the
   0.385 answered-with-evidence ceiling, especially on arithmetic.
3. **k sweep** -> free on the retrieval side; trades context length for recall.

---

## 11. Experiment 1 — BM25 + dense hybrid retrieval

**Hypothesis.** The dense baseline often retrieves the correct company but the
wrong fiscal year. BM25 should restore literal year-token matching (`2018`,
`FY2018`) and improve evidence-page retrieval when fused with dense search.

**Method.** Built a dependency-free Okapi BM25 inverted index over the same
211,671 chunks (`src/retrieval/build_bm25.py`) and fused its top-100 ranking
with the bge-small top-100 ranking using Reciprocal Rank Fusion (RRF,
`c=60`). The tokenizer splits mixed tokens such as `FY2018` into `fy2018`,
`fy`, and `2018`. Because equal fusion weights can overvalue a weak retriever,
`src/eval/eval_hybrid_sweep.py` evaluates dense weights
`{0, .5, .7, .8, .9, 1}` from one shared retrieval pass.

**Results (overall, 150 questions):**

| dense weight | BM25 weight | MRR | page_hit@5 | page_hit@10 | page_hit@20 |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 1.0 | 0.078 | 0.120 | 0.140 | 0.147 |
| 0.5 | 0.5 | 0.136 | 0.153 | 0.227 | 0.293 |
| 0.7 | 0.3 | 0.162 | 0.200 | 0.293 | 0.407 |
| 0.8 | 0.2 | **0.168** | 0.213 | 0.327 | 0.407 |
| 0.9 | 0.1 | 0.163 | **0.227** | **0.333** | 0.407 |
| 1.0 | 0.0 | 0.158 | 0.220 | 0.307 | 0.407 |

**Conclusion.** The original equal-weight hybrid hypothesis is **rejected**:
50/50 fusion reduces page_hit@10 from 0.307 to 0.227 because BM25 alone is
substantially weaker than dense retrieval. A small lexical contribution is
directionally useful: 90/10 fusion raises page_hit@10 to 0.333 (four net
questions) and 80/20 raises MRR to 0.168. However, the page_hit@10 gain is not
statistically significant on this sample (paired exact McNemar `p=0.125` for
90/10), and no fusion improves page_hit@20. The defensible claim is therefore
**a small, inconclusive early-rank gain**, not a major retrieval breakthrough.

**Interpretation.** Exact year matching helps a few questions, but unfiltered
BM25 is distracted by repeated filing vocabulary and numbers across the full
368-document corpus. A better next intervention is structured metadata
filtering or reranking, rather than giving lexical retrieval equal influence.

**Validity caveat.** Fusion weight was explored on the same 150-question set
used for reporting, so the best weight is exploratory rather than an unbiased
held-out estimate. This limitation must be stated in the report.

**Artifacts:** `data/index/bm25__s256_o50/`,
`runs/retrieval_hybrid_weight_sweep.csv`, and
`runs/retrieval_hybrid-d*__bge-small__s256_o50.jsonl`.

---

## 12. Experiment 2 — generator size (Qwen2.5 3B vs 7B)

**Question.** Once retrieval supplies the gold evidence page, does increasing
the local generator from 3B to 7B improve answer quality?

**Controlled design.** Chunking, bge-small retrieval, top-k (`k=10`), prompt,
context window, temperature, and seed are fixed. The 7B model receives the
**exact stored contexts** from the completed 3B run
(`src/generation/regenerate_from_contexts.py`), rather than repeating
retrieval. This guarantees that generator size is the only semantic independent
variable and avoids unified-memory contention between the BGE encoder and the
7B Ollama model.

The main comparison is conditioned on the 46 questions where the gold evidence
page was retrieved (19 metrics-generated, 16 novel-generated, 11
domain-relevant). This is intentional: retrieval misses cannot identify a
generator's ability to use correct evidence. The 7B output file contains 70
questions total, including 24 retrieval misses from the initial performance
diagnostic, but those misses are excluded from the main quality comparison.

**Objective behavior on the 46 matched questions:**

| generator | refusal | any citation | cites a gold page | mean words |
|---|---:|---:|---:|---:|
| Qwen2.5 3B | 0.435 | 0.587 | 0.370 | 32.2 |
| Qwen2.5 7B | 0.435 | **0.891** | **0.609** | 36.2 |

The larger model does not change abstention, but substantially improves source
use: +30.4 percentage points for citation presence and +23.9 points for citing
an actual gold page. It is also somewhat more verbose (median 31 vs 13 words).
Observed 7B generation was roughly 30--50 seconds/question after removing GPU
contention, versus 17.3 seconds/question for the full 3B baseline run, so the
quality behavior comes at about a 2--3x latency cost on this Mac.

**Blinded pairwise quality evaluation.** Both `qwen2.5:3b` and `qwen2.5:7b`
judged the same answer pairs against the gold answer, with deterministic
alternating candidate order:

| judge | prefers 7B | prefers 3B | tie | sign-test p |
|---|---:|---:|---:|---:|
| Qwen2.5 3B | 14 | 11 | 21 | 0.690 |
| Qwen2.5 7B | 25 | 14 | 7 | 0.108 |

Neither comparison reaches conventional significance. More importantly, exact
judge agreement is only 18/46 (39.1%), and both judges show position
sensitivity. Every pair was therefore judged again with A/B swapped. Requiring
the *same* preference in both orders leaves:

| judge | robustly 7B | robustly 3B | robust tie | order-unstable |
|---|---:|---:|---:|---:|
| Qwen2.5 3B | 3 | 4 | 14 | 25 (54.3%) |
| Qwen2.5 7B | 11 | 3 | 5 | 27 (58.7%) |

Both judges robustly agree on only 3 7B wins, 2 3B wins, and 4 ties. Thus the
pairwise LLM judges do **not** support a defensible claim that 7B is generally
more correct. This instability is itself an evaluation finding: small local
LLM judges are highly sensitive to candidate order and self/family style.

**Conclusion.** The hypothesis is **partially supported**. Qwen2.5 7B is
clearly better at citation behavior and grounding to the gold page, with equal
refusal behavior, but its general correctness advantage is inconclusive under
order-controlled pairwise judging. For this hardware, 3B remains the efficient
baseline; 7B is preferable when citation quality matters enough to justify
2--3x latency. The scientific report must separate the objective citation
improvement from the inconclusive subjective correctness comparison.

**Artifacts:** `configs/gen_qwen7b.yaml`,
`src/generation/regenerate_from_contexts.py`,
`src/eval/compare_generators.py`,
`src/eval/summarize_generator_experiment.py`,
`runs/generator_size_*_summary.csv`, and
`runs/generator_pairwise_*_page-hit*.jsonl`.

---

## 13. Evidence-text retrieval evaluation

**Motivation.** FinanceBench provides both `evidence_page_num` and the exact
`evidence_text`. Page-hit is objective and useful, but retrieving one chunk
from the correct page does not guarantee that the evidence-bearing part of that
page was retrieved. Conversely, PDF page-number mismatches or duplicated
exhibits can make page-hit report a miss even when the exact evidence text is
present. The assignment explicitly asks for comparison with `evidence_text`.

**Metric definition.** `src/eval/eval_retrieval.py` now normalizes text to
lowercase alphanumeric tokens and computes, at each
`k in {1,3,5,10,20}`:

- `evidence_token_recall@k`: mean unique-token recall over gold evidence spans;
- `evidence_ngram_recall@k`: mean unique-trigram recall;
- `evidence_hit@k`: any gold span reaches at least 30% trigram recall;
- `evidence_all_hit@k`: every gold span reaches that threshold.

For each span, overlap is computed only against retrieved chunks from the
span's gold document. This prevents unrelated filings with generic financial
vocabulary from creating false evidence matches. Multiple chunks at top-k are
unioned before scoring because a FinanceBench evidence span can cross a chunk
boundary.

**Threshold calibration.** Across all 189 gold evidence spans:

- union of all chunks on the gold page: median trigram recall 0.995, mean 0.950,
  minimum 0.346;
- best single gold-page chunk: median 0.689, mean 0.699;
- a 0.30 threshold detects 94.2% of best single gold-page chunks and 100% of
  gold-page unions.

Thus 0.30 is tolerant of chunk boundaries and extraction spacing while still
requiring contiguous phrase overlap rather than isolated common words.

**Results (overall, 150 questions):**

| retrieval | page_hit@10 | evidence_hit@10 | evidence 3-gram recall@10 | page_hit@20 | evidence_hit@20 | evidence 3-gram recall@20 |
|---|---:|---:|---:|---:|---:|---:|
| dense | 0.307 | 0.293 | 0.199 | 0.407 | **0.387** | **0.278** |
| hybrid 90/10 | **0.333** | **0.313** | **0.219** | 0.407 | 0.373 | 0.272 |

The hybrid gives a small early evidence gain at k=10 (three net questions;
paired exact McNemar `p=0.25`) but is slightly worse by k=20. This confirms the
earlier conclusion: weak lexical fusion can improve early ordering for a few
questions, but does not create a robust recall improvement.

**Manual validation of disagreements.**

- Four questions have `page_hit@10=1` but `evidence_hit@10=0`. In the Block
  FY2020 cash-flow question, the gold page appears only at rank 9 and the
  retrieved chunk contains only part of the long statement (trigram recall
  0.163), so page-hit overstates usable retrieval.
- Two questions have `page_hit@10=0` but `evidence_hit@10=1`. For Foot Locker's
  CEO question, FinanceBench labels page 2, while the matching exhibit text is
  extracted on page 29 of the same PDF. Evidence-text scoring correctly detects
  the content despite the page mismatch.
- The known 3M FY2018 capex failure has both page-hit and evidence recall zero.

**Conclusion.** Evidence-text metrics add information beyond page-hit and
satisfy the assignment's expected-evidence comparison requirement. The dense
baseline at k=10 retrieves sufficient exact evidence for only 29.3% of
questions, which strengthens the conclusion that retrieval is the primary
end-to-end bottleneck.

**Runtime note.** After a Python/Homebrew upgrade, the optimized
PyTorch/Sentence-Transformers query path segfaulted on this Mac. Query
evaluation now supports `--device cpu`, uses eager attention, and sets one
PyTorch CPU thread. This is numerically equivalent for the encoder and stable;
the 150-query evaluation still completes in seconds once imports finish.

**Artifacts:** enhanced `runs/retrieval_bge-small__s256_o50.{jsonl,csv}`,
enhanced `runs/retrieval_hybrid-d0p90__bge-small__s256_o50.{jsonl,csv}`, and
`runs/evidence_retrieval_summary.csv`.

---

## 14. Experiment 3 — controlled chunking pilot

**Question.** How do chunk size, overlap, and structure preservation affect
retrieval when the embedding model, documents, questions, and top-k are fixed?

**Pilot design.** Full-corpus re-embedding was unnecessarily expensive for an
initial ablation, so a deterministic stratified pilot sampled 30 FinanceBench
questions (seed 42): 10 each from `domain-relevant`, `metrics-generated`, and
`novel-generated`. Their 26 unique filings, spanning 20 companies, form the
candidate corpus for every variant. This is a paired comparison: each variant
is evaluated on the exact same questions and distractor documents.

The baseline's existing fp16 BGE vectors were filtered from the full index.
New vectors use the same BGE-small model, fp16 precision, normalization, query
instruction, and exact FAISS inner-product search. Index construction is
resumable in 1,024-vector shards. Exact unchanged passage text can reuse a
baseline vector; for the zero-overlap condition this avoided recomputing 4,349
of 11,775 vectors without changing the index numerically.

**Controlled variants:**

| label | strategy | size | overlap | chunks |
|---|---|---:|---:|---:|
| baseline | recursive natural boundaries | 256 | 50 | 13,797 |
| size384 | recursive natural boundaries | 384 | 50 | 8,886 |
| overlap0 | recursive natural boundaries | 256 | 0 | 11,775 |
| fixed windows | exact token windows | 256 | 50 | 12,849 |

The fixed-window condition deliberately ignores paragraphs, sentences, and
table-row boundaries. Comparing it with baseline isolates structure
preservation while size, overlap, page attribution, and corpus remain fixed.

**Results (30 paired questions):**

| variant | MRR | page hit@5 | page hit@10 | evidence hit@5 | evidence hit@10 | evidence 3-gram recall@10 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | **0.324** | 0.467 | 0.600 | 0.400 | 0.500 | 0.451 |
| size384 | 0.247 | 0.467 | 0.600 | 0.467 | **0.600** | **0.518** |
| overlap0 | 0.252 | 0.400 | 0.600 | 0.400 | 0.533 | 0.421 |
| fixed windows | **0.352** | **0.533** | 0.600 | **0.500** | 0.567 | 0.454 |

Paired bootstrap 95% confidence intervals show:

- size384 evidence hit@10 delta `+0.100`, CI `[-0.067, +0.267]`, while MRR
  delta is `-0.077`, CI `[-0.193, +0.021]`;
- removing overlap lowers MRR by `-0.072`, CI `[-0.153, -0.012]`, the only
  interval excluding zero; its page hit@10 does not change;
- fixed windows raise MRR by `+0.028`, CI `[-0.034, +0.094]`, and evidence
  hit@10 by `+0.067`, CI `[-0.067, +0.200]`.

**Interpretation.**

- Larger chunks trade early precision for context completeness: they create a
  36% smaller index and improve evidence coverage at top 10, but rank the first
  gold page later.
- A 50-token overlap has a real early-ranking benefit in this pilot. Removing
  it saves 15% of vectors but significantly reduces MRR.
- The hypothesis that natural-boundary preservation improves retrieval is not
  supported here. Fixed windows are competitive or slightly better on early
  retrieval, although their decoded text is less readable and can split table
  rows or sentences, which remains undesirable for generator input.

**Recommendation.** Keep recursive 256/50 as the primary configuration: it
retains readable structure, has the strongest established full-corpus
pipeline, and avoids the significant early-rank loss from zero overlap.
Recursive 384/50 is the most promising next configuration if the priority is
top-10 evidence completeness and a smaller index, but its apparent gain should
be confirmed on all 150 questions before replacing the baseline. Do not adopt
zero overlap. Fixed windows are a useful ablation, not the recommended
generator context format.

**Validity caveat.** The pilot's 26-document candidate corpus is easier than
the 368-document production corpus, and 30 questions give wide intervals.
Absolute scores must not be compared with the earlier full-corpus scores; only
paired differences among these four pilot variants are interpretable.

**Artifacts:** `data/financebench/financebench_pilot_30.jsonl`,
`configs/*_pilot.yaml`, `data/chunks/*_pilot.parquet`,
`data/index/*_pilot/`, `runs/retrieval_*_pilot.{jsonl,csv}`,
`runs/chunking_pilot_summary.csv`,
`src/experiments/select_pilot.py`,
`src/experiments/subset_index.py`, and
`src/experiments/summarize_chunking.py`.
