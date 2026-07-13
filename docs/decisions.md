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
- [ ] Embedding model choice
- [ ] Vector store (FAISS flat vs. HNSW; metadata storage)
- [ ] Retrieval configuration (k, hybrid on/off, reranker on/off)
- [ ] Generation model choice
- [ ] Evaluation metrics (RAGAS, LLM-as-judge, retrieval hit@k)

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
