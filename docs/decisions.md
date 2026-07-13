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

- [ ] Text cleaning strategy (header/footer detection, ligature fixes)
- [ ] Chunking strategy (size, overlap, structure-awareness)
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
