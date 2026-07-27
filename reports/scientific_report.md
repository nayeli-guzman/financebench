# Abstract

This project designs, implements, and evaluates a complete retrieval-augmented generation (RAG) system for closed-corpus financial question answering. The system uses the 150-question open-source FinanceBench benchmark and a local collection of 368 corporate filings. A reproducible ingestion pipeline downloads the documents, extracts 54,120 pages with pdfplumber, removes recurring headers and footers conservatively, and produces 211,671 page-bounded chunks. Passages are represented by 384-dimensional BGE-small embeddings and searched with an exact FAISS cosine index; a BM25 index supports an exploratory hybrid alternative. Retrieved passages are passed to locally served Qwen2.5 3B or 7B models under a prompt that requires evidence citations and abstention when context is insufficient.

The dense baseline retrieves a gold evidence page for 30.7% of questions at k=10 and sufficient exact evidence text for 29.3%. A 90% dense / 10% BM25 fusion raises these values to 33.3% and 31.3%, respectively, but the paired improvement is not statistically significant. The 3B end-to-end system obtains 12.0% judged-correct answers and 24.0% correct-or-partial answers; correctness is 21.7% when a gold page is retrieved and 7.7% otherwise. Controlled experiments show that removing chunk overlap significantly harms early ranking, 384-token chunks improve top-10 evidence coverage while weakening early precision, and Qwen2.5 7B improves citation behavior without a robustly demonstrated correctness advantage. The principal conclusion is that retrieval coverage, especially document and fiscal-year disambiguation in long financial reports, is the system's dominant bottleneck.

# 1. Introduction

Retrieval-augmented generation combines a parametric language model with an external, inspectable document memory [1]. Instead of relying only on facts stored in model parameters, a RAG system searches a closed corpus, places selected passages in the model prompt, and asks the model to answer from those passages. This design is attractive for financial question answering because source filings are long, frequently updated, numerically dense, and subject to audit requirements. An answer is more useful when it can be connected to a specific filing and page.

Financial reports are also unusually difficult retrieval targets. Annual and quarterly reports from the same company contain repeated vocabulary, similar statements, recurring line items, and overlapping fiscal years. Important evidence often appears in tables where extraction order matters, while apparently simple questions can require unit normalization or arithmetic. FinanceBench was created to test these conditions through questions paired with human-annotated answers and evidence strings [2].

This work addresses four research questions. RQ1 asks how financial PDFs can be converted into a retrievable corpus without destroying table-row structure. RQ2 asks how retrieval depth, sparse fusion, chunk size, overlap, and structure preservation affect evidence retrieval. RQ3 asks whether increasing a local generator from Qwen2.5 3B to 7B improves grounded answer quality. RQ4 asks which stage of the pipeline accounts for most end-to-end failures.

The project contributes a reproducible local pipeline, page-level provenance for every chunk, exact evidence-text evaluation in addition to page-hit metrics, controlled retrieval and generation experiments, and an error analysis that separates retrieval failure from generation failure. No paid or remote model API is required.

[[FIGURE:figure_1_pipeline.png|Figure 1. End-to-end system architecture. Every generated answer retains the retrieved document, page, chunk identifier, similarity score, and citation index used for evaluation.]]

# 2. Corpus Description

## 2.1 FinanceBench data

FinanceBench contains 10,231 financial question-answer-evidence triplets; its public repository provides an open-source subset of 150 questions [2]. The subset used here is balanced across three construction categories: 50 metrics-generated, 50 domain-relevant, and 50 novel-generated questions. Each record includes a question, gold answer, document name, and one or more evidence objects with source text and a zero-indexed page number. The accompanying catalog links document names to PDF URLs.

The evaluation questions reference 84 unique documents. The retrieval corpus is intentionally broader: all 368 valid PDFs available from the FinanceBench catalog and repository mirror are retained. This creates realistic distractors, including filings from the same company in different years.

| Corpus property | Value |
|---|---:|
| Catalog entries | 360 |
| Valid PDFs retained | 368 |
| FinanceBench evaluation questions | 150 |
| Question types | 3 x 50 |
| Unique documents referenced by evaluation | 84 |
| Parsed pages | 54,120 |
| Extracted characters before cleaning | 178,532,301 |
| Baseline chunks | 211,671 |

## 2.2 Acquisition and integrity

The primary downloader attempted every catalog URL with retries, timeout handling, a declared user agent, and checkpointed status metadata. It recovered 263 of 360 catalog documents directly. The remaining 97 links failed because of timeouts, anti-bot responses, obsolete wrapper URLs, or link rot. All failures were recovered from the FinanceBench repository mirror. Files returning HTTP 200 but HTML rather than PDF content were detected through the `%PDF` magic bytes instead of trusting file extensions.

The resulting corpus contains 368 valid PDFs: the 360 catalog documents plus eight additional PDFs present in the official repository. Acquisition status, HTTP response, file size, timestamp, and error details are retained in `data/manifest.csv`.

## 2.3 Financial layout characteristics

The collection includes 10-K, 10-Q, 8-K, earnings, and annual-report documents. Pages mix narrative prose with statements, footnotes, multi-column layouts, and numerical tables. A parser comparison on four representative documents showed that pdfplumber and pypdf kept important financial rows on one line, while PyMuPDF frequently separated row labels and values into different lines. Pdfplumber was selected because preserving row coherence reduces the chance that chunking separates a line-item name from its values.

No separate optical character recognition stage was required for the downloaded corpus. However, the implementation does not explicitly support future scanned documents, and extracted tables remain textual rows rather than formal relational tables.

# 3. Methodology

## 3.1 Page extraction and cleaning

Each PDF is parsed independently with pdfplumber. The output schema contains `doc_name`, one-indexed `page_number`, extracted `text`, and character count. Parsing is resumable and records success, page count, character count, runtime, and error details in a manifest. All 368 documents parsed successfully.

Cleaning is deliberately conservative. Unicode NFKC normalization repairs compatibility characters and ligatures; leading, trailing, and repeated internal whitespace is normalized; and multiple blank lines are collapsed. A potential running header or footer is removed only when it occurs within the first or last three lines, contains at most 60 characters, and its digit-masked signature appears on at least 50% of the document's pages. Documents shorter than four pages are exempt. This procedure removed 0.279% of corpus characters while preserving table rows in manual checks.

## 3.2 Chunking and metadata

The baseline uses recursive, page-bounded chunks of at most 256 BGE tokenizer tokens with a 50-token overlap. Text is divided in priority order at blank lines, line boundaries, sentence boundaries, and spaces; pieces are then greedily packed. Page boundaries are never crossed, so every chunk maps to one unambiguous page. Metadata stores a globally unique chunk ID, document name, page number, within-document chunk index, text, token count, and character count.

Page-bounded chunks support objective page-level evaluation and readable citations. Their main disadvantage is that a concept crossing a PDF page cannot appear in one chunk. Overlap protects evidence near chunk boundaries but increases index size and creates near-duplicate candidates.

## 3.3 Dense vector retrieval

Passages are embedded with `BAAI/bge-small-en-v1.5`, a 33-million-parameter English embedding model that produces 384-dimensional vectors. Sentence-embedding architectures make semantic similarity search practical by encoding queries and passages independently [3]. BGE's recommended retrieval instruction is prepended to queries only. Passage and query vectors are normalized.

Vectors are stored in `faiss.IndexFlatIP`. For unit-normalized vectors, inner product equals cosine similarity. The flat index performs exact search, so approximate-nearest-neighbor error cannot confound the experiments. FAISS provides efficient vector indexing and search primitives [4]. Each FAISS row is positionally aligned with a Parquet metadata row.

## 3.4 Sparse and hybrid retrieval

An in-project Okapi BM25 implementation provides lexical retrieval [5]. Its tokenizer emits complete alphanumeric tokens and letter/digit sub-runs, allowing `FY2018` to match both `fy2018` and `2018`. Dense and BM25 rankings are combined with weighted reciprocal-rank fusion. Six dense weights from 0 to 1 were explored. This hybrid experiment targets a known dense-retrieval weakness: literal fiscal years can be blurred across semantically similar filings.

## 3.5 Grounded answer generation

The generator is a locally executable Qwen2.5 instruction model served by Ollama. Qwen2.5 is available in multiple open-weight sizes and was designed for instruction following, long text, and structured-data tasks [6]. The baseline uses the 3B model with temperature 0, a fixed seed, an 8,192-token context window, and a maximum of 384 generated tokens.

The top 10 passages are formatted with numeric labels, document names, and page numbers. The system prompt requires the model to use only the supplied context, place a citation such as `[3]` after each claim, include units for numeric answers, and return a fixed refusal sentence if the answer is absent. The output parser records cited passage numbers and refusal behavior. A 7B configuration changes only model size.

## 3.6 Evaluation metrics

Retrieval is evaluated on all 150 FinanceBench questions at k in {1, 3, 5, 10, 20}. Document hit indicates whether any result belongs to a gold document. Page hit requires an exact gold document-page pair. Page recall measures the fraction of all gold pages found. Mean reciprocal rank (MRR) averages the reciprocal rank of the first gold page, assigning zero when no page is found in the evaluated pool.

Page hit alone can overstate useful retrieval when the returned chunk is on the correct page but does not contain the evidence. Therefore, each gold evidence span is normalized to lowercase alphanumeric tokens. Unique-token recall and unique 3-gram recall are measured against the union of retrieved chunks from the gold document. Evidence hit is one when at least one evidence span obtains 30% 3-gram recall; evidence-all-hit requires every span to meet the threshold. Calibration against all gold-page chunks showed that this threshold tolerates extraction spacing and chunk boundaries while still requiring contiguous phrase overlap.

Generated answers are evaluated with two complementary signals. A deterministic parser compares numeric values after unit normalization and one-percent relative tolerance. A separate local Qwen2.5 7B judge classifies each answer as correct, partial, or incorrect relative to the gold answer. Citation presence, citation to a gold page, refusal rate, and answer length are also recorded. The generator-size experiment uses blinded A/B pairwise judging with both 3B and 7B judges, deterministic alternating candidate order, and a complete swapped-order repetition to expose position sensitivity.

# 4. Experimental Setup

## 4.1 Baseline configuration

| Component | Baseline setting |
|---|---|
| Parser | pdfplumber, one row per page |
| Cleaning | NFKC, whitespace, recurring edge header/footer removal |
| Chunking | Recursive, page-bounded, 256 tokens, 50-token overlap |
| Embedding | BAAI/bge-small-en-v1.5, 384 dimensions, normalized |
| Dense index | FAISS IndexFlatIP |
| Retrieval | Dense top 10 |
| Generator | Qwen2.5 3B through Ollama |
| Decoding | Temperature 0, seed 0, 384-token maximum |
| Context window | 8,192 tokens |
| Evaluation set | FinanceBench open-source, 150 questions |

The full pipeline was executed on an Apple Silicon Mac with Python 3.14.6, PyTorch 2.13, Sentence Transformers 5.6, FAISS 1.14, and Ollama 0.32. Full parsing took approximately 91 minutes, baseline embedding 32 minutes, and 3B generation 17.3 seconds per question. Query evaluation was performed on CPU with eager attention for stability; this does not change the encoder's numerical objective.

## 4.2 Controlled experiments

Retrieval depth is evaluated from a single top-20 retrieval pass. The hybrid sweep holds chunks, embedding model, questions, and candidate pool fixed while varying only dense/BM25 fusion weight.

The chunking ablation uses a deterministic, balanced pilot of 30 questions: 10 from each FinanceBench question type. Their 26 unique source documents form the candidate corpus for all four configurations. The baseline is recursive 256/50. Three alternatives independently change chunk size to 384, remove overlap, or replace natural-boundary splitting with exact token windows. Because the pilot corpus is easier than the 368-document corpus, only paired differences among pilot variants are interpreted.

The generator-size experiment reuses the exact contexts stored by the 3B run. The main analysis is conditioned on the 46 questions where a gold page was retrieved. This prevents retrieval variability from being attributed to the generator.

## 4.3 Statistical analysis

Paired exact McNemar tests assess binary retrieval differences on the same questions. Chunking differences use 5,000-sample paired bootstrap confidence intervals. Generator pairwise preferences use two-sided sign tests after excluding ties. Because hybrid weights were selected on the same 150 questions used for reporting, the weight sweep is exploratory rather than an unbiased held-out model selection procedure.

# 5. Results

## 5.1 Dense and hybrid retrieval

Dense retrieval improves steadily with k but remains the largest system constraint. Page hit rises from 10.0% at k=1 to 30.7% at k=10 and 40.7% at k=20. Evidence hit closely follows page hit, reaching 29.3% at k=10 and 38.7% at k=20. Thus a correct page usually contains sufficient matching evidence, but a substantial majority of questions still lack gold evidence in the retrieved pool.

| Retrieval | MRR | Page hit@5 | Page hit@10 | Page hit@20 | Evidence hit@10 | Evidence 3-gram recall@10 |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 0.158 | 0.220 | 0.307 | 0.407 | 0.293 | 0.199 |
| Hybrid 90/10 | 0.163 | 0.227 | 0.333 | 0.407 | 0.313 | 0.219 |

The 90% dense / 10% BM25 fusion adds four page hits at k=10 over dense retrieval, but the paired exact McNemar p-value is 0.125. Evidence hit adds three net questions with p=0.25. Neither configuration improves page hit at k=20. Equal-weight fusion is substantially worse because BM25 alone is weak on semantic financial questions.

[[FIGURE:figure_2_retrieval.png|Figure 2. Full-corpus retrieval performance across k. Hybrid fusion produces a small early improvement, while both methods converge by k=20.]]

## 5.2 End-to-end answer quality

The Qwen2.5 3B baseline produces 18 correct answers out of 150 and 36 answers classified as correct or partial. It refuses 89 questions. Sixty-five outputs contain at least one citation, but only 17 cite a passage on a gold evidence page.

| Group | Questions | Correct | Correct or partial | Refusal rate |
|---|---:|---:|---:|---:|
| Overall | 150 | 0.120 | 0.240 | 0.593 |
| Domain-relevant | 50 | 0.120 | 0.260 | 0.600 |
| Metrics-generated | 50 | 0.140 | 0.160 | 0.600 |
| Novel-generated | 50 | 0.100 | 0.300 | 0.580 |
| Gold page missed | 104 | 0.077 | 0.202 | - |
| Gold page retrieved | 46 | 0.217 | 0.326 | - |

Correctness nearly triples when a gold page is retrieved, from 7.7% to 21.7%, but remains far from perfect. This demonstrates two separate limitations: retrieval frequently fails to expose the evidence, and the 3B generator still struggles with arithmetic, unit conversion, and long financial tables even when the correct page is available.

[[FIGURE:figure_3_bottleneck.png|Figure 3. Answer verdicts conditioned on retrieval of a gold page. Retrieval success improves answer quality, but generation errors remain.]]

## 5.3 Chunking ablation

| Variant | Chunks | MRR | Page hit@5 | Page hit@10 | Evidence hit@10 | Evidence 3-gram recall@10 |
|---|---:|---:|---:|---:|---:|---:|
| Recursive 256/50 | 13,797 | 0.324 | 0.467 | 0.600 | 0.500 | 0.451 |
| Recursive 384/50 | 8,886 | 0.247 | 0.467 | 0.600 | 0.600 | 0.518 |
| Recursive 256/0 | 11,775 | 0.252 | 0.400 | 0.600 | 0.533 | 0.421 |
| Fixed windows 256/50 | 12,849 | 0.352 | 0.533 | 0.600 | 0.567 | 0.454 |

Larger chunks reduce index size by 36% and improve top-10 evidence coverage, but reduce MRR by 0.077. Removing overlap saves 15% of vectors but lowers MRR by 0.072 with a paired 95% confidence interval of [-0.153, -0.012], the only chunking interval excluding zero. Fixed token windows are competitive in retrieval, but their decoded text is harder to read and may split sentences or table rows. The recommended primary configuration remains recursive 256/50; recursive 384/50 is the strongest candidate for a future full-corpus validation.

[[FIGURE:figure_4_chunking.png|Figure 4. Chunking pilot trade-offs. Values are paired over 30 questions and should not be compared with the absolute full-corpus results.]]

## 5.4 Generator-size experiment

On the 46 questions with a retrieved gold page, 3B and 7B refuse at the same 43.5% rate. The 7B model produces citations in 89.1% of answers, compared with 58.7% for 3B, and cites an actual gold page in 60.9% of cases, compared with 37.0%.

| Generator | Refusal | Any citation | Gold-page citation | Mean answer words |
|---|---:|---:|---:|---:|
| Qwen2.5 3B | 0.435 | 0.587 | 0.370 | 32.2 |
| Qwen2.5 7B | 0.435 | 0.891 | 0.609 | 36.2 |

Blinded correctness judging is less decisive. The 3B judge prefers 7B in 14 cases, 3B in 11, and ties 21; the sign-test p-value is 0.690. The 7B judge prefers 7B in 25 cases, 3B in 14, and ties 7; p=0.108. After swapping A/B order, 54.3% of 3B-judge decisions and 58.7% of 7B-judge decisions are unstable. Therefore, citation behavior clearly improves with 7B, but a general correctness advantage is not supported robustly.

[[FIGURE:figure_5_generator.png|Figure 5. Generator behavior on matched gold-page contexts. Qwen2.5 7B improves source-use behavior, while refusal remains unchanged.]]

# 6. Error Analysis

## 6.1 Retrieval failures

The most frequent retrieval failure is document disambiguation. Annual reports from the same company share vocabulary, table titles, and line-item names, so dense search may retrieve the correct topic from the wrong fiscal year. Literal-year BM25 signals help a small number of questions, but repeated financial vocabulary also creates sparse-retrieval distractions.

Chunk localization creates a second failure mode. A correct page can be retrieved while the selected chunk omits the evidence-bearing part of a long table. For a Block FY2020 cash-flow question, the gold page appears at rank 9, yet the retrieved chunk covers only part of the statement and obtains 0.163 evidence 3-gram recall. Page hit therefore overstates usable retrieval.

Evidence annotations and PDF extraction occasionally disagree about page location. For a Foot Locker chief-executive question, FinanceBench labels page 2 while matching exhibit text is extracted on page 29 of the same PDF. Evidence-text scoring detects the content even when page hit reports a miss. This justifies reporting both metrics.

Some failures are complete misses. The 3M FY2018 capital-expenditure question has neither a page hit nor evidence overlap in the top results. Such cases cannot be repaired by changing the generator because the required number never enters the prompt.

## 6.2 Generation failures

Of the 46 questions with a gold page retrieved, 31 are still judged incorrect and five partial. The local generator often refuses when relevant information is buried in a table, extracts the wrong year or unit, or states a value without completing the requested calculation. Metrics-generated questions show the lowest partial-credit rate, consistent with exact numerical matching.

The fixed refusal policy reduces unsupported hallucination but contributes to the 59.3% overall refusal rate. Citation presence also does not guarantee grounding: 65 answers cite at least one retrieved passage, while only 17 cite a gold page. A citation can be syntactically valid yet point to irrelevant evidence.

| Error source or symptom | Evidence | Consequence |
|---|---|---|
| Gold page absent from top 10 | 104 of 150 questions | Generator cannot access annotated evidence |
| Gold page present but answer incorrect | 31 of 46 questions | Reasoning, extraction, or unit failure |
| Refusal | 89 of 150 answers | Lower hallucination risk but low coverage |
| Citation without gold-page grounding | 48 cited answers | Apparent provenance can be misleading |
| LLM judge order instability | More than 54% in generator comparison | Subjective model comparisons are uncertain |

# 7. Discussion

## 7.1 Research-question findings

RQ1 is answered by the page-preserving pdfplumber pipeline. Conservative cleaning removes recurring boilerplate without materially changing financial tables, and recursive splitting retains natural layout boundaries better than fixed windows. The corpus is fully indexed with auditable metadata.

For RQ2, dense semantic retrieval is clearly stronger than BM25 alone. Small lexical fusion can improve a few early ranks, but does not create a robust recall improvement. A 50-token overlap benefits early ranking. Larger chunks offer a meaningful context-completeness and index-size trade-off, but the pilot is too small to justify replacing the full baseline.

For RQ3, increasing the generator to 7B improves citation presence and citation to gold pages at an estimated two-to-threefold latency cost. Correctness improvement is inconclusive because pairwise judges disagree and are sensitive to answer order. The 3B model remains the efficient baseline; 7B is preferable when source-use behavior justifies additional latency.

RQ4 has the clearest answer: retrieval is the dominant end-to-end bottleneck. Only 46 of 150 prompts contain a gold page at k=10. Generation quality also matters, because most of those 46 answers remain incorrect, but generator-only improvements cannot address the larger set of retrieval misses.

## 7.2 Strengths

The system is fully local and reproducible. Every chunk retains document and page provenance; exact FAISS search prevents approximate-index error; generation is deterministic; and answer runners are resumable. Evaluation goes beyond a single headline score by combining page localization, evidence-text overlap, objective numeric checks, local judging, citation behavior, paired statistical tests, and manual disagreement analysis.

The experiments are controlled carefully. Generator models see identical stored contexts. Chunking variants use identical questions and documents. Hybrid weights reuse a single retrieval pass. These choices reduce unintended variables and make negative or inconclusive findings scientifically useful.

## 7.3 Limitations and threats to validity

The chunking experiment uses 30 questions and 26 candidate documents rather than the complete 368-document corpus. Its confidence intervals are wide and its absolute scores are optimistic. Hybrid weights were explored and reported on the same evaluation set, which creates selection bias.

The embedding study uses one dense model. A finance-specialized encoder or cross-encoder reranker may change the conclusions. Tables are represented as text, not structured cells, and no OCR path exists for scanned documents. Page-number annotations can disagree with extracted PDF pages.

Answer evaluation depends partly on a local LLM judge. Numeric matching provides an objective check only for suitable answers, while pairwise judging shows substantial position sensitivity. The report therefore avoids claiming that 7B is generally more correct.

Finally, FinanceBench questions can require arithmetic. The current system asks a language model to calculate directly and does not include a deterministic calculator, unit resolver, or table query engine.

## 7.4 Recommended next work

The highest-priority intervention is metadata-aware retrieval. Company, filing type, and fiscal year can be inferred from many questions and used to filter or rerank candidates before semantic search. A cross-encoder reranker over the top dense results could then improve page localization without rebuilding the full index.

Second, parent-child retrieval could search compact chunks but return a larger table-aware parent region, combining early precision with context completeness. Third, a structured financial reasoning stage should extract operands with citations, normalize units, and execute arithmetic deterministically. Fourth, recursive 384/50 should be evaluated on all 150 questions before any baseline change. Finally, stronger general or finance-specific embedding models should be compared on a held-out development split.

# 8. Conclusion

This project delivers a complete local RAG pipeline for FinanceBench, from automated PDF acquisition through cited answer generation and multi-level evaluation. The system demonstrates that provenance and reproducibility can be maintained across a large, noisy financial corpus using page-bounded chunks, normalized dense embeddings, exact vector search, and locally served language models.

The experiments show that design choices create measurable trade-offs rather than a single universally best configuration. Overlap improves early retrieval; larger chunks improve evidence completeness but weaken ranking; weak lexical fusion helps a few year-sensitive questions; and a larger generator improves citation behavior without a robust correctness advantage. The recommended baseline is recursive 256-token chunks with 50-token overlap, BGE-small dense top-10 retrieval, and Qwen2.5 3B generation.

The principal practical lesson is that generated-answer quality is bounded by retrieval. Better metadata filtering, reranking, table-aware context expansion, and deterministic financial calculation are more promising next steps than simply increasing generator size.

# References

[1] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Kuttler, M. Lewis, W.-t. Yih, T. Rocktaschel, S. Riedel, and D. Kiela. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." Advances in Neural Information Processing Systems 33, 2020. arXiv:2005.11401.

[2] P. Islam, A. Kannappan, D. Kiela, R. Qian, N. Scherrer, and B. Vidgen. "FinanceBench: A New Benchmark for Financial Question Answering." 2023. arXiv:2311.11944.

[3] N. Reimers and I. Gurevych. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." Proceedings of EMNLP-IJCNLP, 2019. arXiv:1908.10084.

[4] J. Johnson, M. Douze, and H. Jegou. "Billion-scale similarity search with GPUs." 2017. arXiv:1702.08734.

[5] S. Robertson and H. Zaragoza. "The Probabilistic Relevance Framework: BM25 and Beyond." Foundations and Trends in Information Retrieval, 3(4):333-389, 2009.

[6] Qwen Team. "Qwen2.5 Technical Report." 2024. arXiv:2412.15115.

[7] Beijing Academy of Artificial Intelligence. "BGE v1 and v1.5 Model Documentation: bge-small-en-v1.5." https://bge-model.com/bge/bge_v1_v1.5.html.

# Appendix A. Reproducibility and artifacts

All source code and exact commands are documented in `README.md`. The principal implementation files are `src/ingest/`, `src/chunk/build_chunks.py`, `src/retrieval/`, `src/generation/`, and `src/eval/`. YAML files under `configs/` capture chunking, embedding, and generator parameters.

The complete baseline artifacts are:

| Stage | Artifact |
|---|---|
| Acquisition | `data/manifest.csv`, `data/raw_pdfs/` |
| Parsing and cleaning | `data/parse_manifest.csv`, `data/clean_manifest.csv`, page Parquet files |
| Baseline chunks | `data/chunks/s256_o50.parquet` |
| Dense index | `data/index/bge-small__s256_o50/` |
| BM25 index | `data/index/bm25__s256_o50/` |
| Retrieval evaluation | `runs/retrieval_bge-small__s256_o50.*` |
| Generated answers | `runs/answers_qwen2.5-3b__bge-small__s256_o50.jsonl` |
| Answer evaluation | `runs/answers_eval_qwen2.5-3b__bge-small__s256_o50.*` |
| Experiment summaries | `runs/*summary.csv`, `runs/chunking_pilot_summary.csv` |

Random seeds and temperatures are fixed, index configurations record provenance, and long-running generation and embedding tasks are resumable. Data, indexes, and raw run files are excluded from version control because of size and can be regenerated with the README workflow.
