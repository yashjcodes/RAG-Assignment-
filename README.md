# RAG Chunking Strategy Comparison: Naive vs Structure-Aware

A small Retrieval-Augmented Generation (RAG) pipeline over a Type 2 diabetes reference guide, built to compare two ways of chunking the same document and measure which one retrieves better.

## Repository layout

```
rag-chunking/
├── data/
│   └── diabetes_reference_document.md   # the reference document (input)
├── src/
│   ├── chunking.py      # Strategy A (fixed-size) and Strategy B (structure-aware)
│   ├── embeddings.py    # TF-IDF embedder (default) and MiniLM embedder (optional)
│   ├── retrieval.py     # NumPy cosine-similarity vector store, top-k retrieval
│   ├── questions.py     # the 10 test questions + ground-truth rubric
│   ├── generate.py      # OPTIONAL LLM answer generation (needs ANTHROPIC_API_KEY)
│   └── evaluate.py      # runs both pipelines, judges retrieval, writes results/
├── tests/
│   ├── test_chunking.py                # unit tests for naive & structure chunking
│   ├── test_embeddings_and_retrieval.py# unit tests for TF-IDF & vector store
│   └── test_evaluation_and_pipeline.py # unit tests for rubric judge & pipeline
├── results/
│   ├── results_table.md / .csv     # question-by-question comparison
│   ├── summary.json                # chunk counts, sizes, accuracy
│   ├── retrieval_log.json          # every retrieved chunk + score, per question/strategy
│   └── chunks_A_naive.txt / chunks_B_structure.txt   # all chunks, for manual inspection
├── requirements.txt
└── README.md
```

## How to run

### 1. Run the test suite
```bash
python -m unittest discover -s tests -v
```

### 2. Run the evaluation pipeline
```bash
pip install -r requirements.txt
python src/evaluate.py                    # default: TF-IDF embeddings, top-3
python src/evaluate.py --k 1              # change top-k (e.g. top-1)
python src/evaluate.py --embedder minilm  # semantic embeddings (pip install sentence-transformers)
```

## Approach

| Step | Choice |
|---|---|
| **Chunking A (naive)** | Plain string slicing: 500-character windows, 50-character overlap. Ignores headers, bullets, tables and sentences. |
| **Chunking B (advanced)** | Structure-aware, written in plain Python (no LangChain). It splits on markdown headers first, then packs whole paragraphs/lists/tables up to 1000 characters, and only falls back to line and sentence splitting for oversized blocks. A line ending in `:` is glued to the list or table below it. Every chunk is prefixed with its header breadcrumb, e.g. `[... > 5. Treatment Approaches > 5.2 Oral Medications]`, so a chunk keeps its context even when read in isolation. |
| **Embeddings** | TF-IDF (unigrams + bigrams, sublinear tf, English stop-words), L2-normalised. See limitations below. `all-MiniLM-L6-v2` is supported via `--embedder minilm`. |
| **Vector store** | NumPy matrix and cosine similarity (dot product of normalised vectors), top-k = 3. |
| **Generation** | **Retrieval-only evaluation.** No LLM API was used for the reported numbers. `generate.py` will produce answers if `ANTHROPIC_API_KEY` is set, and they are written to `retrieval_log.json`. |
| **Judging** | Each question has a rubric of required facts (`src/questions.py`). The union of the top-k chunks is judged **yes** (enough required facts present), **partial** (some present) or **no**. I then read the retrieved chunks in `retrieval_log.json` to confirm the verdicts. Top-1 is also reported, because with only 13–17 chunks in the whole document, top-3 covers about 20% of the corpus and is a lenient test. |

### Notes on the question set

- **Question 9 is not answerable from this document.** The guide gives *diagnostic thresholds* (e.g. fasting glucose of 126 mg/dL or higher), but nowhere states a *recommended glucose target range* for treatment. The right behaviour for a RAG system is to say "not in the document". It is marked `n/a` and excluded from accuracy, which is computed over the 9 answerable questions. A 10-question figure (counting Q9 as a miss) is also in `summary.json`.
- **Question 3 is loosely worded.** The document calls only metformin "first-line", then lists four other oral drug classes. The rubric treats the question as satisfied when at least 3 oral drug classes appear in the retrieved text.

## Results

(top-3 retrieval, TF-IDF embeddings. "chunks" are the retrieved chunk IDs in rank order; "top-k" = judged on all 3 retrieved chunks, "top-1" = judged on the best chunk alone.)

| # | Question | A: chunks | A: top-k | A: top-1 | B: chunks | B: top-k | B: top-1 |
|---|---|---|---|---|---|---|---|
| 1 | What are the two main diagnostic criteria for Type 2 diabetes based on HbA1c and fasting plasma glucose? | 4,0,5 | yes | yes | 3,0,9 | yes | yes |
| 2 | What HbA1c range indicates prediabetes? | 15,4,11 | yes | no | 12,3,6 | yes | no |
| 3 | Name three first-line oral medications used to treat Type 2 diabetes. | 7,16,0 | yes | yes | 5,6,12 | yes | yes |
| 4 | What are the classic symptoms of hyperglycemia? | 5,3,2 | yes | no | 2,3,1 | yes | yes |
| 5 | What lifestyle modifications are recommended for managing Type 2 diabetes? | 6,5,0 | yes | yes | 4,10,12 | yes | yes |
| 6 | Name two microvascular complications of diabetes. | 9,10,11 | yes | yes | 7,8,9 | yes | yes |
| 7 | Name two macrovascular complications of diabetes. | 10,9,11 | yes | yes | 8,7,9 | yes | yes |
| 8 | How often should HbA1c be monitored in someone with stable Type 2 diabetes? | 0,11,16 | yes | no | 9,3,0 | yes | yes |
| 9 | What blood glucose target range is recommended for most adults with Type 2 diabetes? | 0,12,15 | n/a | n/a | 10,0,2 | n/a | n/a |
| 10 | What symptoms should prompt someone to seek emergency medical care? | 13,3,5 | partial | partial | 11,2,3 | yes | yes |

### Summary statistics

| Metric | A: Naive (500 / 50) | B: Structure-aware |
|---|---|---|
| Number of chunks | 17 | 13 |
| Average chunk size (chars) | 490.3 | 631.9 |
| Min / max chunk size (chars) | 335 / 500 | 415 / 940 |
| **Retrieval accuracy, top-3, strict** (yes only, 9 answerable Qs) | **88.9%** (8/9) | **100.0%** (9/9) |
| Retrieval accuracy, top-3, lenient (yes + partial) | 100.0% | 100.0% |
| **Retrieval accuracy, top-1, strict** | **55.6%** (5/9) | **88.9%** (8/9) |
| Strict top-3 accuracy counting Q9 as a miss (out of 10) | 80.0% | 90.0% |

## Analysis

**Structure-aware chunking (B) performed better, and the gap is clearest in top-1 retrieval.** With top-3 retrieval both strategies look good (A 88.9%, B 100% strict), but that is partly a property of the test: the document yields only 13–17 chunks, so retrieving three of them already covers a large part of the corpus. Top-1 is the more telling number: B put a chunk containing the answer first for 8 of 9 questions, versus 5 of 9 for A.

The failures of A trace directly to boundaries it cannot see. Its 500-character window cut the diagnostic table in the middle of a row (chunks 4 and 5), so the fasting-glucose and HbA1c thresholds are split across two chunks and Q1 only succeeds if both are retrieved. For the emergency-care question (Q10), A's chunk 13 ends partway through the first bullet, so the best chunk had only the 300 mg/dL item. The chest-pain, severe-hypoglycaemia and foot-wound bullets sit in the next chunk, which top-3 did not return, hence only a *partial* result. B keeps each section whole, so the single "9. When to Seek Emergency Care" chunk had all four warning signs. Q4 (symptoms) and Q8 (HbA1c monitoring frequency) show the same effect: for A the top-1 chunk was the wrong one, for B it was the right section.

Neither strategy did well on **Q2 (prediabetes range)** at top-1. The query word "prediabetes" appears most densely in the FAQ section, which outranked the table. This is a weakness of TF-IDF matching and not of chunking, and a semantic embedder might fix it. **Q9** is unanswerable. Both pipelines return plausible-looking but irrelevant chunks, with similarity scores no clearly lower than for answerable questions. Retrieval alone cannot signal "not in the document", so a generation step with an explicit abstain instruction or a score threshold is needed.

**Trade-offs.** A is about 10 lines of code, has no assumptions about document format, and gives uniform chunk sizes, but it is blind to structure. B costs more code, depends on the document having headers and clean paragraphs (it would need adapting for a PDF without markup), and produces uneven chunk sizes (415–940 characters). Here the document is well structured, so B's extra complexity paid off.

## Limitations (please read)

- **Embedding model.** The sandbox used to build this repo could not download HuggingFace models, so the reported numbers use **TF-IDF**, a lexical method, not a neural embedding model. Absolute scores will differ with `--embedder minilm`, and the A-vs-B gap may shrink or grow. Run both and compare.
- **Small corpus.** One 7.5 KB document and 10 questions means differences of one question (about 11 points) are within noise. Treat the result as illustrative.
- **Retrieval-only.** No generated answers are included, so "is the generated answer correct" was not evaluated.
- **Rubric judging.** Verdicts come from keyword rules, checked by reading the retrieved chunks. A keyword match shows a fact is present, but not that a human would find the chunk easy to answer from.
- **Structure-aware, not semantic.** Strategy B is rule-based. Semantic chunking (e.g. LangChain's `SemanticChunker`) was not implemented.
