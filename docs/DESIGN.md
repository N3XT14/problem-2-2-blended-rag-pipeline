# Blended RAG — Design

## The Problem

An HVAC field technician standing at a unit, asking questions about that specific equipment.

Knowledge base: manufacturer installation, service, and product-data manuals across several brands and product families — 15 PDFs yielding 2352 chunks after ingestion.

Why this is hard:

- Questions mix exact identifiers ("charge for GSX140361") with conceptual ones ("why does the compressor short-cycle")
- Answers often sit inside spec tables, not prose
- Many manuals are near-identical across model variants and retrieval can land on the right family but wrong model. A wrong refrigerant charge or fuse size is a real-world error.

Success criterion: the right page of the right model's manual appears in top-k, and the wrong variant does not outrank it.

---

## Deconstruction — One Sub-Problem to One Component

**P1 — Exact identifiers (model numbers, error codes)**
- Problem: queries contain exact strings like "GSX14-0361" or "E04"
- Why hard: tokenizers split hyphenated IDs into pieces, dense embeddings blur them across similar-looking models
- How: BM25 with a custom model-number analyzer that keeps the full identifier as one token

**P2 — Spec values in tables**
- Problem: refrigerant charge, fuse size, and other critical specs live in tables
- Why hard: PDF extraction flattens tables into word soup — rows and columns lose meaning
- How: table-aware parsing via pymupdf4llm, tables stored as their own chunks with section context

**P3 — Vocabulary mismatch**
- Problem: technician says "short cycling", manual says "Time Guard II control circuit"
- Why hard: no keyword overlap between the query and the answer
- How: KNN dense index for semantic similarity, SPLADE for learned term expansion across the vocabulary gap

**P4 — Near-identical variant manuals**
- Problem: 48AJ and 48AK manuals share spec sections verbatim with different values
- Why hard: BM25 and KNN score them identically — wrong model outranks the right one
- How: model_numbers field weighting in best_fields, model-aware filter to hard-route when model is known

**P5 — Long structured documents**
- Problem: the relevant section is on page 47, irrelevant intro is on page 1
- Why hard: a flat chunk of the full doc buries the answer
- How: section-aware chunking, section_title^2 boost in best_fields so the right section ranks higher

**P6 — No single retriever wins every query type**
- Problem: lookup, conceptual, and trap queries each need a different retrieval strategy
- Why hard: BM25 misses synonyms, KNN blurs exact codes, each has blind spots
- How: three indexes (BM25, KNN, SPLADE) across two query strategies (match, best_fields), fused with RRF

**P7 — Technicians need to verify the answer**
- Problem: a confident wrong answer in this domain means wrong refrigerant charge or fuse size which is a very big error
- Why hard: generic RAG returns text with no source attribution
- How: every chunk carries doc_id and page number, every result shows which manual and page it came from

---

## Stack

- **Elasticsearch 9.0.3** — three indexes: BM25, dense KNN, sparse SPLADE
- **all-MiniLM-L6-v2** — 384-dim dense embeddings, no GPU needed
- **naver/splade-cocondenser-selfdistil** — sparse encoder, substitutes IBM's ELSER
- **RRF** — fuses all three ranked lists, IBM picks best single combo, we blend all three
- **pymupdf4llm** — PDF to markdown with table structure preserved

---

## Full WorkFlow
![Workflow Diagram](<Blended RAG pipeline.drawio.png>)

---

## What BM25 + KNN Got Right

Both strategies tested: `match` (single field) and `best_fields` (text + section_title^2 + model_numbers).

- doc_recall@5: best_fields 0.875, match 0.850
- MRR: best_fields 0.735, match 0.732
- wrong_variant@1: both 0.350

Reasoning questions (conceptual "why" queries): doc_recall = 1.0 on both. The dense KNN handles vocabulary mismatch well for open-ended questions.

Lookup questions: recall 0.923, but wrong_variant@1 = 0.385 — the right section exists but it's from the sibling manual.

---

## Where BM25 + KNN Failed

Three failure modes that motivated adding SPLADE:

**1. Synonym gap**
Query "coolant recharge procedure" — the word "coolant" does not appear in HVAC manuals. The manual says "refrigerant". BM25 has no way to bridge this.

**2. Exact-code dilution**
Query "error code E04" — BM25 surfaces generic troubleshooting sections because "troubleshooting" has high term frequency. The specific fault-code table is ranked lower.

**3. Wrong variant at rank 1**
Sibling manuals (48AJ vs 48AK) share spec sections verbatim. BM25 and KNN both pull the wrong variant 35% of the time at rank 1.

---

## Why SPLADE

SPLADE (Sparse Lexical and Expansion) uses a BERT masked language model to expand queries and documents into a weighted token vocabulary. It learns which tokens co-occur in technical contexts — so "coolant" expands to include "refrigerant", "charge", "recharge" at query time.

It's a sparse model (like BM25) so it stays interpretable and runs on CPU. It substitutes IBM's ELSER, which requires Elastic Cloud. Same learned expansion principle, no cloud dependency.

---

## SPLADE in Practice — Problems It Solved

**Query: "coolant recharge procedure"**

Baseline @1: Section "Manifold Adjustment Procedure" — gas valve and manifold pressure, wrong domain entirely.

+SPLADE @1: Section "A−Refrigerant Charge and Check" — the exact section covering refrigerant charge procedure, nameplate charge, evacuation steps.

SPLADE expanded "coolant" -> "refrigerant", "recharge" -> "charge". The answer was not reachable without it.

---

**Query: "error code E04 troubleshooting"**

Baseline @1: Generic "Troubleshooting" section. @2: "ReliaTel™ Control" — general control docs.

+SPLADE @1: "Heating Failure" — contains the specific LED flash-code table (1 Flash = Communication Failure, 4 Flashes = TC01/TC02 Open, etc.). More specific than generic troubleshooting prose.

E04 does not exist in these manuals. SPLADE still surfaces fault-code reference tables over generic sections by weighting the diagnostic token context.

---

**Query: "compressor short cycling"**

Baseline @1: "Time Guard II control circuit" — correct. Time Guard II is the anti-short-cycle protection circuit.

+SPLADE @1: same. @2: "Time guard II control circuit" from a second model variant — also correct.

Both retrieve the right answer. SPLADE additionally surfaces the same section from a second relevant model. No regression.

---

## Results After Adding SPLADE

**best_fields strategy:**
- doc_recall@5: 0.875 to **0.925** (+5.0pp)
- MRR: 0.735 to **0.765** (+3.0pp)
- wrong_variant@1: 0.350 to **0.325** (-2.5pp)

**match strategy:**
- doc_recall@5: 0.850 to **0.925** (+7.5pp)
- MRR: 0.732 to **0.765** (+3.3pp)
- wrong_variant@1: 0.350 to **0.325** (-2.5pp)

SPLADE closed the gap between the two strategies on recall — match was 2.5pp behind best_fields at baseline, both reach 0.925 with SPLADE. The vocabulary expansion compensated for the narrower single-field search in match.

No query type regressed. The 3 demo queries confirm the pattern holds on real technician questions, not just benchmark statistics.

---

## Two Metrics, Two Different Questions

IBM's metric asks: does the expected answer string appear word-for-word in a returned chunk? My solution asks: did the right document show up in top-5?

ibm_recall@5 = 0.600, doc_recall@5 = 0.925 — a 32.5pp gap. The retrieval is working. The verbatim string isn't there because answers get paraphrased across PDFs, split across chunk boundaries, or live in table cells.

The clearest example: reasoning questions score ibm_recall = 0.143 but doc_recall = 1.0. "Why does the compressor short-cycle" has no single chunk that contains the full answer verbatim and hence the answer requires reading across sections. IBM's metric says we failed. my solution retrieved the right manual every time.

Both are reported. doc_recall is the primary signal.

---

## References

- IBM "Blended RAG: Improving RAG Accuracy with Semantic Search and Hybrid Query-Based Retrievers", 2024
- SPLADE "SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking", 2021
- SPLADE model: `naver/splade-cocondenser-selfdistil` (Naver Labs, HuggingFace)
- Dense model: `sentence-transformers/all-MiniLM-L6-v2`

Full eval results: `results/eval_results.json` (SPLADE+best_fields), `results/eval_match.json` (SPLADE+match), `results/eval_baseline_best_fields.json`, `results/eval_baseline_match.json`

---

## Scale

Retrieval scales — Elasticsearch handles millions of documents via sharding and replication, the query path stays fast. The bottleneck is indexing: SPLADE encoding takes ~17 min for 2352 chunks on CPU. At 100K chunks that would be several hours. GPU (auto-detected via CUDA) or distributed encoding would be needed at that scale. The architecture is correct, the indexing throughput needs hardware to match corpus size.

---

## What's Left

- **wrong_variant@1 = 0.325** — model metadata filter would hard-route queries to the right manual. Target <0.15.
- **SPLADE indexing throughput** — ~17 min for 2352 chunks on CPU, needs GPU or batch parallelism at larger scale
- **CLIP not wired for vision** — manuals have wiring diagrams, image retrieval not implemented
- **Query classifier** — route lookup queries to BM25-heavy, conceptual to KNN-heavy automatically, rather than running both strategies and picking
