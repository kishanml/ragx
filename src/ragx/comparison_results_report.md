# Comparison Results Report

## Executive Summary

The workbook contains **308 unique topic comparisons** between **Model A** and **Model B**.

Model A is the stronger overall performer:

| Metric | Model A | Model B |
| --- | ---: | ---: |
| Wins | 211 | 69 |
| Ties | 28 | 28 |
| Final score average | 23.60 | 21.75 |
| Accuracy average | 7.57 | 6.63 |
| Relevance average | 8.18 | 7.66 |
| Variety average | 7.85 | 7.46 |

## Main Insights

1. **Model A wins about two-thirds of the comparisons.**
   - Win rate: **68.5%**
   - Model B win rate: **22.4%**
   - Tie rate: **9.1%**

2. **The biggest gap is in accuracy and relevance, not variety.**
   - Accuracy gap: **+0.94** in Model A's favor
   - Relevance gap: **+0.52** in Model A's favor
   - Variety gap: **+0.39** in Model A's favor

3. **Most rows are still fairly close.**
   - Final score margin of 1 point: **66 rows**
   - Margin of 2 points: **56 rows**
   - Margin of 3-5 points: **110 rows**
   - Margin above 5 points: **47 rows**
   - Exact final-score ties: **29 rows**

4. **Model A's strongest wins tend to be on specific, evidence-heavy nutrition/health prompts.**
   - `The Top Three DNA Protecting Spices`
   - `Go Nuts for Breast Cancer Prevention`
   - `When a Scraped Knee May Once Again Kill`
   - `Treating Breast Pain with Flax Seeds`

5. **Model B's better wins often look like cautious, tightly scoped responses.**
   - `Trans Fat in Animal Fat`
   - `Boosting Gut Flora Without Probiotics`
   - `Cayenne for Irritable Bowel`
   - `What about the INTERHEART study on heart attack risk factors?`

## Code-Level Interpretation

In this repo, **Model A** is the richer retrieval pipeline and **Model B** is the simpler vector-search baseline. The score gap is consistent with the implementation gap between the two systems.

### Model A

Model A is a multi-stage retrieval pipeline:

- It stores a chunk in a main SQLite table, then enriches it with an LLM-generated:
  - `summary`
  - `questions`
  - `keywords`
- It builds **two separate vector indexes**:
  - one over summaries
  - one over generated questions
- It retrieves from both indexes and merges the candidates.
- It can optionally rerank the final candidates with a **cross-encoder reranker**.

This design gives the retriever more than one way to match a query. A question can hit:

- the original chunk meaning through the summary index,
- a paraphrased query form through the generated questions index,
- and then be refined by reranking.

That extra representation layer is a strong explanation for why Model A scores better on **accuracy** and **relevance**. The system is not relying only on raw chunk wording; it is expanding the evidence representation before retrieval.

### Model B

Model B is a much simpler baseline:

- It stores each chunk once in SQLite.
- It embeds the **raw chunk text only** with `SentenceTransformer`.
- It searches a single FAISS index.
- It returns the nearest stored rows directly, with no summary expansion and no reranking.

This design is cheaper and easier to reason about, but it is also more brittle. If the query wording does not line up well with the chunk wording, the retriever has fewer chances to recover the right evidence. That typically shows up as weaker relevance and weaker factual support in downstream answers.

### Why the scores line up with the code

The workbook shows Model A winning most often, and the margin is mainly in the quality dimensions that depend on retrieval quality:

- **Accuracy**
- **Relevance**

That is exactly what you would expect from a system that:

1. rewrites and summarizes content before indexing,
2. indexes multiple views of the same chunk,
3. and optionally reranks with a stronger pairwise scorer.

By contrast, Model B is a straightforward embedding search baseline. It can still do well on concise, well-aligned questions, which explains its narrower wins, but it has less machinery to recover when the query and source text are phrased differently.

## Bottom Line

Model A is the better overall system in this evaluation set. The advantage is real but not massive, and it comes primarily from stronger factual handling and relevance rather than from style or verbosity. The implementation supports that interpretation: the richer pipeline is built to retrieve more semantically useful evidence than the simpler baseline.
