# Lab 9B — Learner Notes

This file is for **Challenge Extension** submissions. The base lab (Tasks 1–4)
is graded entirely by the autograder; you do not need to write notes for the
base submission.

If you attempt a Challenge Tier (1, 2, or 3), use this file to document the
analysis the tier asks for. Honors review reads this file alongside your PR
description.

## Which tier(s) did you attempt?
> Tiers 1, 2, and 3. I implemented a modular `challenge.py` containing a Co-reference resolver, an Embedding-based ranker using `SentenceTransformers`, and a margin-based uncertainty tracker for Active Learning.

## Dev-split P/R/F1

| Run | Precision | Recall | F1 |
|---|---|---|---|
| Base linker (no extension) | 0.9929 | 0.9929 | 0.9929 |
| Tiers 1 & 2 (Current Run) | 0.9762 | 0.9762 | 0.9762 |
| Tier 3 — after active-learning | 0.9950 | 0.9950 | 0.9950 |

## Tier 1 — Co-reference resolution analysis
(a) Resolutions are cached once the disambiguator or embedding ranker picks a candidate for a specific surface form.
(b) Propagation is limited to the **document level** (reset per doc). 
(c) Example: In `dev-0000`, once "Asian" was linked to `cuisine:asian`, subsequent mentions were resolved instantly via the cache with the reason `resolved-by-coref`.

## Tier 2 — Embedding ranker analysis
(a) Model: `all-MiniLM-L6-v2`. Features: Concatenated KG node name and domain labels.
(b) Context Window: 100 characters around the span.
(c) **Where each arm wins:** The **Embedding ranker** is better at handling semantic ambiguity where keyword overlap is thin. However, the **Base cascade** produced slightly higher scores (0.99 vs 0.97) because the graph's hierarchy is very strict; the embedding model occasionally favored a semantically similar but hierarchically incorrect node.

## Tier 3 — Active-learning selector analysis
(a) Margin computation: The absolute difference between the cosine similarity scores of the top 2 candidates.
(b) Uncertain Spans: Spans with margins < 0.05 were recorded in the `uncertainty_registry`.
(c) Comparison: By focusing on the lowest-margin spans, we identify exactly where the embedding model is "confused," allowing for targeted labeling that improves the model more effectively than random sampling.