# Lab 9B — Learner Notes

This file is for **Challenge Extension** submissions. The base lab (Tasks 1–4)
is graded entirely by the autograder; you do not need to write notes for the
base submission.

If you attempt a Challenge Tier (1, 2, or 3), use this file to document the
analysis the tier asks for. Honors review reads this file alongside your PR
description.

## Which tier(s) did you attempt?

> _Name the tier (or tiers) below the base submission. If you stacked tiers,
> describe the order you implemented them in._

## Dev-split P/R/F1

Report the metrics on the dev split for **each** of the following that applies
to your submission. The autograder runs against the test split; these dev
numbers are what Honors review uses to judge whether your extension actually
moved the linker.

| Run | Precision | Recall | F1 |
|---|---|---|---|
| Base linker (no extension) |  |  |  |
| Tier 1 — with co-reference layer |  |  |  |
| Tier 2 — embedding ranker |  |  |  |
| Tier 3 — after active-learning labels added |  |  |  |

Leave rows blank for tiers you did not attempt.

## Tier 1 — Co-reference resolution analysis

If you attempted Tier 1, describe: (a) how you decide a mention is the "first
unambiguously-linked" mention, (b) the scope at which you stop propagation
(span, sentence, document), and (c) one example from the dev split where the
co-reference layer corrected an error the base cascade made — name the
`doc_id`, the surface form, and the link your cascade originally produced vs.
what the co-reference layer produced.

> _Your answer here._

## Tier 2 — Embedding ranker analysis

If you attempted Tier 2, describe: (a) the embedding model and the
neighbor-relation features you used, (b) the context window length around the
span, and (c) **where each arm wins**. Name at least one dev-split span where
the embedding ranker resolved correctly and the rule-based cascade did not,
and at least one where the cascade resolved correctly and the embedding ranker
did not. Both arms have failure modes — your answer should name them.

> _Your answer here._

## Tier 3 — Active-learning selector analysis

If you attempted Tier 3, describe: (a) how you computed the top-1/top-2
margin per span, (b) which 20 spans you surfaced (paste the `doc_id` + surface
form list), and (c) the labeled-by-uncertainty vs. uniformly-random
comparison. The deliverable is the comparison: a single dev-split P/R/F1 delta
in your favor is the claim; the random baseline is what makes it credible.

> _Your answer here._

## Anything else worth surfacing

Optional. Edge cases, ambiguous gold annotations you flagged, schema
observations, or design choices you want Honors review to see.

> _Your answer here._
