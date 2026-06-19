"""Linker P/R/F1 scoring."""
import collections
from linker.types import LinkResult, GoldSpan


def score(predictions: list[LinkResult], gold: list[GoldSpan]) -> dict:
    """Compute precision, recall, F1 over (node_id, type_label) tuples.

    Triple-stated methodology (verbatim in lab-spec.md, lab guide page, and
    this docstring):

    - Predictions are filtered to the gold span set (same doc_id, start, end)
      before scoring; predictions on spans absent from gold are dropped.
    - A span is a true positive iff the predicted (node_id, type_label)
      EXACTLY matches gold AND gold is non-NIL.
    - A prediction of a wrong (node_id, type_label) on a non-NIL gold is a
      false positive AND a false negative on that span.
    - A NIL prediction on a non-NIL gold is a false negative only.
    - A non-NIL prediction on a NIL gold is a false positive only.
    - A NIL prediction on a NIL gold is a true negative (not counted in
      precision or recall).
    - Aggregation is macro-average across documents (per-doc P/R/F1 averaged
      with equal weight per doc; docs with no gold spans are skipped).

    Returns {'precision': float, 'recall': float, 'f1': float}.
    """
    # 
    # 1. Build a gold-span index keyed by (doc_id, start, end) for fast lookup.
    # 2. Filter predictions to the gold span set per the methodology.
    # 3. For each doc_id in gold, accumulate TP / FP / FN per the rules above
    #    (TN-NIL is informational only — not in P or R).
    # 4. Compute per-doc P, R, F1 (with 0/0 convention: P=R=F1=0 when the
    #    denominator is 0; skip docs with no gold spans entirely).
    # 5. Macro-average the per-doc metrics; return the dict.

    # 1. Group gold spans by doc_id
    gold_by_doc = collections.defaultdict(list)
    for g in gold:
        gold_by_doc[g.doc_id].append(g)

    # 2. Group predictions by doc_id
    preds_by_doc = collections.defaultdict(list)
    for p in predictions:
        preds_by_doc[p.doc_id].append(p)

    doc_precisions = []
    doc_recalls = []
    doc_f1s = []

    # 3. Process each document that has gold spans
    for doc_id, doc_gold in gold_by_doc.items():
        if not doc_gold:
            continue

        # Map predictions in this doc by (start, end) for fast lookup
        doc_preds_lookup = {(p.start, p.end): p for p in preds_by_doc.get(doc_id, [])}

        tp = 0
        fp = 0
        fn = 0

        for g in doc_gold:
            # Get the prediction for this specific gold span
            p = doc_preds_lookup.get((g.start, g.end))

            pred_id = p.predicted_node_id if p else None
            pred_type = p.predicted_type_label if p else None
            
            # Using the correct GoldSpan attributes identified in the trace
            gold_id = g.gold_node_id
            gold_type = g.gold_type_label

            if gold_id is not None:  # Case: Gold is non-NIL
                if pred_id == gold_id and pred_type == gold_type:
                    tp += 1
                elif pred_id is None:
                    fn += 1
                else:
                    # Wrong prediction on non-NIL gold counts as both FP and FN
                    fp += 1
                    fn += 1
            else:  # Case: Gold is NIL
                if pred_id is not None:
                    # Non-NIL Prediction on a NIL gold counts as FP
                    fp += 1
                else:
                    # NIL prediction on NIL gold is a True Negative (ignored)
                    pass

        # 4. Calculate per-doc metrics with 0/0 convention
        p_doc = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r_doc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_doc = (2 * p_doc * r_doc) / (p_doc + r_doc) if (p_doc + r_doc) > 0 else 0.0

        doc_precisions.append(p_doc)
        doc_recalls.append(r_doc)
        doc_f1s.append(f1_doc)

    # 5. Macro-average across documents
    if not doc_precisions:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    num_docs = len(doc_precisions)
    return {
        "precision": sum(doc_precisions) / num_docs,
        "recall": sum(doc_recalls) / num_docs,
        "f1": sum(doc_f1s) / num_docs
    }