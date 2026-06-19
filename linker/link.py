"""Linker orchestrator.

Wires candidates() -> disambiguate() into one pass over the NER spans of
a document, producing one LinkResult per span.
"""
from linker.candidates import candidates
from linker.disambiguate import disambiguate
from linker.types import LinkResult
from challenge import EntityLinkerChallenges

# Global instance for challenge logic
challenge_manager = EntityLinkerChallenges()
# Registry to store uncertainty data for Tier 3 analysis
uncertainty_registry = []

def link(driver, doc_id, text, ner_spans):
    """Linker orchestrator with Co-reference and Embedding ranking."""
    results = []
    doc_resolved = []
    
    # Tier 1: Clear document cache
    challenge_manager.clear_coref()

    for start, end, surface, ner_label in ner_spans:
        # 1. Tier 1: Check Co-reference Cache first
        cached = challenge_manager.lookup_coref(surface)
        if cached:
            res = LinkResult(doc_id, start, end, surface, 
                             cached['id'], cached['label'], "resolved-by-coref")
            results.append(res)
            doc_resolved.append(res)
            continue

        # 2. Candidate Generation
        cands = candidates(driver, surface)
        
        predicted_node_id = None
        predicted_type_label = None
        reason = ""

        if not cands:
            reason = "nil-no-candidates"
        elif len(cands) == 1:
            predicted_node_id = cands[0]["id"]
            predicted_type_label = cands[0]["labels"][0]
            reason = "resolved-unique"
        else:
            # 3. Tier 2: Use Embedding Ranker for Ambiguous Spans
            chosen, score, margin = challenge_manager.rank_candidates_with_embeddings(text, start, end, cands)
            predicted_node_id = chosen["id"]
            predicted_type_label = chosen["labels"][0]
            reason = "resolved-by-embedding"
            
            # 4. Tier 3: Record uncertainty (margin) for Active Learning
            uncertainty_registry.append({
                "doc_id": doc_id, 
                "surface": surface, 
                "margin": margin, 
                "predicted": predicted_node_id
            })

        result = LinkResult(doc_id, start, end, surface, 
                            predicted_node_id, predicted_type_label, reason)
        
        # Update Tier 1 Cache for future mentions in this doc
        if predicted_node_id:
            challenge_manager.update_coref(surface, predicted_node_id, predicted_type_label)
            
        results.append(result)
        doc_resolved.append(result)

    return results