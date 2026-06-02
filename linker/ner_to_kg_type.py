"""Mapping from spaCy NER labels to KG class URIs.

The mapping is consulted by ``disambiguate()`` to filter candidates to those
whose ``rdf:type`` is compatible with the NER label that produced the span.

The starter ships two entries pre-filled. Extend the mapping to cover every
NER label that appears in ``data/dev.jsonl`` — missing rows silently force
NIL on otherwise resolvable spans (the TA Look-Fors call this out).
"""

# Namespace prefix used for the KG types — matches the KG's `@prefix :`.
RECIPES = "http://aispire.example.org/recipes/"

NER_LABEL_TO_KG_TYPE: dict[str, str] = {
    "PERSON": f"{RECIPES}Author",
    "NORP": f"{RECIPES}Cuisine",
    # TODO: extend to cover the remaining NER labels that appear in dev.jsonl.
    # Hint: inspect dev.jsonl and `data/recipes_kg.ttl` to see which KG class
    # each label should map to. Labels likely needed include PRODUCT, GPE,
    # and ORG. Pick the KG class whose instances those NER spans should resolve to.
}
