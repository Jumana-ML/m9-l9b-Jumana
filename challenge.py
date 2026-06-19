# challenge.py
import numpy as np
from sentence_transformers import SentenceTransformer, util

class EntityLinkerChallenges:
    def __init__(self):
        # Tier 1: Cache for Co-reference resolution
        self.coref_cache = {}
        
        # Tier 2: Embedding Model (Lazy loaded)
        self.model = None 

    # --- Tier 1: Co-Reference Logic ---
    def lookup_coref(self, surface: str):
        """Check if surface form was already resolved in this document."""
        return self.coref_cache.get(surface)

    def update_coref(self, surface: str, node_id: str, type_label: str):
        """Store a successful resolution in the cache."""
        if node_id and surface not in self.coref_cache:
            self.coref_cache[surface] = {"id": node_id, "label": type_label}

    def clear_coref(self):
        """Reset cache at the start of a new document."""
        self.coref_cache = {}

    # --- Tier 2: Embedding Ranker Logic ---
    def rank_candidates_with_embeddings(self, text, start, end, candidates):
        """Challenge Tier 2: Rank candidates using semantic similarity."""
        if self.model is None:
            # Load lightweight model
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 1. Extract context window (e.g., 100 characters around the span)
        window = 100
        context = text[max(0, start-window): min(len(text), end+window)]
        
        # 2. Build descriptions for candidates (Name + Domain Labels)
        descriptions = [f"{c['name']} is a {' '.join(c['labels'])}" for c in candidates]
        
        # 3. Compute Embeddings
        context_emb = self.model.encode(context, convert_to_tensor=True)
        cand_embs = self.model.encode(descriptions, convert_to_tensor=True)
        
        # 4. Compute Cosine Similarity
        scores = util.cos_sim(context_emb, cand_embs)[0]
        
        # Sort by similarity score
        ranked_indices = scores.argsort(descending=True)
        best_idx = ranked_indices[0].item()
        
        # Calculate Margin for Tier 3 (Confidence gap between top 2 candidates)
        margin = 0
        if len(scores) > 1:
            margin = scores[ranked_indices[0]].item() - scores[ranked_indices[1]].item()
            
        return candidates[best_idx], scores[best_idx].item(), margin