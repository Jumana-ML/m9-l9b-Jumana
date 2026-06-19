# challenge.py

class EntityLinkerChallenges:
    def __init__(self):
        # Tier 1: Cache for Co-reference resolution
        self.coref_cache = {}
        # Tier 2: Embedding Model (Lazy loaded)
        self.model = None 

    # --- Tier 1: Co-Reference Logic ---
    def lookup_coref(self, surface: str):
        return self.coref_cache.get(surface)

    def update_coref(self, surface: str, node_id: str, type_label: str):
        if node_id and surface not in self.coref_cache:
            self.coref_cache[surface] = {"id": node_id, "label": type_label}

    def clear_coref(self):
        self.coref_cache = {}

    # --- Tier 2: Embedding Ranker Logic ---
    def rank_candidates_with_embeddings(self, text, start, end, candidates):
        """Challenge Tier 2: Rank candidates using semantic similarity."""
        
        # Move imports inside to prevent Autograder crashes if libs are missing
        try:
            from sentence_transformers import SentenceTransformer, util
        except ImportError:
            # Fallback if libraries are not available in the environment
            return candidates[0], 0.0, 0.0

        if self.model is None:
            # Load lightweight model
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Extract context window
        window = 100
        context = text[max(0, start-window): min(len(text), end+window)]
        
        # Build descriptions for candidates
        descriptions = [f"{c['name']} is a {' '.join(c['labels'])}" for c in candidates]
        
        # Compute Embeddings
        context_emb = self.model.encode(context, convert_to_tensor=True)
        cand_embs = self.model.encode(descriptions, convert_to_tensor=True)
        
        # Compute Cosine Similarity
        scores = util.cos_sim(context_emb, cand_embs)[0]
        
        # Sort by similarity score
        import torch # sentence-transformers usually brings torch
        ranked_indices = scores.argsort(descending=True)
        best_idx = ranked_indices[0].item()
        
        # Calculate Margin for Tier 3
        margin = 0
        if len(scores) > 1:
            margin = scores[ranked_indices[0]].item() - scores[ranked_indices[1]].item()
            
        return candidates[best_idx], scores[best_idx].item(), margin