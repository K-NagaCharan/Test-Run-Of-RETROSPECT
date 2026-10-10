"""
Method C: Laya Encoder Embedding Matcher
Uses ModernBERT-based representations extracted via laya.embed_fn_from_agent()
Evaluates both English and Multilingual checkpoints.
"""
import hashlib
from typing import Dict, Any, List, Optional
import numpy as np
import laya

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))

class LayaEmbeddingMatcher:
    def __init__(self, checkpoint_type: str = "english", device: str = "cpu", new_threshold: float = 0.75):
        self.checkpoint_type = checkpoint_type
        self.device = device
        self.new_threshold = new_threshold
        self._cache = {}

        print(f"Loading Laya embedding model ({checkpoint_type}) on {device}...")
        if checkpoint_type == "multilingual":
            self.agent = laya.load("convaiinnovations/laya", device=device, subfolder="multilingual")
        elif checkpoint_type == "typed-decisions":
            self.agent = laya.load("convaiinnovations/laya", device=device, subfolder="typed-decisions")
        else:
            self.agent = laya.load("convaiinnovations/laya", device=device)

        self.embed_fn = laya.embed_fn_from_agent(self.agent, max_length=512, batch_size=32)
        print("Laya embedding model ready.")

    def _hash(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def get_embedding(self, text: str) -> np.ndarray:
        h = self._hash(text)
        if h in self._cache:
            return self._cache[h]
        # Generate embedding
        cleaned = text.strip() if text.strip() else "empty_function"
        vec = self.embed_fn([cleaned])[0]
        self._cache[h] = vec
        return vec

    def predict(self, target_info: Dict[str, Any], candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        target_src = target_info.get("source", "")
        t_vec = self.get_embedding(target_src)

        scores = {}
        for c in candidates:
            cid = c["candidate_id"]
            c_src = c.get("source", "")
            c_vec = self.get_embedding(c_src)
            sim = cosine_similarity(t_vec, c_vec)
            scores[cid] = sim

        if not scores:
            return {"prediction": "NEW", "scores": {}, "confidence": 1.0}

        best_cand = max(scores, key=scores.get)
        best_score = scores[best_cand]

        sorted_scores = sorted(scores.values(), reverse=True)
        margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else sorted_scores[0]

        prediction = best_cand if best_score >= self.new_threshold else "NEW"

        return {
            "prediction": prediction,
            "scores": scores,
            "best_candidate": best_cand,
            "best_score": best_score,
            "margin": margin,
            "confidence": best_score
        }
