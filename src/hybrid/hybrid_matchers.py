"""
Method E: Hybrid Matchers
- E1: Structural retrieval followed by Laya reranking
- E2: Combined scoring (weighted blend of structural, lexical, and Laya embeddings)
- E3: Uncertainty-gated fallback (Laya consulted only when structural matcher is uncertain)
"""
from typing import Dict, Any, List, Optional

class HybridRetrievalReranker:
    """E1: Structural matcher retrieves top-K candidates; Laya reranks among top-K."""
    def __init__(self, structural_matcher, laya_embedder, top_k: int = 2):
        self.structural_matcher = structural_matcher
        self.laya_embedder = laya_embedder
        self.top_k = top_k

    def predict(self, target_info: Dict[str, Any], candidates: List[Dict[str, Any]], language: str) -> Dict[str, Any]:
        # 1. Structural scoring
        struct_res = self.structural_matcher.predict(target_info, candidates, language)
        struct_scores = struct_res.get("scores", {})

        # Filter candidates to top-K
        sorted_cands = sorted(candidates, key=lambda c: struct_scores.get(c["candidate_id"], 0.0), reverse=True)
        top_candidates = sorted_cands[:self.top_k]

        # 2. Laya embedding reranking on top-K
        laya_res = self.laya_embedder.predict(target_info, top_candidates)
        laya_pred = laya_res["prediction"]

        # If structural matcher strongly thinks it's NEW
        if struct_res["prediction"] == "NEW" and struct_res["best_score"] < 0.20:
            final_pred = "NEW"
        else:
            final_pred = laya_pred

        return {
            "prediction": final_pred,
            "structural_prediction": struct_res["prediction"],
            "laya_prediction": laya_pred,
            "structural_scores": struct_scores,
            "laya_scores": laya_res.get("scores", {}),
            "confidence": laya_res.get("confidence", 0.0)
        }

class HybridWeightedCombiner:
    """E2: Linear blend of structural, lexical, and Laya embedding scores."""
    def __init__(self, structural_matcher, lexical_matcher, laya_embedder,
                 w_struct: float = 0.50, w_lex: float = 0.30, w_laya: float = 0.20,
                 new_threshold: float = 0.35):
        self.structural_matcher = structural_matcher
        self.lexical_matcher = lexical_matcher
        self.laya_embedder = laya_embedder
        self.w_struct = w_struct
        self.w_lex = w_lex
        self.w_laya = w_laya
        self.new_threshold = new_threshold

    def predict(self, target_info: Dict[str, Any], candidates: List[Dict[str, Any]], language: str) -> Dict[str, Any]:
        s_res = self.structural_matcher.predict(target_info, candidates, language)
        l_res = self.lexical_matcher.predict(target_info, candidates)
        e_res = self.laya_embedder.predict(target_info, candidates)

        s_scores = s_res.get("scores", {})
        l_scores = l_res.get("scores", {})
        e_scores = e_res.get("scores", {})

        combined_scores = {}
        for c in candidates:
            cid = c["candidate_id"]
            s_val = s_scores.get(cid, 0.0)
            l_val = l_scores.get(cid, 0.0)
            e_val = e_scores.get(cid, 0.0)
            combined_scores[cid] = (
                self.w_struct * s_val +
                self.w_lex * l_val +
                self.w_laya * e_val
            )

        if not combined_scores:
            return {"prediction": "NEW", "scores": {}, "confidence": 1.0}

        best_cand = max(combined_scores, key=combined_scores.get)
        best_score = combined_scores[best_cand]

        prediction = best_cand if best_score >= self.new_threshold else "NEW"

        return {
            "prediction": prediction,
            "scores": combined_scores,
            "best_candidate": best_cand,
            "best_score": best_score,
            "confidence": best_score
        }

class UncertaintyGatedHybrid:
    """E3: Consults Laya only when structural matcher is uncertain."""
    def __init__(self, structural_matcher, laya_matcher,
                 uncertainty_margin: float = 0.15,
                 high_conf_threshold: float = 0.75,
                 low_conf_threshold: float = 0.30):
        self.structural_matcher = structural_matcher
        self.laya_matcher = laya_matcher
        self.uncertainty_margin = uncertainty_margin
        self.high_conf_threshold = high_conf_threshold
        self.low_conf_threshold = low_conf_threshold

    def predict(self, target_info: Dict[str, Any], candidates: List[Dict[str, Any]], language: str) -> Dict[str, Any]:
        s_res = self.structural_matcher.predict(target_info, candidates, language)
        best_score = s_res.get("best_score", 0.0)
        margin = s_res.get("margin", 0.0)

        # Certain cases:
        # 1. Definite match: high score and high margin
        # 2. Definite NEW: very low score across all candidates
        is_certain_match = (best_score >= self.high_conf_threshold and margin >= self.uncertainty_margin)
        is_certain_new = (best_score < self.low_conf_threshold)

        if is_certain_match:
            return {
                "prediction": s_res["prediction"],
                "gated_by": "structural_certain_match",
                "structural_score": best_score,
                "margin": margin,
                "confidence": best_score
            }
        elif is_certain_new:
            return {
                "prediction": "NEW",
                "gated_by": "structural_certain_new",
                "structural_score": best_score,
                "margin": margin,
                "confidence": 1.0 - best_score
            }
        else:
            # Uncertain region: trigger Laya!
            laya_res = self.laya_matcher.predict(target_info, candidates, language) if hasattr(self.laya_matcher, 'agent') else self.laya_matcher.predict(target_info, candidates)
            return {
                "prediction": laya_res["prediction"],
                "gated_by": "laya_fallback",
                "structural_score": best_score,
                "margin": margin,
                "laya_scores": laya_res.get("scores", {}),
                "confidence": laya_res.get("confidence", 0.0)
            }
