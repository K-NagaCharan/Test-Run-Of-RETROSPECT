"""
Deterministic Baselines for Historical Function Matching:
- Baseline A: Lexical and Exact Matcher (Normalized Source, Token Jaccard, Name Levenshtein)
- Baseline B: Structural AST Matcher (AST Node Sequence Jaccard, Parameter Match, Statement Ratio)
"""
import re
from typing import Dict, Any, List, Tuple
from src.extractors.ast_extractors import extract_features, ASTFeatures

def normalize_code(code: str) -> str:
    # Remove single line comments
    code = re.sub(r'(?://|#).*$', '', code, flags=re.MULTILINE)
    # Remove multi-line comments
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    # Normalize whitespace
    tokens = code.split()
    return " ".join(tokens)

def tokenize_code(code: str) -> List[str]:
    # Extract alphanumeric identifiers and keywords
    return re.findall(r'[a-zA-Z_]\w*', code)

def jaccard_similarity(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

def levenshtein_similarity(s1: str, s2: str) -> float:
    s1, s2 = s1.lower(), s2.lower()
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    dist = dp[m][n]
    max_len = max(m, n)
    return max(0.0, 1.0 - dist / max_len)

class LexicalMatcher:
    def __init__(self, new_threshold: float = 0.35):
        self.new_threshold = new_threshold

    def score_candidate(self, target_src: str, target_name: str, cand_src: str, cand_name: str) -> float:
        # 1. Exact normalized match
        norm_t = normalize_code(target_src)
        norm_c = normalize_code(cand_src)
        if norm_t == norm_c and norm_t != "":
            return 1.0

        # 2. Name similarity
        name_sim = levenshtein_similarity(target_name, cand_name)

        # 3. Token Jaccard
        tokens_t = set(tokenize_code(target_src))
        tokens_c = set(tokenize_code(cand_src))
        token_sim = jaccard_similarity(tokens_t, tokens_c)

        # Composite lexical score
        # 40% name similarity, 60% token similarity
        return 0.4 * name_sim + 0.6 * token_sim

    def predict(self, target_info: Dict[str, Any], candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        target_src = target_info.get("source", "")
        target_name = target_info.get("name", "")

        scores = {}
        for c in candidates:
            cid = c["candidate_id"]
            c_src = c.get("source", "")
            c_name = c.get("name", "")
            scores[cid] = self.score_candidate(target_src, target_name, c_src, c_name)

        if not scores:
            return {"prediction": "NEW", "scores": {}, "confidence": 1.0}

        best_cand = max(scores, key=scores.get)
        best_score = scores[best_cand]

        # If best score is below new_threshold, predict NEW
        prediction = best_cand if best_score >= self.new_threshold else "NEW"

        return {
            "prediction": prediction,
            "scores": scores,
            "best_candidate": best_cand,
            "best_score": best_score,
            "confidence": best_score
        }

class StructuralMatcher:
    def __init__(self, new_threshold: float = 0.30):
        self.new_threshold = new_threshold
        self._feat_cache = {}

    def get_features(self, src: str, lang: str) -> ASTFeatures:
        key = (src, lang)
        if key not in self._feat_cache:
            self._feat_cache[key] = extract_features(src, lang)
        return self._feat_cache[key]

    def score_candidate(self, target_feats: ASTFeatures, cand_feats: ASTFeatures) -> float:
        # 1. Exact normalized match
        if target_feats.raw_source and target_feats.raw_source == cand_feats.raw_source:
            return 1.0

        # 2. Name similarity
        name_sim = levenshtein_similarity(target_feats.name, cand_feats.name)

        # 3. Parameter match
        p_t = set(target_feats.parameters)
        p_c = set(cand_feats.parameters)
        param_sim = jaccard_similarity(p_t, p_c)

        # 4. AST node types sequence similarity
        nodes_t = set(target_feats.node_types)
        nodes_c = set(cand_feats.node_types)
        node_sim = jaccard_similarity(nodes_t, nodes_c)

        # 5. Statement count ratio
        s_t = max(1, target_feats.statement_count)
        s_c = max(1, cand_feats.statement_count)
        stmt_sim = min(s_t, s_c) / max(s_t, s_c)

        # Weighted structural score:
        # 30% name, 25% AST nodes, 25% parameters, 20% statement ratio
        score = 0.30 * name_sim + 0.25 * node_sim + 0.25 * param_sim + 0.20 * stmt_sim
        return score

    def predict(self, target_info: Dict[str, Any], candidates: List[Dict[str, Any]], language: str) -> Dict[str, Any]:
        target_src = target_info.get("source", "")
        target_feats = self.get_features(target_src, language)

        scores = {}
        for c in candidates:
            cid = c["candidate_id"]
            c_src = c.get("source", "")
            c_feats = self.get_features(c_src, language)
            scores[cid] = self.score_candidate(target_feats, c_feats)

        if not scores:
            return {"prediction": "NEW", "scores": {}, "confidence": 1.0}

        best_cand = max(scores, key=scores.get)
        best_score = scores[best_cand]

        # Margin between top-1 and top-2
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
