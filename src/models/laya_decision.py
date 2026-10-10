"""
Method D: Laya Decision Head Matcher
Uses Agent.predict() with typed choice question schema.
Supports English, Multilingual, and Typed-Decisions checkpoints.
"""
from typing import Dict, Any, List, Optional
import laya

class LayaDecisionMatcher:
    def __init__(self, checkpoint_type: str = "english", device: str = "cpu"):
        self.checkpoint_type = checkpoint_type
        self.device = device

        print(f"Loading Laya decision head model ({checkpoint_type}) on {device}...")
        if checkpoint_type == "multilingual":
            self.agent = laya.load("convaiinnovations/laya", device=device, subfolder="multilingual")
        elif checkpoint_type == "typed-decisions":
            self.agent = laya.load("convaiinnovations/laya", device=device, subfolder="typed-decisions")
        else:
            self.agent = laya.load("convaiinnovations/laya", device=device)
        print("Laya decision head model ready.")

    def _truncate_code(self, code: str, max_lines: int = 15, max_chars: int = 400) -> str:
        lines = code.splitlines()[:max_lines]
        truncated = "\n".join(lines)
        if len(truncated) > max_chars:
            truncated = truncated[:max_chars] + "..."
        return truncated

    def predict(self, target_info: Dict[str, Any], candidates: List[Dict[str, Any]], language: str = "") -> Dict[str, Any]:
        target_name = target_info.get("name", "")
        target_src = target_info.get("source", "")
        trunc_target = self._truncate_code(target_src)

        state = (
            f"Programming Language: {language}\n"
            f"Target Function: {target_name}\n"
            f"Target Code:\n{trunc_target}\n\n"
            f"Task: Historical function predecessor matching across Git commits."
        )

        criteria = {}
        for c in candidates:
            cid = c["candidate_id"]
            c_name = c.get("name", "")
            c_src = self._truncate_code(c.get("source", ""))
            criteria[cid] = f"Candidate '{c_name}':\n{c_src}"

        criteria["NEW"] = "No predecessor exists in the earlier commit (this is a newly introduced function)."

        questions = {
            "match": {
                "type": "choice",
                "instructions": (
                    "Select the candidate that corresponds to the earlier historical version of the target function, "
                    "or select NEW if this is a newly introduced function."
                ),
                "criteria": criteria
            }
        }

        try:
            res = self.agent.predict(state, questions)
            ans = res.get("answers", {}).get("match", {})
            choice = ans.get("choice", "NEW")
            probs = ans.get("probabilities", {})
            conf = ans.get("confidence", 0.0)
            tokens = res.get("usage", {}).get("input_tokens", 0)

            return {
                "prediction": choice,
                "scores": probs,
                "confidence": conf,
                "tokens": tokens,
                "raw_result": ans
            }
        except Exception as e:
            # Fallback if forward pass errors
            print(f"Error in Laya decision head predict: {e}")
            fallback_scores = {cid: 1.0 / (len(candidates) + 1) for cid in criteria}
            return {
                "prediction": "NEW",
                "scores": fallback_scores,
                "confidence": 0.0,
                "tokens": 0,
                "error": str(e)
            }
