"""
Smoke test to verify runner and evaluation pipeline on sample instances.
"""
import unittest
from src.models.baselines import LexicalMatcher, StructuralMatcher
from src.metrics.evaluator import compute_metrics

class TestRunnerPipeline(unittest.TestCase):
    def test_pipeline_smoke(self):
        toy_inst = {
            "instance_id": "test_1",
            "language": "python",
            "target_function": {"name": "foo", "source": "def foo(x):\n    return x + 1"},
            "candidates": [
                {"candidate_id": "cand_0", "name": "foo", "source": "def foo(x):\n    return x + 1"},
                {"candidate_id": "cand_1", "name": "bar", "source": "def bar(y):\n    return y * 2"}
            ],
            "ground_truth_id": "cand_0"
        }

        lex = LexicalMatcher()
        res_lex = lex.predict(toy_inst["target_function"], toy_inst["candidates"])
        self.assertEqual(res_lex["prediction"], "cand_0")

        struct = StructuralMatcher()
        res_struct = struct.predict(toy_inst["target_function"], toy_inst["candidates"], "python")
        self.assertEqual(res_struct["prediction"], "cand_0")

        toy_preds = [
            {"ground_truth_id": "cand_0", "prediction": "cand_0", "confidence": 0.95, "scores": {"cand_0": 0.95, "cand_1": 0.1}},
            {"ground_truth_id": "NEW", "prediction": "NEW", "confidence": 0.8, "scores": {"cand_0": 0.2, "cand_1": 0.1}}
        ]
        metrics = compute_metrics(toy_preds)
        self.assertEqual(metrics["top1_accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)

if __name__ == "__main__":
    unittest.main()
