"""
Unit test for language-specific AST extractors across Python, Java, JavaScript, and TypeScript.
"""
import unittest
from src.extractors.ast_extractors import extract_features

class TestASTExtractors(unittest.TestCase):
    def test_python(self):
        code = "def process_data(items, strict=False):\n    return [x for x in items if x]"
        feats = extract_features(code, "python")
        self.assertEqual(feats.name, "process_data")
        self.assertIn("items", feats.parameters)
        self.assertGreater(len(feats.node_types), 5)

    def test_java(self):
        code = "public static int findMax(int[] arr) { int m = 0; for (int x : arr) { if (x > m) m = x; } return m; }"
        feats = extract_features(code, "java")
        self.assertEqual(feats.name, "findMax")
        self.assertIn("arr", feats.parameters)
        self.assertGreater(len(feats.node_types), 5)

    def test_javascript(self):
        code = "function handleRequest(req, res) { res.status(200).send('OK'); }"
        feats = extract_features(code, "javascript")
        self.assertEqual(feats.name, "handleRequest")
        self.assertIn("req", feats.parameters)
        self.assertGreater(len(feats.node_types), 5)

    def test_typescript(self):
        code = "export function divide(a: number, b: number): number { if (b === 0) return 0; return a / b; }"
        feats = extract_features(code, "typescript")
        self.assertEqual(feats.name, "divide")
        self.assertIn("a", feats.parameters)
        self.assertIn("b", feats.parameters)
        self.assertGreater(len(feats.node_types), 5)

if __name__ == "__main__":
    unittest.main()
