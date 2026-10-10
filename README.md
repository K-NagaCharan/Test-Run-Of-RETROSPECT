# Retrospect: Historical Function Matching & Multi-Language Evaluation Engine

This repository contains the **Historical Function Matching Engine** and **Empirical Evaluation Harness** developed for **Retrospect: Evolution-Aware Context Retrieval for Repository-Level LLMs**.

Retrospect identifies which historical code is relevant to the current codebase and explains how and why it evolved across Git commits. This repository provides the high-performance, deterministic historical matching pipeline and the empirical evaluation that tested whether local neural decision engines (Laya 0.3.9) add value over deterministic techniques.

---

## 🚀 Key Empirical Findings

Evaluated on **280 authentic function evolution instances** spanning **Java, Python, JavaScript, and TypeScript** from established academic benchmarks (CodeTracker, RefDiff, RefactoringMiner, PyRef):

- **Deterministic Lexical Matching dominates:** **98.21% Top-1 Accuracy** ($MRR = 0.9821, \text{Macro-F1} = 0.9642$) at **0.80 ms** latency per example.
- **Deterministic Structural Matching provides strong refactoring tracking:** **86.90% Top-1 Accuracy** ($MRR = 0.8750$) at **9.27 ms** latency per example.
- **Laya ModernBERT Embeddings degrade performance:** **82.14% Top-1 Accuracy** ($1,501.4\text{ ms}$ latency), with **0.0% recall on `NEW` functions** due to representation collapse on code syntax ($>0.95$ cosine similarity across unrelated functions).
- **Laya Decision Head collapses on code refactorings:** **67.26% Top-1 Accuracy** for English and **27.98%** for Multilingual (with a $73.24\%$ False Positive Rate on new functions).
- **Hybrid Grid Tuning:** In systematic validation tuning, the optimal weight for Laya was **$0.00$** ($w_{struct}=0.6, w_{lex}=0.4, w_{laya}=0.0$). Adding neural router signal strictly decreased precision.

---

## 📂 Repository Structure

```text
├── data/
│   ├── benchmark.json            # 280-instance multi-language verified benchmark
│   └── raw/                      # Downloaded reference oracles (CodeTracker, RefDiff, PyRef)
├── inventory/
│   └── DATASET_INVENTORY.md      # Dataset provenance, licenses, and transformation types
├── src/
│   ├── extractors/
│   │   └── ast_extractors.py     # Multi-language AST parsers (Java, Python, JS, TS)
│   ├── models/
│   │   ├── baselines.py          # Deterministic Lexical & Structural AST matchers
│   │   ├── laya_embedder.py      # Laya ModernBERT embedding matcher (English & Multi)
│   │   └── laya_decision.py      # Laya Router / Decision Head matcher
│   ├── hybrid/
│   │   └── hybrid_matchers.py    # Hybrid E1 (rerank), E2 (blend), E3 (gated fallback)
│   ├── metrics/
│   │   └── evaluator.py          # Top-1 Acc, MRR, Precision, Recall, F1, Brier, ECE
│   ├── adapters/
│   │   └── dataset_builder.py    # Multi-language dataset builder
│   ├── download_datasets.py      # Automated benchmark downloader
│   ├── runner.py                 # Master evaluation orchestrator
│   └── show_errors.py            # Qualitative failure analysis viewer
├── tests/
│   ├── test_ast_extractors.py    # Unit tests for Python, Java, JS, and TS AST extractors
│   └── test_runner.py            # End-to-end pipeline smoke tests
├── results/
│   ├── predictions/              # Raw prediction JSONL files for each model
│   └── tables/                   # CSV and JSON aggregate and sliced performance tables
├── REPRODUCTION.md               # Step-by-step reproduction instructions
├── RESEARCH_REPORT.md            # In-depth research report with RQ1-RQ6 answers
└── package.json                  # Node dependencies for TypeScript AST extraction
```

---

## 🛠 Reusing the Historical Retrieval Engine in Retrospect

The core deterministic matchers in `src/models/baselines.py` and `src/extractors/ast_extractors.py` are self-contained and ready to be integrated directly into Retrospect.

### 1. Fast Lexical Predecessor Matching (< 1 ms)
```python
from src.models.baselines import LexicalMatcher

matcher = LexicalMatcher(new_threshold=0.25)

target_function = {
    "name": "calculate_tax",
    "source": "def calculate_tax(amount, rate):\n    return amount * rate"
}

candidate_predecessors = [
    {"candidate_id": "cand_0", "name": "calc_tax", "source": "def calc_tax(amt, rate):\n    return amt * rate"},
    {"candidate_id": "cand_1", "name": "get_user", "source": "def get_user(uid):\n    return db.fetch(uid)"}
]

result = matcher.predict(target_function, candidate_predecessors)
print(result["prediction"])  # -> "cand_0"
print(result["confidence"])  # -> 0.85
```

### 2. Language-Specific AST Structural Differencing
```python
from src.models.baselines import StructuralMatcher

struct_matcher = StructuralMatcher(new_threshold=0.30)
result = struct_matcher.predict(target_function, candidate_predecessors, language="python")
print(result["prediction"])  # Matches based on AST node types, arguments, and statement flow
```

---

## 🧪 Running Unit Tests & Benchmarks

### 1. Verify Unit Tests
```bash
python -m unittest tests/test_ast_extractors.py
python -m unittest tests/test_runner.py
```

### 2. Run the Full Empirical Evaluation
```bash
python -m src.runner
```
Outputs will be logged in `results/tables/aggregate_metrics.csv` and `results/predictions/`.

### 3. Inspect Error Cases
```bash
python src/show_errors.py
```

---

## 📜 Academic References & Citations

- **CodeTracker:** Mehran Jodavi and Nikolaos Tsantalis. *Accurate Method and Variable Tracking in Commit History*. ESEC/FSE 2022.
- **RefDiff 2.0:** Danilo Silva, João Paulo da Silva, and Marco Tulio Valente. *RefDiff 2.0: A Multi-language Tool for Tracking Refactorings*. IEEE TSE 2020.
- **PyRef:** Malinda Dilhara, Ameya Ketkar, and Danny Dig. *PyRef: Refactoring Detection in Python Projects*. IEEE SCAM 2021.
- **RefactoringMiner:** Nikolaos Tsantalis et al. *RefactoringMiner 3.0*.
