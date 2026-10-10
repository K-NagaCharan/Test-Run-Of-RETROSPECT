# Reproduction Guide: Multi-Language Laya Evaluation for Retrospect

This guide provides exact, turnkey instructions to reproduce the empirical evaluation from scratch on Windows 11 with Python 3.13 and Node.js.

---

## 1. Prerequisites & Environment

- **OS:** Windows 11 (AMD64)
- **Python:** 3.13.5
- **Node.js:** v22.17.1 (with npm 10.9.2)
- **Git:** Git for Windows (>= 2.50)
- **Laya:** 0.3.9 (`convaiinnovations/laya` checkpoints cached locally)
- **Required Python Packages:**
  ```bash
  pip install javalang esprima torch transformers pandas numpy psutil GitPython
  ```
- **Required Node Packages:**
  ```bash
  npm install --save-dev typescript @babel/parser
  ```

---

## 2. Directory Structure

```text
d:\base papers considered\laya testing\
├── data/
│   ├── benchmark.json            # Standardized 280-instance benchmark
│   ├── raw/                      # Downloaded oracles (CodeTracker, PyRef, RefDiff)
│   └── repos/                    # Reference Git clones (dit, redux, chartjs)
├── inventory/
│   └── DATASET_INVENTORY.md      # Full source provenance & licensing
├── src/
│   ├── adapters/
│   │   └── dataset_builder.py    # Multi-language benchmark builder
│   ├── extractors/
│   │   └── ast_extractors.py     # Native AST extractors (Java, Python, JS, TS)
│   ├── models/
│   │   ├── baselines.py          # Lexical and Structural AST matchers
│   │   ├── laya_embedder.py      # Laya ModernBERT embedding matcher
│   │   └── laya_decision.py      # Laya Router/Decision Head matcher
│   ├── hybrid/
│   │   └── hybrid_matchers.py    # Hybrid E1, E2, E3 matchers
│   ├── metrics/
│   │   └── evaluator.py          # Top-1, MRR, F1, FPR, ECE, Brier score
│   ├── download_datasets.py      # Automated oracle downloader
│   ├── runner.py                 # Master evaluation orchestrator
│   └── show_errors.py            # Qualitative error inspection utility
├── tests/
│   ├── test_ast_extractors.py    # Unit tests for multi-language AST extraction
│   └── test_runner.py            # End-to-end pipeline smoke tests
├── results/
│   ├── predictions/              # Raw prediction JSONL files per model
│   └── tables/                   # CSV and JSON aggregate and sliced tables
└── RESEARCH_REPORT.md            # Comprehensive empirical research report
```

---

## 3. Step-by-Step Reproduction Instructions

### Step 1: Run Automated Tests
Verify that all language parsers and evaluation pipelines operate correctly:
```powershell
python -m unittest tests/test_ast_extractors.py
python -m unittest tests/test_runner.py
```

### Step 2: Download Raw Benchmarks & Repositories (Optional if already cached)
```powershell
python src/download_datasets.py
```

### Step 3: Rebuild Benchmark Dataset
Regenerate the standardized 280-instance benchmark with a 40% validation / 60% test split:
```powershell
python -m src.adapters.dataset_builder
```

### Step 4: Execute Full Empirical Evaluation
Run all 9 configurations across parameter tuning and held-out testing:
```powershell
python -m src.runner
```

Outputs are automatically generated in:
- `results/tables/aggregate_metrics.csv`
- `results/tables/per_language_metrics.csv`
- `results/tables/per_change_type_metrics.csv`
- `results/tables/efficiency_metrics.csv`
- `results/tables/error_cases.json`
- `results/predictions/*.jsonl`

### Step 5: Inspect Error Cases
```powershell
python src/show_errors.py
```
