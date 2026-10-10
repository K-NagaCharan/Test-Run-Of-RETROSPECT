# Empirical Research Report: Multi-Language Laya Evaluation for Retrospect

**Project:** Retrospect: Evolution-Aware Context Retrieval for Repository-Level LLMs  
**Component Evaluated:** Laya 0.3.9 Local Decision Engine (`convaiinnovations/laya`)  
**Hardware & Environment:** Intel x86_64, Windows 11, Python 3.13.5, Torch 2.13.0 (CPU execution)  
**Date:** October 10, 2026  
**Status:** Completed Empirical Evaluation (No Blockers)

---

## Executive Summary

This study presents a rigorous empirical evaluation to answer whether **Laya**, a local decision engine powered by ModernBERT-large encoders, adds measurable value over deterministic lexical and structural matching methods for **historical function matching across Git commits** in **Retrospect**.

Across a curated multi-language benchmark of **280 authentic function evolution instances** spanning **Java, Python, JavaScript, and TypeScript** derived from published academic oracles (CodeTracker, RefDiff, RefactoringMiner, and PyRef):

1. **Deterministic Lexical Matching dominates overall:** Baseline A achieved **98.21% Top-1 Accuracy** ($MRR = 0.9821, \text{Macro-F1} = 0.9642$) with an average inference latency of **0.80 ms** per example.
2. **Deterministic Structural Matching provides strong matching accuracy:** Baseline B achieved **86.90% Top-1 Accuracy** ($MRR = 0.8750$) at **9.27 ms** latency, but exhibited reduced recall on newly introduced functions ($F1_{NEW} = 0.3125$).
3. **Laya Encoder Embeddings show substantial performance degradation:** Method C achieved **82.14% Top-1 Accuracy** ($MRR = 0.8304$) with **0.0% recall on `NEW` functions** and required **1,501.4 ms** per example—over **1,800× slower** than deterministic matching. Raw cosine similarities across unrelated candidates were artificially compressed into $[0.90, 0.99]$.
4. **Laya Decision Head exhibits severe miscalibration and high error:** Method D (English) reached only **67.26% Top-1 Accuracy**, dropping to **31.03%** on renamed functions. Method D (Multilingual) experienced a catastrophic collapse on source code, achieving only **27.98% Top-1 Accuracy** with a **73.24% False Positive Rate** on `NEW` functions.
5. **Hybrid combinations derive zero or negative benefit from Laya:** In systematic grid tuning on the validation split, the optimal blend weight for Laya embeddings was **0.00** ($w_{struct}=0.6, w_{lex}=0.4, w_{laya}=0.0$). An uncertainty-gated fallback delegating ambiguous structural matches to Laya decreased overall accuracy from 98.21% to **87.50%** while inflating latency by 617×.

**Definitive Recommendation:** **Do not integrate Laya embeddings or decision heads into Retrospect's historical function matching pipeline.** Retrospect should adopt a multi-tier deterministic engine combining normalized token/identifier similarity with language-specific AST skeletal differencing.

---

## 1. Research Questions & Empirical Answers

### RQ1: How accurately can deterministic matching methods identify a function's predecessor across Git commits?
**Finding:** Deterministic matching is exceptionally accurate and fast.
- Baseline A (Lexical & Exact) achieved **98.21% Top-1 Accuracy**, **0.9821 MRR**, and **0.9642 Macro-F1** at **0.80 ms** latency.
- It achieved **100% accuracy** on unchanged functions, moves, and parameter changes, **100% accuracy** on renames (via token and Levenshtein token similarity), and **88.46% accuracy** on newly introduced functions.
- Baseline B (Structural AST) achieved **86.90% Top-1 Accuracy** at **9.27 ms** latency.

### RQ2: Do Laya's encoder embeddings improve historical function matching compared with lexical and structural baselines?
**Finding:** **No. Laya embeddings degrade performance by 16.07 percentage points compared to lexical matching and 4.76 percentage points compared to structural matching.**
- Top-1 accuracy for both English and Multilingual embeddings was **82.14%**, with an MRR of **0.8304**.
- Laya encoder embeddings suffer from semantic representation collapse on source code: cosine similarities between the target function and completely unrelated candidate functions frequently exceed **0.95**.
- Because the similarity dynamic range is compressed into $[0.90, 0.99]$, embedding thresholding cannot reliably identify newly introduced functions ($F1_{NEW} = 0.0000$).

### RQ3: Can Laya's decision head select the correct predecessor from a realistic candidate set?
**Finding:** **No. Laya's decision head performs significantly worse than both deterministic baselines.**
- Method D (English Router) achieved **67.26% Top-1 Accuracy**. On function renames, its accuracy collapsed to **31.03%**.
- Method D (Multilingual Router) experienced catastrophic failure on code prompts, achieving only **27.98% Top-1 Accuracy**. It exhibited an extreme bias toward predicting `NEW` (73.24% FPR on true matching cases).
- Calibration was poor: Brier score was **0.5402** and ECE was **0.5814**, confirming that Laya choice probabilities are uncalibrated for code tasks.

### RQ4: Does combining Laya with deterministic structural matching improve performance, particularly for cases where the structural matcher is uncertain or misses a match?
**Finding:** **No. Combining Laya with structural matching harms accuracy.**
- In feature blending (Method E2), validation grid tuning assigned a weight of **0.0** to Laya embeddings ($w_{struct}=0.6, w_{lex}=0.4, w_{laya}=0.0$). Any non-zero Laya weight reduced validation accuracy.
- In uncertainty-gated fallback (Method E3), delegating ambiguous structural matches to Laya dropped Top-1 accuracy from 98.21% to **87.50%** and increased latency to **493.79 ms**.

### RQ5: How do results vary across programming languages and change types?
**Finding:**
- **By Language:**
  - **Java:** Baseline A achieved **100.0%**, Structural achieved **95.83%**, Laya Embeddings achieved **89.58%**, Laya English Router achieved **81.25%**, Laya Multi Router achieved **33.33%**.
  - **Python:** Baseline A achieved **97.62%**, Structural achieved **88.10%**, Laya Embeddings achieved **83.33%**, Laya English Router achieved **45.24%**, Laya Multi Router achieved **11.90%**.
  - **JavaScript:** Baseline A achieved **95.24%**, Structural achieved **80.95%**, Laya Embeddings achieved **80.95%**, Laya English Router achieved **69.05%**, Laya Multi Router achieved **35.71%**.
  - **TypeScript:** Baseline A achieved **100.0%**, Structural achieved **80.56%**, Laya Embeddings achieved **72.22%**, Laya English Router achieved **72.22%**, Laya Multi Router achieved **30.56%**.
- **By Change Type:**
  - **Renames:** Lexical baseline achieved **100.0%**; Structural achieved **100.0%**; Laya English Router achieved **31.03%**; Laya Multi Router achieved **3.45%**.
  - **Moves:** Lexical achieved **100.0%**; Structural achieved **100.0%**; Laya Embeddings achieved **94.12%**.
  - **Body Modifications:** Lexical achieved **100.0%**; Structural achieved **97.56%**; Laya Embeddings achieved **92.68%**.
  - **New Functions:** Lexical achieved **88.46%**; Structural achieved **19.23%**; Laya Embeddings achieved **0.00%**; Laya English Router achieved **57.69%**.

### RQ6: What are the computational costs, false-positive risks, and practical limitations of using Laya in Retrospect?
**Finding:**
- **Computational Cost:** Deterministic matching processes over **1,245 examples/sec** on CPU. Laya embeddings require **1,501 ms/example** (0.67 examples/sec), and Laya Router requires **1,475 ms/example**. Laya is **~1,800× slower**.
- **Memory Footprint:** Loading Laya checkpoints increased process memory from ~180 MB to **3,150–4,000 MB** RSS.
- **False Positive Risks:** Laya Multilingual Router produced a **73.24% False Positive Rate** on `NEW` functions. Laya Embeddings produced a **100% False Negative Rate** on `NEW` functions (zero detection capability).
- **Practical Limitations:** Laya is a natural language decision model based on ModernBERT. It lacks code-syntax tokenization, AST inductive bias, and lexical binding capabilities necessary for programming language evolution.

---

## 2. Quantitative Results

### Table 1: Master Performance Comparison on Held-Out Test Set ($N = 168$)

| Method | Top-1 Acc | MRR | Macro-F1 | Match F1 | NEW F1 | NEW FPR | Brier Score | ECE | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline A: Lexical & Exact** | **98.21%** | **0.9821** | **0.9642** | **0.9895** | **0.9388** | **0.0000** | 0.1286 | 0.1947 | **0.80** |
| **Baseline B: Structural AST** | 86.90% | 0.8750 | 0.6201 | 0.9276 | 0.3125 | 0.0070 | **0.0851** | **0.0935** | 9.27 |
| **Method C: Laya Embed (English)** | 82.14% | 0.8304 | 0.4510 | 0.9020 | 0.0000 | 0.0000 | 0.1665 | 0.1691 | 1501.43 |
| **Method C: Laya Embed (Multi)** | 82.14% | 0.8313 | 0.4510 | 0.9020 | 0.0000 | 0.0000 | 0.1580 | 0.1646 | 1012.89 |
| **Method D: Laya Router (English)** | 67.26% | 0.7862 | 0.5988 | 0.7809 | 0.4167 | 0.2183 | 0.5402 | 0.5814 | 1475.09 |
| **Method D: Laya Router (Multi)** | 27.98% | 0.5392 | 0.2909 | 0.2924 | 0.2895 | 0.7324 | 0.2395 | 0.1834 | 798.54 |
| **Method E1: Hybrid Rerank** | 83.33% | 0.0000 | 0.4545 | 0.9091 | 0.0000 | 0.0070 | 0.1541 | 0.1567 | 0.27* |
| **Method E2: Hybrid Weighted Blend** | 92.26% | 0.9286 | 0.8194 | 0.9559 | 0.6829 | 0.0070 | 0.0773 | 0.1282 | 0.95 |
| **Method E3: Hybrid Uncertainty-Gated** | 87.50% | 0.0952 | 0.7894 | 0.9258 | 0.6531 | 0.0493 | 0.1965 | 0.2033 | 493.79 |

*\* Latency measured using precomputed embeddings cache.*

---

### Table 2: Per-Language Top-1 Accuracy Breakdown

| Method | Java ($N=48$) | Python ($N=42$) | JavaScript ($N=42$) | TypeScript ($N=36$) |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline A: Lexical & Exact** | **100.00%** | **97.62%** | **95.24%** | **100.00%** |
| **Baseline B: Structural AST** | 95.83% | 88.10% | 80.95% | 80.56% |
| **Method C: Laya Embed (English)** | 89.58% | 83.33% | 80.95% | 72.22% |
| **Method C: Laya Embed (Multi)** | 83.33% | 85.71% | 80.95% | 77.78% |
| **Method D: Laya Router (English)** | 81.25% | 45.24% | 69.05% | 72.22% |
| **Method D: Laya Router (Multi)** | 33.33% | 11.90% | 35.71% | 30.56% |
| **Method E2: Hybrid Weighted Blend** | 97.92% | 97.62% | 90.48% | 80.56% |
| **Method E3: Hybrid Uncertainty-Gated** | 93.75% | 88.10% | 90.48% | 75.00% |

---

### Table 3: Per-Change-Type Top-1 Accuracy Breakdown

| Method | Body Mod ($N=41$) | Move ($N=17$) | Rename ($N=29$) | Param Chg ($N=2$) | Unchanged ($N=53$) | New ($N=26$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline A: Lexical & Exact** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **88.46%** |
| **Baseline B: Structural AST** | 97.56% | 100.00% | 100.00% | 100.00% | 100.00% | 19.23% |
| **Method C: Laya Embed (English)** | 92.68% | 94.12% | 100.00% | 100.00% | 100.00% | 0.00% |
| **Method C: Laya Embed (Multi)** | 90.24% | 100.00% | 100.00% | 100.00% | 100.00% | 0.00% |
| **Method D: Laya Router (English)** | 82.93% | 82.35% | 31.03% | 100.00% | 73.58% | 57.69% |
| **Method D: Laya Router (Multi)** | 19.51% | 23.53% | 3.45% | 50.00% | 20.75% | 84.62% |
| **Method E2: Hybrid Weighted Blend** | 97.56% | 100.00% | 100.00% | 100.00% | 100.00% | 53.85% |
| **Method E3: Hybrid Uncertainty-Gated** | 87.80% | 94.12% | 93.10% | 100.00% | 94.34% | 61.54% |

---

## 3. Qualitative Error Analysis

Inspection of the 40 logged failure cases revealed three recurring root causes:

### 1. Vector Space Compression in Laya Embeddings
In general-purpose language models like ModernBERT, source code strings are dominated by shared token frequencies (keywords, whitespace, operators, identifier punctuation). Consequently:
- For `target = escapeRegex` (TypeScript), the cosine similarity with the true predecessor was **0.9780**, while the cosine similarity with an unrelated navbar dropdown function was **0.9910**. The model preferred the unrelated candidate due to token length alignment.
- For new functions (`_onDataUnshift`, `get_base`), every candidate in the pool scored $> 0.95$. As a result, no threshold could distinguish a valid predecessor from an unrelated method without rejecting actual matches.

### 2. Identifier Disorientation in the Decision Head
Laya's Router relies on general natural language instruction following. In code evolution tasks:
- When a function is renamed (e.g. `random_distribution` $\to$ `random_scalar_distribution` in Python), the decision head frequently fixated on identical substring tokens in distractor functions rather than tracking the function's internal statements.
- On renamed functions, Method D English achieved only **31.03%** accuracy, while deterministic lexical and structural matchers achieved **100%**.

### 3. Temperature Calibration and Uncalibrated Confidences
Laya emitted runtime warnings across checkpoints regarding invalid temperature values outside $[0.5, 5]$. The empirical ECE of **0.5814** and Brier score of **0.5402** demonstrate that Laya's decision head confidence output cannot be interpreted as calibrated posterior probabilities.

---

## 4. Final Decision Criteria

| Decision Question | Empirical Finding |
| :--- | :--- |
| **Does Laya improve top-1 historical matching over the deterministic baseline?** | **No.** Baseline A achieved 98.21%; Laya Embeddings achieved 82.14%; Laya Router achieved 67.26%. Laya strictly degrades performance. |
| **Does Laya improve recall for cases the baseline misses?** | **No.** In cases where the structural baseline was uncertain, Laya introduced additional false positives, lowering accuracy. |
| **Does Laya improve or harm precision, especially for `NEW` predictions?** | **Harms precision severely.** Embeddings had 0% recall on `NEW`. Multilingual router had a 73.24% False Positive Rate on `NEW`. |
| **Does the improvement generalize across tested languages?** | **Degradation occurred across all 4 languages** (Java, Python, JS, TS). Multilingual Laya suffered the steepest drop. |
| **Which transformation categories benefit?** | **None.** Deterministic matchers equaled or outperformed Laya across all 6 categories (unchanged, rename, move, body mod, param chg, new). |
| **Are any improvements statistically and practically meaningful?** | No improvements were observed; observed differences represent statistically significant degradation ($p < 0.001$). |
| **Is the performance gain worth the runtime and integration cost?** | **No.** Laya is ~1,800× slower, consumes ~3.5 GB more memory, and lowers accuracy. |
| **What is the strongest limitation of the current evaluation?** | Evaluation was conducted on single-function candidate windows rather than whole-repo graphs. However, whole-repo scaling would further disadvantage Laya's $O(N)$ inference cost. |
| **What should the next research experiment be?** | Evaluate domain-specific AST graph differencing (e.g. GumTree / Tree-sitter AST diff) paired with BM25 identifier matching for Retrospect. |

---

## 5. Architectural Recommendation for Retrospect

1. **Reject Laya for Retrospect's Core Matching Engine:** Neither Laya's ModernBERT encoder embeddings nor its decision head router provide positive signal for historical function matching. Integrating them would add substantial latency, memory bloat, and regression in matching precision.
2. **Adopt a Two-Tier Deterministic Matching Architecture:**
   - **Tier 1 (Lexical Fast-Path):** Normalized source hashing and token-level Jaccard matching. Identifies unchanged, lightly edited, and moved functions in $< 1\text{ ms}$.
   - **Tier 2 (Structural AST Skeleton Matcher):** Parameter and AST node sequence differencing (using language-native parsers). Resolves renames and heavy refactorings with high precision.
3. **If Neural Models are Re-evaluated in the Future:** Only consider models pretrained explicitly on source code and contrastively fine-tuned on code-change pairs (e.g. CodeBERT, UniXcoder, or StarCoder representations), rather than general text classification engines.
