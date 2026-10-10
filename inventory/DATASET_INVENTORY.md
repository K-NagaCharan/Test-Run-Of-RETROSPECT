# Dataset Inventory: Multi-Language Historical Function Matching Benchmark

This inventory documents all primary dataset sources, licenses, schemas, ground-truth construction methods, and known limitations used in the Retrospect empirical evaluation.

---

## 1. CodeTracker Method Tracking Oracle (Java)

- **Primary Source URL:** `https://github.com/jodavimehran/code-tracker`
- **Associated Submodule:** `https://github.com/EmpiricalSEConcordia/oracle`
- **Academic Reference:**
  > Mehran Jodavi and Nikolaos Tsantalis. *Accurate Method and Variable Tracking in Commit History*. ESEC/FSE 2022.
  > Extended in: *Refactoring-aware Block Tracking in Commit History*, IEEE TSE 2024.
- **Languages:** Java
- **Underlying Repositories:** Apache Commons-IO, Apache Commons-Lang, Checkstyle, Apache Flink.
- **Dataset Size & Scope:**
  - 100 test methods and 100 training methods.
  - 844 commit transitions in the test oracle, 747 of which contain verbatim method source snapshots (`actualSource`).
- **Change Types Represented:**
  - `body change` (424 commits)
  - `container change / file rename` (179 commits)
  - `introduced / new` (97 commits)
  - `parameter change / meta change` (60+ commits)
  - `moved` (18 commits)
  - `rename` (15+ commits)
- **Ground Truth Construction:**
  - Manually constructed and verified commit-by-commit method tracking history, cross-validated against CodeShovel, Git log, and RefactoringMiner.
- **License:** MIT License.
- **Limitations:** Focuses on class-based object-oriented Java code; method overloads in the same class require signature awareness.

---

## 2. RefDiff Multi-Language Evaluation Oracle (JavaScript & TypeScript)

- **Primary Source URL:** `https://github.com/aserg-ufmg/RefDiff`
- **Academic Reference:**
  > Danilo Silva, João Paulo da Silva, and Marco Tulio Valente. *RefDiff 2.0: A Multi-language Tool for Tracking Refactorings*. IEEE Transactions on Software Engineering (TSE), 2020.
  > Preceded by: MSR 2017.
- **Languages:** JavaScript, TypeScript
- **Underlying Repositories:** `react.git`, `react-native.git`, `vue.git`, `material-ui.git`, `chartjs/Chart.js.git`, `reduxjs/redux.git`.
- **Dataset Size & Scope:**
  - 4,365 labeled refactoring instances in `refdiff-evaluation/data/js-evaluation/data.csv`.
- **Change Types Represented:**
  - `MOVE` (2,361 instances)
  - `EXTRACT_MOVE` (674 instances)
  - `RENAME` (503 instances)
  - `EXTRACT` (338 instances)
  - `INLINE` (335 instances)
  - `MOVE_RENAME` (123 instances)
- **Ground Truth Construction:**
  - AST diff extraction followed by manual validation by researchers.
- **License:** MIT License.
- **Limitations:** Predominantly frontend and component libraries; high variance in JavaScript function syntax (arrow functions, method shorthand, class methods).

---

## 3. RefactoringMiner & PyRef Benchmark (Python)

- **Primary Source URLs:**
  - RefactoringMiner Python Oracle: `https://github.com/tsantalis/RefactoringMiner/tree/master/src/test/resources/oracle/python-dataset`
  - PyRef Benchmark: `https://github.com/PyRef/PyRef/tree/main/data`
- **Academic References:**
  > Malinda Dilhara, Ameya Ketkar, and Danny Dig. *PyRef: Refactoring Detection in Python Projects*. IEEE SCAM 2021.
  > Nikolaos Tsantalis et al. *RefactoringMiner 3.0*.
- **Languages:** Python
- **Underlying Repositories:** `dit/dit.git`, `asyml/texar.git`, `kkroening/ffmpeg-python.git`.
- **Dataset Size & Scope:**
  - 202 verified commit transitions in RefactoringMiner Python oracle.
  - 573 labeled instances in PyRef (451 True Positives).
- **Change Types Represented:**
  - `Rename Method` (139 instances)
  - `Move Method` (88 instances)
  - `Extract Method` (24 instances)
  - `Add/Change Parameter` (249 instances)
- **Ground Truth Construction:**
  - Mined via AST analysis and verified by multiple human software engineering researchers (including original RefactoringMiner authors).
- **License:** MIT License.
- **Limitations:** Python indentation and dynamic typing mean function signatures lack explicit static types in older commits.

---

## 4. Curated Historical Benchmark Assembly (`benchmark.json`)

To ensure identical candidate pools, leakage-resistant evaluation, and uniform testing:
- **Total Instances:** 280
- **Validation Split (40%):** 112 instances (used strictly for tuning thresholds and hybrid weights)
- **Held-Out Test Split (60%):** 168 instances (completely untouched during tuning)
- **Language Distribution:**
  - Java: 80 instances (CodeTracker test oracle)
  - Python: 70 instances (`dit` verified history)
  - JavaScript: 70 instances (`Chart.js` & `Redux` JS history)
  - TypeScript: 60 instances (`Redux` TS v5 history)
- **Candidate Set Structure:**
  - Candidate size: 4 candidates per instance (`cand_0` through `cand_3`) plus `NEW`.
  - Realistic hard negatives: Same file/repo functions, similar names, structural distractors.
  - Candidate retrieval coverage on test set: **100.00%** (142/142 true predecessors captured).
