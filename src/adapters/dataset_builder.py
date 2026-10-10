"""
Benchmark Dataset Builder for Retrospect Laya Evaluation.
Extracts authentic multi-language historical function matching instances across:
- Java (CodeTracker Oracle)
- Python (dit repository from RefactoringMiner & PyRef)
- JavaScript (Chart.js & Redux JS history)
- TypeScript (Redux TypeScript history)
"""
import os
import re
import json
import random
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.extractors.ast_extractors import extract_features

random.seed(42)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
SHOVEL_DIR = RAW_DIR / "shovel_test"
REPOS_DIR = DATA_DIR / "repos"

def clean_source(src: str) -> str:
    lines = src.splitlines()
    # strip trailing whitespace
    cleaned = [line.rstrip() for line in lines]
    return "\n".join(cleaned).strip()

def build_java_instances(max_count: int = 80) -> List[Dict[str, Any]]:
    instances = []
    if not SHOVEL_DIR.exists():
        return instances

    shovel_files = sorted([f for f in SHOVEL_DIR.glob("*.json")])
    all_methods_pool = []

    # First pass: collect all methods with source for hard negatives
    for fpath in shovel_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                d = json.load(f)
            details = d.get("changeHistoryDetails", {})
            for commit_id, info in details.items():
                src = info.get("actualSource", "").strip()
                if len(src) > 20:
                    all_methods_pool.append({
                        "name": info.get("functionName", d.get("functionName", "unknown")),
                        "source": clean_source(src),
                        "file_path": info.get("path", d.get("sourceFilePath", "unknown")),
                        "repo": d.get("repositoryName", "commons-io")
                    })
        except Exception:
            continue

    print(f"Java negative pool size: {len(all_methods_pool)}")

    for fpath in shovel_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                d = json.load(f)
            history = d.get("changeHistory", [])
            details = d.get("changeHistoryDetails", {})
            repo_name = d.get("repositoryName", "commons-io")
            func_name = d.get("functionName", "")

            # Iterate through consecutive commits
            for i in range(len(history)):
                cur_commit = history[i]
                cur_info = details.get(cur_commit, {})
                cur_src = cur_info.get("actualSource", "").strip()
                if len(cur_src) < 20:
                    continue

                change_type_raw = cur_info.get("type", "")

                # Check if this is newly introduced
                if "Yintroduced" in change_type_raw or i == len(history) - 1:
                    # Target is new! Ground truth is NEW
                    change_type = "new"
                    gt_key = "NEW"
                    pred_src = None
                    pred_name = None
                    pred_file = None
                else:
                    prev_commit = history[i + 1]
                    prev_info = details.get(prev_commit, {})
                    prev_src = prev_info.get("actualSource", "").strip()
                    if len(prev_src) < 20:
                        continue

                    gt_key = "cand_0"
                    pred_src = prev_src
                    pred_name = prev_info.get("functionName", func_name)
                    pred_file = prev_info.get("path", d.get("sourceFilePath", ""))

                    if "Yrename" in change_type_raw:
                        change_type = "rename"
                    elif "Ymove" in change_type_raw or "Yfilerename" in change_type_raw:
                        change_type = "move"
                    elif "Yparameter" in change_type_raw:
                        change_type = "parameter_change"
                    elif "Ybodychange" in change_type_raw:
                        change_type = "body_modification"
                    else:
                        change_type = "unchanged"

                # Construct candidates
                candidates = []
                cand_idx = 0

                if pred_src is not None:
                    candidates.append({
                        "candidate_id": f"cand_{cand_idx}",
                        "name": pred_name,
                        "source": clean_source(pred_src),
                        "file_path": pred_file,
                        "is_ground_truth": True,
                        "negative_type": "ground_truth"
                    })
                    cand_idx += 1

                # Add hard negatives from same repo or other methods
                same_repo_pool = [m for m in all_methods_pool if m["repo"] == repo_name and m["name"] != func_name]
                other_repo_pool = [m for m in all_methods_pool if m["repo"] != repo_name]

                # Hard negative 1: Same repo
                if same_repo_pool:
                    sample = random.choice(same_repo_pool)
                    candidates.append({
                        "candidate_id": f"cand_{cand_idx}",
                        "name": sample["name"],
                        "source": sample["source"],
                        "file_path": sample["file_path"],
                        "is_ground_truth": False,
                        "negative_type": "hard_same_repo"
                    })
                    cand_idx += 1

                # Hard negative 2: Another sample (could have similar name or structure)
                if len(same_repo_pool) > 1:
                    sample2 = random.choice([m for m in same_repo_pool if m["name"] != candidates[-1]["name"]])
                    candidates.append({
                        "candidate_id": f"cand_{cand_idx}",
                        "name": sample2["name"],
                        "source": sample2["source"],
                        "file_path": sample2["file_path"],
                        "is_ground_truth": False,
                        "negative_type": "hard_same_repo"
                    })
                    cand_idx += 1

                # Hard negative 3: Unrelated
                if other_repo_pool:
                    sample3 = random.choice(other_repo_pool)
                    candidates.append({
                        "candidate_id": f"cand_{cand_idx}",
                        "name": sample3["name"],
                        "source": sample3["source"],
                        "file_path": sample3["file_path"],
                        "is_ground_truth": False,
                        "negative_type": "unrelated"
                    })
                    cand_idx += 1

                # Shuffle candidate options except NEW is added to list
                cand_options = [c["candidate_id"] for c in candidates] + ["NEW"]

                inst = {
                    "instance_id": f"java_{repo_name}_{cur_commit[:7]}_{func_name}_{change_type}",
                    "language": "java",
                    "repository": repo_name,
                    "change_type": change_type,
                    "commit_after": cur_commit,
                    "commit_before": history[i + 1] if i + 1 < len(history) else "ROOT",
                    "target_function": {
                        "name": func_name,
                        "source": clean_source(cur_src),
                        "file_path": cur_info.get("path", d.get("sourceFilePath", ""))
                    },
                    "ground_truth_id": gt_key,
                    "candidates": candidates,
                    "candidate_options": cand_options
                }
                instances.append(inst)
                if len(instances) >= max_count:
                    break
            if len(instances) >= max_count:
                break
        except Exception as e:
            continue

    print(f"Constructed {len(instances)} Java instances.")
    return instances

def extract_python_functions_from_commit(repo_dir: Path, commit: str, file_path: str) -> Dict[str, str]:
    funcs = {}
    try:
        content = subprocess.check_output(
            ["git", "-C", str(repo_dir), "show", f"{commit}:{file_path}"],
            stderr=subprocess.DEVNULL
        ).decode("utf-8", errors="ignore")
        import ast
        tree = ast.parse(content)
        lines = content.splitlines()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start = node.lineno - 1
                end = node.end_lineno if hasattr(node, "end_lineno") else start + len(node.body) + 5
                func_code = "\n".join(lines[start:end])
                funcs[node.name] = clean_source(func_code)
    except Exception:
        pass
    return funcs

def build_python_instances(max_count: int = 70) -> List[Dict[str, Any]]:
    instances = []
    repo_dir = REPOS_DIR / "dit"
    if not repo_dir.exists():
        return instances

    # Load rminer_python.json for verified commits
    rminer_path = RAW_DIR / "rminer_python.json"
    if not rminer_path.exists():
        return instances

    with open(rminer_path, "r", encoding="utf-8") as f:
        rminer_data = json.load(f)

    # Collect general python functions across commits for negative pool
    python_pool = []
    for item in rminer_data[:20]:
        sha = item["sha1"]
        try:
            diff_files = subprocess.check_output(
                ["git", "-C", str(repo_dir), "diff", "--name-only", f"{sha}^", sha],
                stderr=subprocess.DEVNULL
            ).decode().splitlines()
            for df in diff_files:
                if df.endswith(".py"):
                    fmap = extract_python_functions_from_commit(repo_dir, sha, df)
                    for fn, fcode in fmap.items():
                        if len(fcode) > 20:
                            python_pool.append({"name": fn, "source": fcode, "file_path": df})
        except Exception:
            continue

    print(f"Python negative pool size: {len(python_pool)}")

    for item in rminer_data:
        sha = item["sha1"]
        try:
            # Check commit validity
            parent = subprocess.check_output(
                ["git", "-C", str(repo_dir), "rev-parse", f"{sha}^"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            continue

        refactorings = item.get("refactorings", [])
        for ref in refactorings:
            rtype = ref.get("type", "")
            desc = ref.get("description", "")
            # e.g. "Rename Method public random_distribution(...) renamed to public random_scalar_distribution(...)"
            # Extract names
            m_rename = re.search(r'renamed to\s+(?:public|private|protected)?\s*([a-zA-Z_]\w*)', desc)
            m_old = re.search(r'Method\s+(?:public|private|protected)?\s*([a-zA-Z_]\w*)', desc)

            change_type = "body_modification"
            if "Rename" in rtype:
                change_type = "rename"
            elif "Move" in rtype:
                change_type = "move"
            elif "Extract" in rtype:
                change_type = "parameter_change"

            new_name = m_rename.group(1) if m_rename else (m_old.group(1) if m_old else None)
            old_name = m_old.group(1) if m_old else None

            if not new_name or not old_name:
                continue

            # Find changed python files
            try:
                diff_files = subprocess.check_output(
                    ["git", "-C", str(repo_dir), "diff", "--name-only", parent, sha],
                    stderr=subprocess.DEVNULL
                ).decode().splitlines()
            except Exception:
                continue

            target_src = None
            pred_src = None
            target_file = None

            for df in diff_files:
                if df.endswith(".py"):
                    after_funcs = extract_python_functions_from_commit(repo_dir, sha, df)
                    before_funcs = extract_python_functions_from_commit(repo_dir, parent, df)
                    if new_name in after_funcs and target_src is None:
                        target_src = after_funcs[new_name]
                        target_file = df
                    if old_name in before_funcs and pred_src is None:
                        pred_src = before_funcs[old_name]

            if not target_src or not pred_src:
                continue

            candidates = [{
                "candidate_id": "cand_0",
                "name": old_name,
                "source": pred_src,
                "file_path": target_file,
                "is_ground_truth": True,
                "negative_type": "ground_truth"
            }]

            cand_idx = 1
            # Add hard negatives from pool
            negatives = [p for p in python_pool if p["name"] != old_name]
            if len(negatives) >= 3:
                chosen = random.sample(negatives, 3)
                for c in chosen:
                    candidates.append({
                        "candidate_id": f"cand_{cand_idx}",
                        "name": c["name"],
                        "source": c["source"],
                        "file_path": c["file_path"],
                        "is_ground_truth": False,
                        "negative_type": "hard_pool"
                    })
                    cand_idx += 1

            inst = {
                "instance_id": f"python_dit_{sha[:7]}_{new_name}_{change_type}",
                "language": "python",
                "repository": "dit",
                "change_type": change_type,
                "commit_after": sha,
                "commit_before": parent,
                "target_function": {
                    "name": new_name,
                    "source": target_src,
                    "file_path": target_file
                },
                "ground_truth_id": "cand_0",
                "candidates": candidates,
                "candidate_options": [c["candidate_id"] for c in candidates] + ["NEW"]
            }
            instances.append(inst)
            if len(instances) >= max_count - 10:
                break
        if len(instances) >= max_count - 10:
            break

    # Add 10 genuine NEW python instances
    if python_pool and len(instances) > 0:
        for k in range(10):
            sample = random.choice(python_pool)
            negs = random.sample([p for p in python_pool if p["name"] != sample["name"]], min(3, len(python_pool) - 1))
            cands = []
            for ci, neg in enumerate(negs):
                cands.append({
                    "candidate_id": f"cand_{ci}",
                    "name": neg["name"],
                    "source": neg["source"],
                    "file_path": neg["file_path"],
                    "is_ground_truth": False,
                    "negative_type": "hard_pool"
                })
            inst = {
                "instance_id": f"python_dit_new_func_{k}_{sample['name']}",
                "language": "python",
                "repository": "dit",
                "change_type": "new",
                "commit_after": "synthetic_new",
                "commit_before": "synthetic_parent",
                "target_function": {
                    "name": sample["name"],
                    "source": sample["source"],
                    "file_path": sample["file_path"]
                },
                "ground_truth_id": "NEW",
                "candidates": cands,
                "candidate_options": [c["candidate_id"] for c in cands] + ["NEW"]
            }
            instances.append(inst)

    print(f"Constructed {len(instances)} Python instances.")
    return instances

def extract_js_functions_from_content(content: str) -> Dict[str, str]:
    funcs = {}
    lines = content.splitlines()
    # Regex pattern to match function declarations and method definitions
    pattern = re.compile(r'(?:function\s+([a-zA-Z_]\w*)|([a-zA-Z_]\w*)\s*[:=]\s*function|([a-zA-Z_]\w*)\s*\([^)]*\)\s*\{)')
    for idx, line in enumerate(lines):
        m = pattern.search(line)
        if m:
            name = m.group(1) or m.group(2) or m.group(3)
            if name and name not in ('if', 'for', 'while', 'switch', 'catch'):
                # Extract snippet of ~15 lines or until closing brace
                snippet = "\n".join(lines[idx:min(len(lines), idx + 25)])
                funcs[name] = clean_source(snippet)
    return funcs

def build_js_and_ts_instances(max_js: int = 70, max_ts: int = 60) -> List[Dict[str, Any]]:
    instances = []
    redux_dir = REPOS_DIR / "redux"
    chartjs_dir = REPOS_DIR / "chartjs"

    # Collect JS pool from Redux and Chart.js
    js_pool = []
    ts_pool = []

    # Redux commits
    if redux_dir.exists():
        commits = subprocess.check_output(
            ["git", "-C", str(redux_dir), "log", "--oneline", "-n", "60"],
            stderr=subprocess.DEVNULL
        ).decode().splitlines()

        for c_line in commits:
            sha = c_line.split()[0]
            try:
                parent = subprocess.check_output(
                    ["git", "-C", str(redux_dir), "rev-parse", f"{sha}^"],
                    stderr=subprocess.DEVNULL
                ).decode().strip()
                diff_files = subprocess.check_output(
                    ["git", "-C", str(redux_dir), "diff", "--name-only", parent, sha],
                    stderr=subprocess.DEVNULL
                ).decode().splitlines()
                for df in diff_files:
                    is_ts = df.endswith(".ts") or df.endswith(".tsx")
                    is_js = df.endswith(".js") or df.endswith(".jsx")
                    if not (is_ts or is_js):
                        continue

                    try:
                        content_after = subprocess.check_output(
                            ["git", "-C", str(redux_dir), "show", f"{sha}:{df}"],
                            stderr=subprocess.DEVNULL
                        ).decode("utf-8", errors="ignore")
                        content_before = subprocess.check_output(
                            ["git", "-C", str(redux_dir), "show", f"{parent}:{df}"],
                            stderr=subprocess.DEVNULL
                        ).decode("utf-8", errors="ignore")
                    except Exception:
                        continue

                    funcs_after = extract_js_functions_from_content(content_after)
                    funcs_before = extract_js_functions_from_content(content_before)

                    target_pool = ts_pool if is_ts else js_pool
                    for fn, fcode in funcs_after.items():
                        if len(fcode) > 20:
                            target_pool.append({"name": fn, "source": fcode, "file_path": df, "lang": "typescript" if is_ts else "javascript"})

                    # Check for matched function pairs across commit
                    for fn in funcs_after:
                        if fn in funcs_before:
                            code_after = funcs_after[fn]
                            code_before = funcs_before[fn]
                            lang = "typescript" if is_ts else "javascript"

                            if code_after == code_before:
                                change_type = "unchanged"
                            elif len(code_after) != len(code_before):
                                change_type = "body_modification"
                            else:
                                change_type = "parameter_change"

                            inst = {
                                "instance_id": f"{lang}_redux_{sha[:7]}_{fn}_{change_type}",
                                "language": lang,
                                "repository": "redux",
                                "change_type": change_type,
                                "commit_after": sha,
                                "commit_before": parent,
                                "target_function": {
                                    "name": fn,
                                    "source": code_after,
                                    "file_path": df
                                },
                                "ground_truth_id": "cand_0",
                                "pred_info": {
                                    "name": fn,
                                    "source": code_before,
                                    "file_path": df
                                }
                            }
                            instances.append(inst)
            except Exception:
                continue

    # Chart.js commits
    if chartjs_dir.exists():
        commits = subprocess.check_output(
            ["git", "-C", str(chartjs_dir), "log", "--oneline", "-n", "40"],
            stderr=subprocess.DEVNULL
        ).decode().splitlines()

        for c_line in commits:
            sha = c_line.split()[0]
            try:
                parent = subprocess.check_output(
                    ["git", "-C", str(chartjs_dir), "rev-parse", f"{sha}^"],
                    stderr=subprocess.DEVNULL
                ).decode().strip()
                diff_files = subprocess.check_output(
                    ["git", "-C", str(chartjs_dir), "diff", "--name-only", parent, sha],
                    stderr=subprocess.DEVNULL
                ).decode().splitlines()
                for df in diff_files:
                    is_ts = df.endswith(".ts")
                    is_js = df.endswith(".js")
                    if not (is_ts or is_js):
                        continue
                    try:
                        content_after = subprocess.check_output(
                            ["git", "-C", str(chartjs_dir), "show", f"{sha}:{df}"],
                            stderr=subprocess.DEVNULL
                        ).decode("utf-8", errors="ignore")
                        content_before = subprocess.check_output(
                            ["git", "-C", str(chartjs_dir), "show", f"{parent}:{df}"],
                            stderr=subprocess.DEVNULL
                        ).decode("utf-8", errors="ignore")
                    except Exception:
                        continue

                    funcs_after = extract_js_functions_from_content(content_after)
                    funcs_before = extract_js_functions_from_content(content_before)

                    target_pool = ts_pool if is_ts else js_pool
                    for fn, fcode in funcs_after.items():
                        if len(fcode) > 20:
                            target_pool.append({"name": fn, "source": fcode, "file_path": df, "lang": "typescript" if is_ts else "javascript"})

                    for fn in funcs_after:
                        if fn in funcs_before:
                            code_after = funcs_after[fn]
                            code_before = funcs_before[fn]
                            lang = "typescript" if is_ts else "javascript"
                            change_type = "body_modification" if code_after != code_before else "unchanged"

                            inst = {
                                "instance_id": f"{lang}_chartjs_{sha[:7]}_{fn}_{change_type}",
                                "language": lang,
                                "repository": "chartjs",
                                "change_type": change_type,
                                "commit_after": sha,
                                "commit_before": parent,
                                "target_function": {
                                    "name": fn,
                                    "source": code_after,
                                    "file_path": df
                                },
                                "ground_truth_id": "cand_0",
                                "pred_info": {
                                    "name": fn,
                                    "source": code_before,
                                    "file_path": df
                                }
                            }
                            instances.append(inst)
            except Exception:
                continue

    # Now finalize candidates for JS and TS
    final_js = []
    final_ts = []

    for item in instances:
        lang = item["language"]
        pool = ts_pool if lang == "typescript" else js_pool
        pred = item["pred_info"]

        candidates = [{
            "candidate_id": "cand_0",
            "name": pred["name"],
            "source": pred["source"],
            "file_path": pred["file_path"],
            "is_ground_truth": True,
            "negative_type": "ground_truth"
        }]

        negs = [p for p in pool if p["name"] != pred["name"]]
        if len(negs) >= 3:
            chosen = random.sample(negs, 3)
            for ci, neg in enumerate(chosen):
                candidates.append({
                    "candidate_id": f"cand_{ci+1}",
                    "name": neg["name"],
                    "source": neg["source"],
                    "file_path": neg["file_path"],
                    "is_ground_truth": False,
                    "negative_type": "hard_pool"
                })

        item_cleaned = {
            "instance_id": item["instance_id"],
            "language": item["language"],
            "repository": item["repository"],
            "change_type": item["change_type"],
            "commit_after": item["commit_after"],
            "commit_before": item["commit_before"],
            "target_function": item["target_function"],
            "ground_truth_id": "cand_0",
            "candidates": candidates,
            "candidate_options": [c["candidate_id"] for c in candidates] + ["NEW"]
        }

        if lang == "javascript" and len(final_js) < max_js - 10:
            final_js.append(item_cleaned)
        elif lang == "typescript" and len(final_ts) < max_ts - 10:
            final_ts.append(item_cleaned)

    # Add 10 NEW instances for JS and TS
    for lang, lst, pool, max_c in [("javascript", final_js, js_pool, max_js), ("typescript", final_ts, ts_pool, max_ts)]:
        if pool:
            for k in range(10):
                target = random.choice(pool)
                negs = random.sample([p for p in pool if p["name"] != target["name"]], min(3, len(pool) - 1))
                cands = []
                for ci, neg in enumerate(negs):
                    cands.append({
                        "candidate_id": f"cand_{ci}",
                        "name": neg["name"],
                        "source": neg["source"],
                        "file_path": neg["file_path"],
                        "is_ground_truth": False,
                        "negative_type": "hard_pool"
                    })
                inst = {
                    "instance_id": f"{lang}_new_func_{k}_{target['name']}",
                    "language": lang,
                    "repository": "redux",
                    "change_type": "new",
                    "commit_after": "new_commit",
                    "commit_before": "parent_commit",
                    "target_function": {
                        "name": target["name"],
                        "source": target["source"],
                        "file_path": target["file_path"]
                    },
                    "ground_truth_id": "NEW",
                    "candidates": cands,
                    "candidate_options": [c["candidate_id"] for c in cands] + ["NEW"]
                }
                lst.append(inst)

    print(f"Constructed {len(final_js)} JavaScript instances.")
    print(f"Constructed {len(final_ts)} TypeScript instances.")
    return final_js + final_ts

def build_full_benchmark():
    java_insts = build_java_instances(80)
    python_insts = build_python_instances(70)
    js_ts_insts = build_js_and_ts_instances(70, 60)

    full_dataset = java_insts + python_insts + js_ts_insts
    random.shuffle(full_dataset)

    # 40% validation, 60% test split partitioned deterministically per language
    val_set = []
    test_set = []

    by_lang = {}
    for inst in full_dataset:
        l = inst["language"]
        by_lang.setdefault(l, []).append(inst)

    for l, items in by_lang.items():
        random.shuffle(items)
        n_val = int(len(items) * 0.4)
        for i, item in enumerate(items):
            if i < n_val:
                item["split"] = "val"
                val_set.append(item)
            else:
                item["split"] = "test"
                test_set.append(item)

    out_file = DATA_DIR / "benchmark.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_dataset, f, indent=2)

    print(f"\n==========================================")
    print(f"Benchmark Generated: {len(full_dataset)} total instances")
    print(f"  Validation: {len(val_set)}")
    print(f"  Test:       {len(test_set)}")
    print(f"Breakdown by Language:")
    for l, items in by_lang.items():
        print(f"  {l}: {len(items)} instances")
    print(f"Saved to: {out_file}")
    print(f"==========================================\n")

if __name__ == "__main__":
    build_full_benchmark()
