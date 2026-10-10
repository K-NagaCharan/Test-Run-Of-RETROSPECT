"""
Master Evaluation Runner for Retrospect Laya Empirical Study.
Executes all baselines, Laya embedding models, Laya decision heads, and hybrid combinations.
Performs validation tuning, held-out testing, latency/memory profiling, and saves all outputs.
"""
import os
import sys
import time
import json
import psutil
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np

from src.models.baselines import LexicalMatcher, StructuralMatcher
from src.models.laya_embedder import LayaEmbeddingMatcher
from src.models.laya_decision import LayaDecisionMatcher
from src.hybrid.hybrid_matchers import (
    HybridRetrievalReranker,
    HybridWeightedCombiner,
    UncertaintyGatedHybrid
)
from src.metrics.evaluator import compute_metrics, slice_metrics

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
PRED_DIR = RESULTS_DIR / "predictions"
TABLES_DIR = RESULTS_DIR / "tables"
PRED_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "benchmark.json"

def get_process_memory_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def run_evaluation():
    start_total_time = time.time()
    print("=" * 60)
    print("STARTING RETROSPECT LAYA EMPIRICAL EVALUATION")
    print("=" * 60)

    # 1. Load benchmark dataset
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    val_data = [item for item in benchmark if item.get("split") == "val"]
    test_data = [item for item in benchmark if item.get("split") == "test"]

    print(f"Total benchmark items: {len(benchmark)}")
    print(f"  Validation split:    {len(val_data)}")
    print(f"  Held-out Test split: {len(test_data)}")

    # 2. Candidate Retrieval Analysis
    # Evaluate candidate set recall (is ground truth in candidate set when gt != NEW?)
    gt_match_items = [item for item in test_data if item["ground_truth_id"] != "NEW"]
    cov_count = sum(1 for item in gt_match_items if any(c["is_ground_truth"] for c in item["candidates"]))
    retrieval_coverage = cov_count / len(gt_match_items) if gt_match_items else 1.0
    print(f"\nCandidate Retrieval Coverage on Test Set: {retrieval_coverage * 100:.2f}% ({cov_count}/{len(gt_match_items)})")

    # 3. Model Initialization
    print("\n--- Initializing Models ---")
    lex_matcher = LexicalMatcher(new_threshold=0.35)
    struct_matcher = StructuralMatcher(new_threshold=0.30)

    laya_embed_eng = LayaEmbeddingMatcher(checkpoint_type="english", device="cpu", new_threshold=0.75)
    laya_embed_multi = LayaEmbeddingMatcher(checkpoint_type="multilingual", device="cpu", new_threshold=0.75)

    laya_router_eng = LayaDecisionMatcher(checkpoint_type="english", device="cpu")
    laya_router_multi = LayaDecisionMatcher(checkpoint_type="multilingual", device="cpu")

    # 4. Validation Tuning Phase
    print("\n--- Phase 1: Parameter Tuning on Validation Set ---")

    # Tune Lexical new_threshold
    best_lex_thresh = 0.35
    best_lex_acc = 0.0
    for th in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]:
        m = LexicalMatcher(new_threshold=th)
        preds = [m.predict(item["target_function"], item["candidates"])["prediction"] for item in val_data]
        acc = sum(1 for p, item in zip(preds, val_data) if p == item["ground_truth_id"]) / len(val_data)
        if acc > best_lex_acc:
            best_lex_acc = acc
            best_lex_thresh = th
    print(f"Tuned Lexical new_threshold: {best_lex_thresh} (Val Acc: {best_lex_acc:.4f})")
    lex_matcher.new_threshold = best_lex_thresh

    # Tune Structural new_threshold
    best_struct_thresh = 0.30
    best_struct_acc = 0.0
    for th in [0.20, 0.25, 0.30, 0.35, 0.40]:
        m = StructuralMatcher(new_threshold=th)
        preds = [m.predict(item["target_function"], item["candidates"], item["language"])["prediction"] for item in val_data]
        acc = sum(1 for p, item in zip(preds, val_data) if p == item["ground_truth_id"]) / len(val_data)
        if acc > best_struct_acc:
            best_struct_acc = acc
            best_struct_thresh = th
    print(f"Tuned Structural new_threshold: {best_struct_thresh} (Val Acc: {best_struct_acc:.4f})")
    struct_matcher.new_threshold = best_struct_thresh

    # Tune Laya Embedding new_threshold
    best_emb_thresh = 0.75
    best_emb_acc = 0.0
    for th in [0.65, 0.70, 0.75, 0.80, 0.85]:
        laya_embed_eng.new_threshold = th
        preds = [laya_embed_eng.predict(item["target_function"], item["candidates"])["prediction"] for item in val_data]
        acc = sum(1 for p, item in zip(preds, val_data) if p == item["ground_truth_id"]) / len(val_data)
        if acc > best_emb_acc:
            best_emb_acc = acc
            best_emb_thresh = th
    print(f"Tuned Laya Embedding new_threshold: {best_emb_thresh} (Val Acc: {best_emb_acc:.4f})")
    laya_embed_eng.new_threshold = best_emb_thresh
    laya_embed_multi.new_threshold = best_emb_thresh

    # Tune Hybrid Combiners on Validation Set
    hybrid_reranker = HybridRetrievalReranker(struct_matcher, laya_embed_eng, top_k=2)

    # Hybrid E2: Weighted Blend
    best_weights = (0.50, 0.30, 0.20)
    best_blend_acc = 0.0
    best_blend_th = 0.35
    for ws in [0.4, 0.5, 0.6]:
        for wl in [0.2, 0.3, 0.4]:
            we = round(1.0 - ws - wl, 2)
            if we < 0: continue
            for th in [0.30, 0.35, 0.40]:
                hb = HybridWeightedCombiner(struct_matcher, lex_matcher, laya_embed_eng,
                                            w_struct=ws, w_lex=wl, w_laya=we, new_threshold=th)
                preds = [hb.predict(item["target_function"], item["candidates"], item["language"])["prediction"] for item in val_data]
                acc = sum(1 for p, item in zip(preds, val_data) if p == item["ground_truth_id"]) / len(val_data)
                if acc > best_blend_acc:
                    best_blend_acc = acc
                    best_weights = (ws, wl, we)
                    best_blend_th = th
    print(f"Tuned Hybrid Blend weights: struct={best_weights[0]}, lex={best_weights[1]}, laya={best_weights[2]}, th={best_blend_th} (Val Acc: {best_blend_acc:.4f})")
    hybrid_blend = HybridWeightedCombiner(struct_matcher, lex_matcher, laya_embed_eng,
                                          w_struct=best_weights[0], w_lex=best_weights[1], w_laya=best_weights[2],
                                          new_threshold=best_blend_th)

    # Hybrid E3: Uncertainty-Gated Fallback
    best_gated = UncertaintyGatedHybrid(struct_matcher, laya_router_eng, uncertainty_margin=0.15, high_conf_threshold=0.70, low_conf_threshold=0.30)

    # 5. Full Evaluation Phase on Held-Out Test Set
    print("\n--- Phase 2: Execution on Held-Out Test Set (168 instances) ---")

    models_to_evaluate = {
        "Baseline_A_Lexical": (lex_matcher, "predict_lex"),
        "Baseline_B_Structural": (struct_matcher, "predict_struct"),
        "Method_C_Laya_Embed_English": (laya_embed_eng, "predict_embed"),
        "Method_C_Laya_Embed_Multi": (laya_embed_multi, "predict_embed"),
        "Method_D_Laya_Router_English": (laya_router_eng, "predict_router"),
        "Method_D_Laya_Router_Multi": (laya_router_multi, "predict_router"),
        "Method_E1_Hybrid_Rerank": (hybrid_reranker, "predict_hybrid_rerank"),
        "Method_E2_Hybrid_Blend": (hybrid_blend, "predict_hybrid_blend"),
        "Method_E3_Hybrid_Gated": (best_gated, "predict_hybrid_gated")
    }

    aggregate_results = []
    efficiency_results = []
    per_language_results = {}
    per_change_type_results = {}
    all_raw_predictions = {}
    error_cases = []

    for model_name, (model_obj, call_type) in models_to_evaluate.items():
        print(f"\nEvaluating {model_name}...")
        mem_before = get_process_memory_mb()
        t0 = time.time()
        latencies = []
        preds_list = []

        for item in test_data:
            item_t0 = time.time()
            tf = item["target_function"]
            cands = item["candidates"]
            lang = item["language"]
            gt = item["ground_truth_id"]

            if call_type == "predict_lex":
                out = model_obj.predict(tf, cands)
            elif call_type == "predict_struct":
                out = model_obj.predict(tf, cands, lang)
            elif call_type == "predict_embed":
                out = model_obj.predict(tf, cands)
            elif call_type == "predict_router":
                out = model_obj.predict(tf, cands, lang)
            elif call_type in ("predict_hybrid_rerank", "predict_hybrid_blend", "predict_hybrid_gated"):
                out = model_obj.predict(tf, cands, lang)

            item_lat_ms = (time.time() - item_t0) * 1000.0
            latencies.append(item_lat_ms)

            pred_choice = out.get("prediction", "NEW")
            conf = out.get("confidence", 0.0)
            scores = out.get("scores", {})

            pred_record = {
                "instance_id": item["instance_id"],
                "language": lang,
                "change_type": item["change_type"],
                "repository": item["repository"],
                "ground_truth_id": gt,
                "prediction": pred_choice,
                "is_correct": (pred_choice == gt),
                "confidence": conf,
                "scores": scores,
                "latency_ms": item_lat_ms
            }
            preds_list.append(pred_record)

            # Record errors
            if pred_choice != gt:
                err_type = "False_NEW" if pred_choice == "NEW" and gt != "NEW" else ("Missed_NEW" if gt == "NEW" else "Mismatched_Predecessor")
                if len(error_cases) < 40 and model_name in ("Baseline_B_Structural", "Method_C_Laya_Embed_English", "Method_D_Laya_Router_English", "Method_E3_Hybrid_Gated"):
                    error_cases.append({
                        "model": model_name,
                        "instance_id": item["instance_id"],
                        "language": lang,
                        "change_type": item["change_type"],
                        "error_type": err_type,
                        "ground_truth": gt,
                        "prediction": pred_choice,
                        "confidence": conf,
                        "scores": scores,
                        "target_name": tf.get("name", ""),
                        "target_snippet": tf.get("source", "")[:200]
                    })

        total_eval_time = time.time() - t0
        mem_after = get_process_memory_mb()
        peak_mem_mb = max(mem_before, mem_after)

        # Compute metrics
        m = compute_metrics(preds_list)
        m["model"] = model_name
        aggregate_results.append(m)

        eff = {
            "model": model_name,
            "total_time_sec": round(total_eval_time, 2),
            "throughput_items_per_sec": round(len(test_data) / total_eval_time, 2),
            "latency_mean_ms": round(float(np.mean(latencies)), 2),
            "latency_median_ms": round(float(np.median(latencies)), 2),
            "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2),
            "memory_usage_mb": round(peak_mem_mb, 2)
        }
        efficiency_results.append(eff)

        # Slice by language and change_type
        by_lang = slice_metrics(preds_list, "language")
        for l, metrics in by_lang.items():
            per_language_results.setdefault(l, []).append({"model": model_name, **metrics})

        by_chg = slice_metrics(preds_list, "change_type")
        for ct, metrics in by_chg.items():
            per_change_type_results.setdefault(ct, []).append({"model": model_name, **metrics})

        # Save raw predictions
        pred_file = PRED_DIR / f"{model_name}.jsonl"
        with open(pred_file, "w", encoding="utf-8") as f:
            for pr in preds_list:
                f.write(json.dumps(pr) + "\n")

        print(f"  -> Top-1 Accuracy: {m['top1_accuracy'] * 100:.2f}% | MRR: {m['mrr']:.4f} | Macro-F1: {m['macro_f1']:.4f} | Mean Latency: {eff['latency_mean_ms']:.1f}ms")

    # 6. Save aggregate and sliced tables
    df_agg = pd.DataFrame(aggregate_results)
    df_agg.to_csv(TABLES_DIR / "aggregate_metrics.csv", index=False)
    with open(TABLES_DIR / "aggregate_metrics.json", "w", encoding="utf-8") as f:
        json.dump(aggregate_results, f, indent=2)

    df_eff = pd.DataFrame(efficiency_results)
    df_eff.to_csv(TABLES_DIR / "efficiency_metrics.csv", index=False)
    with open(TABLES_DIR / "efficiency_metrics.json", "w", encoding="utf-8") as f:
        json.dump(efficiency_results, f, indent=2)

    # Per language tables
    flattened_lang = []
    for l, lst in per_language_results.items():
        for row in lst:
            flattened_lang.append({"language": l, **row})
    df_lang = pd.DataFrame(flattened_lang)
    df_lang.to_csv(TABLES_DIR / "per_language_metrics.csv", index=False)
    with open(TABLES_DIR / "per_language_metrics.json", "w", encoding="utf-8") as f:
        json.dump(per_language_results, f, indent=2)

    # Per change type tables
    flattened_chg = []
    for ct, lst in per_change_type_results.items():
        for row in lst:
            flattened_chg.append({"change_type": ct, **row})
    df_chg = pd.DataFrame(flattened_chg)
    df_chg.to_csv(TABLES_DIR / "per_change_type_metrics.csv", index=False)
    with open(TABLES_DIR / "per_change_type_metrics.json", "w", encoding="utf-8") as f:
        json.dump(per_change_type_results, f, indent=2)

    # Error cases
    with open(TABLES_DIR / "error_cases.json", "w", encoding="utf-8") as f:
        json.dump(error_cases, f, indent=2)

    total_duration = time.time() - start_total_time
    print(f"\n==========================================")
    print(f"EVALUATION COMPLETE in {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    print(f"Results saved to: {TABLES_DIR}")
    print(f"Predictions saved to: {PRED_DIR}")
    print(f"==========================================\n")

if __name__ == "__main__":
    run_evaluation()
