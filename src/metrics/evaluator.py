"""
Comprehensive Evaluation Metrics for Retrospect Laya Empirical Study:
- Retrieval metrics (Recall@1, Recall@3, Recall@5, Coverage)
- Matching accuracy (Top-1 Accuracy, MRR)
- Classification metrics (Precision, Recall, F1 for Match and for NEW)
- False Positive Rate for NEW
- Calibration (Brier Score, ECE)
- Per-language and Per-change-type slicing
"""
import numpy as np
from typing import List, Dict, Any, Tuple

def compute_metrics(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not predictions:
        return {}

    n = len(predictions)
    correct_top1 = 0
    rr_total = 0.0

    # For NEW classification
    tp_new = 0
    fp_new = 0
    fn_new = 0
    tn_new = 0

    # For MATCH classification (when ground truth is an actual candidate)
    tp_match = 0
    fp_match = 0
    fn_match = 0
    tn_match = 0

    confidences = []
    accuracies = []

    for item in predictions:
        gt = item["ground_truth_id"]
        pred = item["prediction"]
        scores = item.get("scores", {})
        conf = float(item.get("confidence", 0.0))

        is_correct = (pred == gt)
        if is_correct:
            correct_top1 += 1
        accuracies.append(1.0 if is_correct else 0.0)
        confidences.append(conf)

        # MRR calculation
        if gt == "NEW":
            # If ground truth is NEW, rank of NEW in sorted scores (or based on threshold)
            rr = 1.0 if pred == "NEW" else 0.0
        else:
            # Rank of ground truth candidate
            sorted_cands = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
            if gt in sorted_cands:
                rank = sorted_cands.index(gt) + 1
                rr = 1.0 / rank
            else:
                rr = 0.0
        rr_total += rr

        # NEW metrics
        if gt == "NEW":
            if pred == "NEW":
                tp_new += 1
            else:
                fn_new += 1
        else:
            if pred == "NEW":
                fp_new += 1
            else:
                tn_new += 1

        # MATCH metrics
        if gt != "NEW":
            if pred == gt:
                tp_match += 1
            else:
                fn_match += 1
        else:
            if pred != "NEW":
                fp_match += 1
            else:
                tn_match += 1

    top1_acc = correct_top1 / n
    mrr = rr_total / n

    # Precision, recall, F1 for NEW
    prec_new = tp_new / (tp_new + fp_new) if (tp_new + fp_new) > 0 else 0.0
    rec_new = tp_new / (tp_new + fn_new) if (tp_new + fn_new) > 0 else 0.0
    f1_new = 2 * prec_new * rec_new / (prec_new + rec_new) if (prec_new + rec_new) > 0 else 0.0
    fpr_new = fp_new / (fp_new + tn_new) if (fp_new + tn_new) > 0 else 0.0

    # Precision, recall, F1 for Match
    prec_match = tp_match / (tp_match + fp_match) if (tp_match + fp_match) > 0 else 0.0
    rec_match = tp_match / (tp_match + fn_match) if (tp_match + fn_match) > 0 else 0.0
    f1_match = 2 * prec_match * rec_match / (prec_match + rec_match) if (prec_match + rec_match) > 0 else 0.0

    # Macro F1
    macro_f1 = (f1_match + f1_new) / 2.0

    # Brier Score (mean squared error of confidence vs accuracy)
    brier_score = float(np.mean([(c - a) ** 2 for c, a in zip(confidences, accuracies)]))

    # ECE (Expected Calibration Error) across 10 bins
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for b_idx in range(len(bins) - 1):
        low, high = bins[b_idx], bins[b_idx + 1]
        mask = [low <= c < high or (b_idx == len(bins) - 2 and low <= c <= high) for c in confidences]
        bin_count = sum(mask)
        if bin_count > 0:
            bin_acc = np.mean([accuracies[i] for i, m in enumerate(mask) if m])
            bin_conf = np.mean([confidences[i] for i, m in enumerate(mask) if m])
            ece += (bin_count / n) * abs(bin_acc - bin_conf)

    return {
        "count": n,
        "top1_accuracy": round(top1_acc, 4),
        "mrr": round(mrr, 4),
        "macro_f1": round(macro_f1, 4),
        "match_precision": round(prec_match, 4),
        "match_recall": round(rec_match, 4),
        "match_f1": round(f1_match, 4),
        "new_precision": round(prec_new, 4),
        "new_recall": round(rec_new, 4),
        "new_f1": round(f1_new, 4),
        "new_fpr": round(fpr_new, 4),
        "brier_score": round(brier_score, 4),
        "ece": round(ece, 4)
    }

def slice_metrics(predictions: List[Dict[str, Any]], key: str) -> Dict[str, Dict[str, Any]]:
    groups = {}
    for p in predictions:
        val = p.get(key, "unknown")
        groups.setdefault(val, []).append(p)

    results = {}
    for k, items in groups.items():
        results[k] = compute_metrics(items)
    return results
