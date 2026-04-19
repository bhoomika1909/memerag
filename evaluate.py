"""
evaluate.py
-----------
Owner: Yazi
Project: MemeRAG (CS 6120 NLP)

What this file does:
    1. Loads dev.jsonl (500 human-labeled entries — never in ChromaDB)
    2. Runs each entry through the full pipeline (embed → retrieve → Llama 3)
    3. Compares predicted labels to ground truth
    4. Reports F1 (macro), precision, recall, accuracy, confusion matrix
    5. Generates detailed evaluation report with citations

Input:  data/dev.jsonl
Output: Evaluation metrics, confusion matrix, and per-class performance

How to run:
    python evaluate.py [--sample SIZE] [--verbose] [--timeout SECONDS]

Examples:
    python evaluate.py              # Full 500-sample evaluation
    python evaluate.py --sample 50  # Quick test with 50 samples
    python evaluate.py --timeout 120  # Set per-query timeout to 2 minutes
    python evaluate.py --verbose    # Show detailed output for each query

Note: Full eval (~3–8 hours depending on GCP latency). Use --sample for testing.
"""

import json
import os
import sys
import time
import argparse
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
from pipeline import analyze_meme

# ── constants ──────────────────────────────────────────────────────────────
EVAL_PATH = "data/dev.jsonl"
DEFAULT_TIMEOUT = 120  # seconds per query
DEFAULT_SAMPLE_SIZE = None  # None = full eval (500)


# ── step 1: load eval data ─────────────────────────────────────────────────
def load_eval_data(file_path: str, max_samples: Optional[int] = None) -> pd.DataFrame:
    """
    Loads dev.jsonl — the held-out evaluation set.
    This file is NEVER ingested into ChromaDB.
    
    Args:
        file_path: Path to dev.jsonl
        max_samples: If set, sample this many random entries (for testing)
    
    Returns:
        DataFrame with columns: id, text, label (0/1)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Evaluation file not found: {file_path}")

    print(f"\n{'='*70}")
    print(f"  Loading evaluation data from {file_path} ...")
    print(f"{'='*70}")

    rows = []
    with open(file_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"  ⚠️  Skipped malformed line {line_num}: {e}")
                    continue

    df = pd.DataFrame(rows)
    
    # Ensure required columns exist
    required_cols = ["id", "text", "label"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dev.jsonl: {missing}")
    
    # Clean data
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)
    df = df[required_cols]
    
    # Sample if requested
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
        print(f"  📊 Using random sample of {max_samples} entries (seed=42)")
    
    print(f"  ✅ Loaded {len(df)} total evaluation entries")
    print(f"     • Hateful (1) : {df['label'].sum()} ({100*df['label'].sum()/len(df):.1f}%)")
    print(f"     • Safe    (0) : {(df['label'] == 0).sum()} ({100*(df['label']==0).sum()/len(df):.1f}%)")
    print(f"{'='*70}\n")
    
    return df


# ── step 2: run pipeline on eval set ───────────────────────────────────────
def run_evaluation(
    df: pd.DataFrame,
    query_timeout: int = DEFAULT_TIMEOUT,
    verbose: bool = False
) -> Dict[str, List]:
    """
    Runs every eval entry through the full RAG pipeline.
    
    Args:
        df: DataFrame with meme texts and ground truth labels
        query_timeout: Max seconds to wait per query (prevents hangs)
        verbose: Print detailed output for each query
    
    Returns:
        Dict with true_labels, pred_labels, latencies, errors, meme_ids
    """
    print(f"\n{'='*70}")
    print(f"  Running evaluation on {len(df)} entries ...")
    print(f"  Query timeout: {query_timeout}s per entry")
    print(f"{'='*70}\n")

    true_labels = []
    pred_labels = []
    latencies = []
    errors = []
    meme_ids = []
    
    start_time = time.time()
    error_count = 0

    for i, (idx, row) in enumerate(df.iterrows()):
        meme_id = row["id"]
        meme_text = row["text"]
        true_label = int(row["label"])
        
        # Progress bar
        progress_pct = int(100 * (i + 1) / len(df))
        progress_bar = "█" * (progress_pct // 5) + "░" * (20 - progress_pct // 5)
        elapsed = time.time() - start_time
        eta_sec = (elapsed / (i + 1)) * (len(df) - i - 1)
        
        sys.stdout.write(
            f"\r  [{progress_bar}] {i+1}/{len(df)} | "
            f"{progress_pct:3d}% | Errors: {error_count} | "
            f"ETA: {int(eta_sec)}s"
        )
        sys.stdout.flush()

        query_start = time.time()
        
        try:
            # Call pipeline with timeout enforcement
            result = analyze_meme(meme_text)
            query_end = time.time()
            latency = query_end - query_start
            
            # Parse predicted label
            raw_label = str(result.get("hate_label", "uncertain")).lower()
            is_hateful = ("hate" in raw_label) and ("not" not in raw_label)
            pred_label = 1 if is_hateful else 0
            
            true_labels.append(true_label)
            pred_labels.append(pred_label)
            latencies.append(round(latency, 2))
            meme_ids.append(meme_id)
            errors.append(None)
            
            if verbose and (i + 1) % 10 == 0:
                conf = result.get("confidence", 0.0)
                print(f"\n    Sample {i+1}: True={true_label}, Pred={pred_label}, "
                      f"Confidence={conf:.2f}, Latency={latency:.2f}s")
        
        except Exception as e:
            error_count += 1
            latency = time.time() - query_start
            
            true_labels.append(true_label)
            pred_labels.append(0)  # Default to safe on error
            latencies.append(round(latency, 2))
            meme_ids.append(meme_id)
            errors.append(str(e))

    print(f"\n{'='*70}")
    print(f"  ✅ Evaluation complete!")
    print(f"  Total time: {time.time() - start_time:.1f}s")
    print(f"  Errors: {error_count}/{len(df)} ({100*error_count/len(df):.1f}%)")
    print(f"{'='*70}\n")

    return {
        "true": true_labels,
        "pred": pred_labels,
        "latencies": latencies,
        "errors": errors,
        "ids": meme_ids,
    }


# ── step 3: compute metrics ────────────────────────────────────────────────
def compute_metrics(true_labels: List[int], pred_labels: List[int]) -> Dict:
    """
    Computes comprehensive evaluation metrics.
    
    Metrics:
        - Per-class: precision, recall, F1
        - Macro averages: F1, precision, recall (for imbalanced classes)
        - Overall: accuracy, confusion matrix
    
    Args:
        true_labels: Ground truth labels (0/1)
        pred_labels: Model predictions (0/1)
    
    Returns:
        Dict with all computed metrics
    """
    assert len(true_labels) == len(pred_labels), "Label lists must have same length"
    
    # Confusion matrix elements
    tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(true_labels, pred_labels) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(true_labels, pred_labels) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == 1 and p == 0)

    total = len(true_labels)

    # Per-class metrics (hateful = 1)
    precision_hateful = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_hateful = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_hateful = (
        2 * precision_hateful * recall_hateful / (precision_hateful + recall_hateful)
        if (precision_hateful + recall_hateful) > 0
        else 0.0
    )

    # Per-class metrics (safe = 0)
    precision_safe = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    recall_safe = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_safe = (
        2 * precision_safe * recall_safe / (precision_safe + recall_safe)
        if (precision_safe + recall_safe) > 0
        else 0.0
    )

    # Macro averages (equal weight to both classes)
    macro_f1 = (f1_hateful + f1_safe) / 2
    macro_precision = (precision_hateful + precision_safe) / 2
    macro_recall = (recall_hateful + recall_safe) / 2
    
    # Micro averages (overall)
    accuracy = (tp + tn) / total if total > 0 else 0.0

    return {
        # Overall metrics
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        
        # Per-class metrics
        "f1_hateful": round(f1_hateful, 4),
        "precision_hateful": round(precision_hateful, 4),
        "recall_hateful": round(recall_hateful, 4),
        
        "f1_safe": round(f1_safe, 4),
        "precision_safe": round(precision_safe, 4),
        "recall_safe": round(recall_safe, 4),
        
        # Confusion matrix
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "total": total,
    }


# ── step 4: print results ──────────────────────────────────────────────────
def print_results(
    metrics: Dict,
    latencies: List[float],
    label: str = "WITH RAG",
    show_confusion: bool = True
) -> None:
    """
    Prints evaluation results in a formatted table.
    
    Args:
        metrics: Dict from compute_metrics()
        latencies: List of query latencies
        label: Label for this evaluation run
        show_confusion: Whether to print confusion matrix
    """
    print(f"\n{'='*70}")
    print(f"  EVALUATION RESULTS — {label}")
    print(f"{'='*70}")
    
    # Primary metrics (macro)
    print(f"\n  📊 MACRO-AVERAGED METRICS (primary for imbalanced classes)")
    print(f"  {'-'*70}")
    print(f"    Macro F1 Score    : {metrics['macro_f1']:.4f}  ← PRIMARY METRIC")
    print(f"    Macro Precision   : {metrics['macro_precision']:.4f}")
    print(f"    Macro Recall      : {metrics['macro_recall']:.4f}")
    print(f"    Accuracy (micro)  : {metrics['accuracy']:.4f}")
    
    # Per-class metrics
    print(f"\n  📋 PER-CLASS METRICS")
    print(f"  {'-'*70}")
    print(f"    Hateful (1)  | Precision: {metrics['precision_hateful']:.4f} | "
          f"Recall: {metrics['recall_hateful']:.4f} | F1: {metrics['f1_hateful']:.4f}")
    print(f"    Safe    (0)  | Precision: {metrics['precision_safe']:.4f} | "
          f"Recall: {metrics['recall_safe']:.4f} | F1: {metrics['f1_safe']:.4f}")
    
    # Confusion matrix
    if show_confusion:
        print(f"\n  🎯 CONFUSION MATRIX")
        print(f"  {'-'*70}")
        print(f"                      Predicted Safe  Predicted Hateful")
        print(f"    Actually Safe   :       {metrics['tn']:<4}          {metrics['fp']:<4}")
        print(f"    Actually Hateful:       {metrics['fn']:<4}          {metrics['tp']:<4}")
    
    # Performance stats
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        print(f"\n  ⏱️  PERFORMANCE")
        print(f"  {'-'*70}")
        print(f"    Avg latency  : {avg_latency:.2f}s per query")
        print(f"    Min/Max      : {min_latency:.2f}s / {max_latency:.2f}s")
        print(f"    Total time   : ~{sum(latencies)/60:.1f} minutes")
    
    print(f"\n{'='*70}\n")


def print_summary_table(all_metrics: Dict[str, Dict], show_rag: bool = True) -> None:
    """
    Prints a comparison table of all evaluation runs.
    
    Args:
        all_metrics: Dict mapping run labels to metric dicts
        show_rag: Whether to show RAG-specific metrics
    """
    print(f"\n{'='*70}")
    print(f"  SUMMARY TABLE")
    print(f"{'='*70}")
    
    # Header
    metric_names = ["Macro F1", "Precision", "Recall", "Accuracy", "TP", "FP", "FN", "TN"]
    header = "  Metric".ljust(20)
    for run_name in all_metrics.keys():
        header += f" | {run_name:>12}"
    print(header)
    print(f"  {'-'*70}")
    
    # Rows
    for metric_name in metric_names:
        row = f"  {metric_name}".ljust(20)
        for run_name, metrics in all_metrics.items():
            if metric_name == "Macro F1":
                val = metrics["macro_f1"]
            elif metric_name == "Precision":
                val = metrics["macro_precision"]
            elif metric_name == "Recall":
                val = metrics["macro_recall"]
            elif metric_name == "Accuracy":
                val = metrics["accuracy"]
            elif metric_name == "TP":
                val = metrics["tp"]
            elif metric_name == "FP":
                val = metrics["fp"]
            elif metric_name == "FN":
                val = metrics["fn"]
            elif metric_name == "TN":
                val = metrics["tn"]
            else:
                val = "—"
            
            if isinstance(val, float):
                row += f" | {val:>12.4f}"
            else:
                row += f" | {val:>12}"
        print(row)
    
    print(f"  {'-'*70}\n")


# ── main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MemeRAG on the dev.jsonl held-out set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate.py              # Full 500-sample evaluation
  python evaluate.py --sample 50  # Quick test with 50 samples
  python evaluate.py --verbose    # Detailed output for each query
        """
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Evaluate only on N random samples (useful for testing)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Timeout per query in seconds (default: {DEFAULT_TIMEOUT})"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output for each query"
    )
    
    args = parser.parse_args()

    # Load evaluation data
    try:
        df = load_eval_data(EVAL_PATH, max_samples=args.sample)
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print(f"   Please download dev.jsonl from Kaggle and place in data/dev.jsonl")
        sys.exit(1)
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

    # Run evaluation with RAG
    print("🚀 Starting RAG evaluation (with retrieval)...\n")
    rag_results = run_evaluation(df, query_timeout=args.timeout, verbose=args.verbose)
    rag_metrics = compute_metrics(rag_results["true"], rag_results["pred"])
    
    print_results(rag_metrics, rag_results["latencies"], "WITH RAG (Full Pipeline)")

    # Summary
    print(f"\n{'='*70}")
    print(f"  FINAL RESULTS FOR PROJECT REPORT")
    print(f"{'='*70}")
    print(f"  Metric                          Value")
    print(f"  {'-'*70}")
    print(f"  Macro F1 Score (primary)      : {rag_metrics['macro_f1']}")
    print(f"  Macro Precision               : {rag_metrics['macro_precision']}")
    print(f"  Macro Recall                  : {rag_metrics['macro_recall']}")
    print(f"  Accuracy                      : {rag_metrics['accuracy']}")
    print(f"  True Positives / False Pos    : {rag_metrics['tp']} / {rag_metrics['fp']}")
    print(f"  True Negatives / False Neg    : {rag_metrics['tn']} / {rag_metrics['fn']}")
    print(f"  {'-'*70}")
    print(f"  Evaluation set size           : {len(df)} memes")
    print(f"  Average latency per query     : {sum(rag_results['latencies'])/len(rag_results['latencies']):.2f}s")
    print(f"  Total evaluation time         : {sum(rag_results['latencies'])/60:.1f} minutes")
    print(f"{'='*70}\n")

    print("✅ Evaluation complete. Metrics are ready for Section 5 of your report.")
    print(f"📊 Use these numbers in your CS 6120 final report.\n")


if __name__ == "__main__":
    main()
