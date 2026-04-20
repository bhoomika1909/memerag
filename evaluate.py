"""
evaluate.py
-----------
Owner: Yazi
Project: MemeRAG (CS 6120 NLP)

What this file does:
    1. Loads twitter_eval.jsonl (100 balanced entries — never in ChromaDB)
    2. Runs each entry through the full pipeline (embed → retrieve → Llama 3)
    3. Compares predicted labels to ground truth
    4. Reports F1 (macro), precision, recall, accuracy, confusion matrix

Note on evaluation set:
    We use data/twitter_eval.jsonl (50 hateful, 50 safe from Twitter dataset)
    instead of data/dev.jsonl (Facebook Hateful Memes) because the Facebook
    dataset requires image context to classify correctly — hate is conveyed
    through the image, not the text. MemeRAG is a text-only system, so
    evaluating on text-based hate speech (Twitter) gives a fair measurement.

Input:  data/twitter_eval.jsonl
Output: Evaluation metrics, confusion matrix, and per-class performance

How to run:
    python evaluate.py              # Full 100-sample evaluation (~4 hours on CPU)
    python evaluate.py --sample 50  # Quick test with 50 samples (~2 hours on CPU)
    python evaluate.py --timeout 120  # Set per-query timeout to 2 minutes
    python evaluate.py --verbose    # Show detailed output for each query

Note: Full eval (~3-8 hours depending on GCP latency). Use --sample for testing.
      Make sure Ollama is running (ollama serve &) before running this script.
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
EVAL_PATH = "data/twitter_eval.jsonl"
DEFAULT_TIMEOUT = 120  # seconds per query
DEFAULT_SAMPLE_SIZE = 50  # None = full eval (100)

# ── step 1: load eval data ─────────────────────────────────────────────────
def load_eval_data(file_path: str, max_samples: Optional[int] = None) -> pd.DataFrame:
    """
    Loads twitter_eval.jsonl — the held-out evaluation set.
    This file is NEVER ingested into ChromaDB.
    """
    print(f"\n{'='*70}")
    print(f"  Loading evaluation data from {file_path} ...")
    print(f"{'='*70}")

    rows = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df = df[["id", "text", "label"]]

    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
        print(f"  📊 Using random sample of {max_samples} entries (seed=42)")

    print(f"  ✅ Loaded {len(df)} total evaluation entries")
    print(f"     • Hateful (1) : {df['label'].sum()} ({df['label'].mean()*100:.1f}%)")
    print(f"     • Safe    (0) : {(df['label'] == 0).sum()} ({(1-df['label'].mean())*100:.1f}%)")

    return df


# ── step 2: run pipeline on eval set ──────────────────────────────────────
def run_evaluation(df: pd.DataFrame, timeout: int = DEFAULT_TIMEOUT, verbose: bool = False) -> dict:
    """
    Runs every eval entry through the full RAG pipeline.
    Returns a dict with true labels, predicted labels, and latencies.
    """
    print(f"\n{'='*70}")
    print(f"  Running evaluation on {len(df)} entries ...")
    print(f"  Query timeout: {timeout}s per entry")
    print(f"{'='*70}\n")

    true_labels = []
    pred_labels = []
    latencies   = []
    errors      = 0

    for idx, (_, row) in enumerate(df.iterrows()):
        start = time.time()

        try:
            result = analyze_meme(str(row["text"]))
            raw    = result.get("hate_label", "uncertain").lower()
            pred   = 1 if ("hate" in raw and "not" not in raw) else 0

            if verbose:
                print(f"\n  [{idx+1}/{len(df)}] Text: {str(row['text'])[:60]}...")
                print(f"         True: {int(row['label'])} | Pred: {pred} | Label: {raw}")

        except Exception as e:
            pred = 0
            errors += 1
            if verbose:
                print(f"\n  [{idx+1}/{len(df)}] ERROR: {e}")

        end = time.time()
        latency = round(end - start, 2)

        true_labels.append(int(row["label"]))
        pred_labels.append(pred)
        latencies.append(latency)

        # progress bar
        pct     = int((idx + 1) / len(df) * 100)
        filled  = int(pct / 5)
        bar     = "█" * filled + "░" * (20 - filled)
        eta     = int((len(df) - idx - 1) * (sum(latencies) / len(latencies))) if latencies else 0
        print(f"\r  [{bar}] {idx+1}/{len(df)} | {pct:3d}% | Errors: {errors} | ETA: {eta}s", end="", flush=True)

    print(f"\n\n{'='*70}")
    print(f"  ✅ Evaluation complete!")
    print(f"  Total time: {sum(latencies):.1f}s")
    print(f"  Errors: {errors}/{len(df)} ({errors/len(df)*100:.1f}%)")
    print(f"{'='*70}")

    return {
        "true"      : true_labels,
        "pred"      : pred_labels,
        "latencies" : latencies,
        "errors"    : errors,
    }


# ── step 3: compute metrics ────────────────────────────────────────────────
def compute_metrics(true_labels: list, pred_labels: list) -> dict:
    """
    Computes F1 (macro), precision, recall, accuracy, and confusion matrix.
    Uses macro F1 as primary metric — handles class imbalance correctly.
    """
    tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(true_labels, pred_labels) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(true_labels, pred_labels) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == 1 and p == 0)

    total = len(true_labels)

    precision_1 = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_1    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_1        = 2 * precision_1 * recall_1 / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0

    precision_0 = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_0    = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_0        = 2 * precision_0 * recall_0 / (precision_0 + recall_0) if (precision_0 + recall_0) > 0 else 0

    macro_precision = (precision_0 + precision_1) / 2
    macro_recall    = (recall_0 + recall_1) / 2
    macro_f1        = (f1_0 + f1_1) / 2
    accuracy        = (tp + tn) / total if total > 0 else 0

    return {
        "accuracy"        : round(accuracy, 4),
        "macro_f1"        : round(macro_f1, 4),
        "macro_precision" : round(macro_precision, 4),
        "macro_recall"    : round(macro_recall, 4),
        "f1_hateful"      : round(f1_1, 4),
        "f1_safe"         : round(f1_0, 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


# ── step 4: print results ──────────────────────────────────────────────────
def print_results(metrics: dict, latencies: list, label: str = "WITH RAG") -> None:
    print(f"\n{'='*70}")
    print(f"  EVALUATION RESULTS — {label}")
    print(f"{'='*70}")

    print(f"\n  📊 MACRO-AVERAGED METRICS (primary for imbalanced classes)")
    print(f"  {'-'*70}")
    print(f"    Macro F1 Score    : {metrics['macro_f1']}  ← PRIMARY METRIC")
    print(f"    Macro Precision   : {metrics['macro_precision']}")
    print(f"    Macro Recall      : {metrics['macro_recall']}")
    print(f"    Accuracy (micro)  : {metrics['accuracy']}")

    print(f"\n  📋 PER-CLASS METRICS")
    print(f"  {'-'*70}")
    print(f"    Hateful (1)  | Precision: {metrics['f1_hateful']:.4f} | Recall: {metrics['macro_recall']:.4f} | F1: {metrics['f1_hateful']:.4f}")
    print(f"    Safe    (0)  | Precision: {metrics['f1_safe']:.4f} | Recall: {metrics['macro_recall']:.4f} | F1: {metrics['f1_safe']:.4f}")

    print(f"\n  🎯 CONFUSION MATRIX")
    print(f"  {'-'*70}")
    print(f"                      Predicted Safe  Predicted Hateful")
    print(f"    Actually Safe   :       {metrics['tn']}            {metrics['fp']}   ")
    print(f"    Actually Hateful:       {metrics['fn']}            {metrics['tp']}  ")

    print(f"\n  ⏱️  PERFORMANCE")
    print(f"  {'-'*70}")
    print(f"    Avg latency  : {round(sum(latencies)/len(latencies), 2)}s per query")
    print(f"    Min/Max      : {min(latencies)}s / {max(latencies)}s")
    print(f"    Total time   : ~{round(sum(latencies)/60, 1)} minutes")
    print(f"{'='*70}")


# ── main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MemeRAG on Twitter hate speech dataset")
    parser.add_argument("--sample", type=int, default=None, help="Evaluate only on N random samples")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Per-query timeout in seconds")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output per query")
    args = parser.parse_args()

    if not os.path.exists(EVAL_PATH):
        print(f"\n  ERROR: {EVAL_PATH} not found.")
        print(f"  Create it by running: python3 create_twitter_eval.py")
        sys.exit(1)

    df = load_eval_data(EVAL_PATH, max_samples=args.sample)

    print("\n🚀 Starting RAG evaluation (with retrieval)...\n")
    rag_results = run_evaluation(df, timeout=args.timeout, verbose=args.verbose)
    rag_metrics = compute_metrics(rag_results["true"], rag_results["pred"])
    print_results(rag_metrics, rag_results["latencies"], "WITH RAG (Full Pipeline)")

    print(f"\n\n{'='*70}")
    print(f"  FINAL RESULTS FOR PROJECT REPORT")
    print(f"{'='*70}")
    print(f"  {'Metric':<35} {'Value':>10}")
    print(f"  {'-'*47}")
    print(f"  {'Macro F1 Score (primary)':<35} : {rag_metrics['macro_f1']}")
    print(f"  {'Macro Precision':<35} : {rag_metrics['macro_precision']}")
    print(f"  {'Macro Recall':<35} : {rag_metrics['macro_recall']}")
    print(f"  {'Accuracy':<35} : {rag_metrics['accuracy']}")
    print(f"  {'True Positives / False Pos':<35} : {rag_metrics['tp']} / {rag_metrics['fp']}")
    print(f"  {'True Negatives / False Neg':<35} : {rag_metrics['tn']} / {rag_metrics['fn']}")
    print(f"  {'-'*47}")
    print(f"  {'Evaluation set size':<35} : {len(df)} memes (Twitter eval set)")
    print(f"  {'Average latency per query':<35} : {round(sum(rag_results['latencies'])/len(rag_results['latencies']), 2)}s")
    print(f"  {'Total evaluation time':<35} : {round(sum(rag_results['latencies'])/60, 1)} minutes")
    print(f"{'='*70}")
    print(f"\n✅ Evaluation complete. Metrics are ready for Section 5 of your report.")
    print(f"📊 Use these numbers in your CS 6120 final report.")
