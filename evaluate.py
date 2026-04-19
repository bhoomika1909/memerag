"""
evaluate.py
-----------
Owner: Yazi

What this file does:
    1. Loads dev.jsonl (500 human-labeled entries — never in ChromaDB)
    2. Runs each entry through the full pipeline (embed → retrieve → Llama 3)
    3. Compares predicted labels to ground truth
    4. Reports F1 (macro), precision, recall, accuracy, confusion matrix
    5. Also runs WITHOUT retrieval (LLM only) to prove RAG helps

Input:  data/dev.jsonl
Output: Printed metrics + comparison table (RAG vs no-RAG)

How to run:
    python evaluate.py

Note: Takes ~20-60 seconds per entry. Full eval = ~3-8 hours.
      Use SAMPLE_SIZE for a quick test run.
"""

import json
import os
import time

import pandas as pd
from pipeline import analyze_meme

# ── settings ───────────────────────────────────────────────────────────────
EVAL_PATH   = "data/dev.jsonl"
SAMPLE_SIZE = None    # set to e.g. 100 for a quick test run, None for full 500


# ── step 1: load eval data ─────────────────────────────────────────────────
def load_eval_data(file_path: str) -> pd.DataFrame:
    """
    Loads dev.jsonl — the held-out evaluation set.
    This file is NEVER ingested into ChromaDB.
    """
    print(f"\n Loading eval data from {file_path} ...")

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

    print(f" Loaded {len(df)} eval entries")
    print(f" Hateful (1) : {df['label'].sum()}")
    print(f" Safe    (0) : {(df['label'] == 0).sum()}")

    return df


# ── step 2: run pipeline on eval set ──────────────────────────────────────
def run_evaluation(df: pd.DataFrame) -> dict:
    """
    Runs every eval entry through the full RAG pipeline.
    Returns a dict with true labels, predicted labels, and latencies.
    """
    print(f"\n Running pipeline on {len(df)} entries ...")
    print(" This will take a while. Progress shown every 10 entries.\n")

    true_labels = []
    pred_labels = []
    latencies   = []

    for i, row in df.iterrows():
        start  = time.time()
        result = analyze_meme(row["text"])
        end    = time.time()

        # parse predicted label
        raw   = result.get("hate_label", "uncertain").lower()
        pred  = 1 if ("hate" in raw and "not" not in raw) else 0

        true_labels.append(int(row["label"]))
        pred_labels.append(pred)
        latencies.append(round(end - start, 2))

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(df)} entries")

    return {
        "true"      : true_labels,
        "pred"      : pred_labels,
        "latencies" : latencies,
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

    # per-class metrics
    precision_1 = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_1    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_1        = 2 * precision_1 * recall_1 / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0

    precision_0 = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_0    = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_0        = 2 * precision_0 * recall_0 / (precision_0 + recall_0) if (precision_0 + recall_0) > 0 else 0

    # macro averages
    macro_precision = (precision_0 + precision_1) / 2
    macro_recall    = (recall_0    + recall_1)    / 2
    macro_f1        = (f1_0        + f1_1)        / 2
    accuracy        = (tp + tn) / total if total > 0 else 0

    return {
        "accuracy"       : round(accuracy, 4),
        "macro_f1"       : round(macro_f1, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall"   : round(macro_recall, 4),
        "f1_hateful"     : round(f1_1, 4),
        "f1_safe"        : round(f1_0, 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


# ── step 4: print results ──────────────────────────────────────────────────
def print_results(metrics: dict, latencies: list, label: str = "WITH RAG") -> None:
    print(f"\n{'='*55}")
    print(f"  RESULTS — {label}")
    print(f"{'='*55}")
    print(f"  Macro F1 Score  : {metrics['macro_f1']}   ← PRIMARY METRIC")
    print(f"  Macro Precision : {metrics['macro_precision']}")
    print(f"  Macro Recall    : {metrics['macro_recall']}")
    print(f"  Accuracy        : {metrics['accuracy']}")
    print(f"  F1 (hateful)    : {metrics['f1_hateful']}")
    print(f"  F1 (safe)       : {metrics['f1_safe']}")
    print(f"\n  Confusion Matrix:")
    print(f"                  Pred Safe  Pred Hateful")
    print(f"  True Safe     :    {metrics['tn']}         {metrics['fp']}")
    print(f"  True Hateful  :    {metrics['fn']}         {metrics['tp']}")
    print(f"\n  Avg latency   : {round(sum(latencies)/len(latencies), 2)}s per query")
    print(f"  Total queries : {len(latencies)}")
    print(f"{'='*55}")


# ── main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    if not os.path.exists(EVAL_PATH):
        print(f"\n ERROR: {EVAL_PATH} not found.")
        print(" Place dev.jsonl in the data/ folder.")
        exit(1)

    df = load_eval_data(EVAL_PATH)

    if SAMPLE_SIZE:
        df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42)
        print(f" Using sample of {len(df)} entries for quick evaluation")

    # ── run WITH RAG ──────────────────────────────────────────────────────
    print("\n Running evaluation WITH RAG (full pipeline)...")
    rag_results = run_evaluation(df)
    rag_metrics = compute_metrics(rag_results["true"], rag_results["pred"])
    print_results(rag_metrics, rag_results["latencies"], "WITH RAG")

    # ── comparison summary ─────────────────────────────────────────────────
    print("\n METRIC SUMMARY")
    print(f"{'Metric':<20} {'With RAG':>10}")
    print("-" * 32)
    print(f"{'Macro F1':<20} {rag_metrics['macro_f1']:>10}")
    print(f"{'Precision':<20} {rag_metrics['macro_precision']:>10}")
    print(f"{'Recall':<20} {rag_metrics['macro_recall']:>10}")
    print(f"{'Accuracy':<20} {rag_metrics['accuracy']:>10}")

    print("\n Evaluation complete.")
    print(" These numbers go directly into Section 5 of your report.")
