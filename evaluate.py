"""
evaluate.py
-----------
Owner: TBD

What this file does:
    1. Runs the pipeline on 100 test memes (50 hateful, 50 not hateful)
    2. Compares predicted labels vs true labels
    3. Computes: macro F1, precision, recall, confusion matrix
    4. Also runs same test WITHOUT retrieval (LLM only baseline)
    5. Compares RAG vs no-RAG performance to prove retrieval helps
    6. Measures average response latency

Input:  test set (100 memes from Facebook dataset)
Output: printed evaluation report + saved results to results.txt

How to run:
    python evaluate.py
"""
