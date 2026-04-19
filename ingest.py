"""
ingest.py
---------
Owner: Srijita Nath

What this file does:
    1. Loads the Facebook Hateful Memes dataset (data/train.jsonl)
    2. Cleans the meme text (lowercase, emoji → text, strip noise)
    3. Generates sentence embeddings using all-MiniLM-L6-v2
    4. Stores everything into ChromaDB (persistent local client)

Input:  data/train.jsonl — 8,500 labeled entries
Output: ChromaDB collection at data/chromadb/

IMPORTANT:
    Run this ONCE before docker-compose up.
    data/dev.jsonl is the evaluation set — NEVER ingest it here.
    Run ingest_twitter.py AFTER this to add Twitter data.

How to run:
    python ingest.py
"""

import json
import os
import re

import chromadb
import emoji
import pandas as pd
from sentence_transformers import SentenceTransformer

# ── settings ───────────────────────────────────────────────────────────────
DATA_PATH       = "data/train.jsonl"
CHROMA_PATH     = "data/chromadb"       # persistent local storage — matches docker-compose volume
COLLECTION_NAME = "memes"
EMBED_MODEL     = "all-MiniLM-L6-v2"
MIN_TEXT_LENGTH = 10


# ── step 1: load data ──────────────────────────────────────────────────────
def load_data(file_path: str) -> pd.DataFrame:
    """
    Reads train.jsonl line by line.
    Drops unlabeled rows (1,000 test entries mixed into the file).
    """
    print(f"\n Loading data from {file_path} ...")

    rows = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    df = pd.DataFrame(rows)

    # FIX — drop unlabeled rows before selecting columns
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df = df[["id", "text", "label"]]

    print(f" Loaded {len(df)} labeled entries")
    print(f" Hateful (1) : {df['label'].sum()}")
    print(f" Safe    (0) : {(df['label'] == 0).sum()}")

    return df


# ── step 2: clean text ─────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = emoji.demojize(text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    print("\n Cleaning text ...")
    df = df.copy()
    df["text"] = df["text"].apply(clean_text)

    before = len(df)
    df = df[df["text"].str.len() >= MIN_TEXT_LENGTH]
    after  = len(df)
    print(f" Removed {before - after} entries under {MIN_TEXT_LENGTH} chars")
    print(f" Remaining : {after}")

    df["dataset"] = "facebook"

    # Citation URL — exact data chunk location as required by professor
    df["source_url"] = df["id"].apply(
        lambda x: f"data/train.jsonl#id={x}"
    )

    return df


# ── step 3: store into chromadb ────────────────────────────────────────────
def store_in_chromadb(df: pd.DataFrame) -> None:
    """
    Uses PersistentClient — runs BEFORE docker-compose.
    Stores data to disk at data/chromadb/.
    Docker-compose then mounts this folder into the ChromaDB container.
    """
    print(f"\n Setting up ChromaDB at {CHROMA_PATH} ...")
    os.makedirs(CHROMA_PATH, exist_ok=True)

    client      = chromadb.PersistentClient(path=CHROMA_PATH)
    embed_model = SentenceTransformer(EMBED_MODEL)
    print(f" Embedding model loaded: {EMBED_MODEL}")

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    print(f"\n Embedding and storing {len(df)} entries ...")
    print(" This takes ~10-15 minutes. Please wait...\n")

    batch_size = 100
    total      = len(df)

    for start in range(0, total, batch_size):
        batch = df.iloc[start : start + batch_size]

        texts     = batch["text"].tolist()
        ids       = [f"facebook_{row_id}" for row_id in batch["id"].tolist()]
        metadatas = [
            {
                "label"      : int(row["label"]),
                "label_str"  : "hateful" if int(row["label"]) == 1 else "not hateful",
                "dataset"    : row["dataset"],
                "source_url" : row["source_url"],
                "meme_id"    : int(row["id"]),
                "id"         : int(row["id"]),   # stored as both keys for compatibility
            }
            for _, row in batch.iterrows()
        ]

        embeddings = embed_model.encode(texts).tolist()

        collection.add(
            documents  = texts,
            embeddings = embeddings,
            metadatas  = metadatas,
            ids        = ids,
        )

        done = min(start + batch_size, total)
        if done % 1000 == 0 or done == total:
            print(f" Progress: {done}/{total} ({int(done / total * 100)}%)")

    print(f"\n Done! {collection.count()} Facebook entries stored in ChromaDB.")
    print(f" Database saved at: {CHROMA_PATH}")


# ── step 4: verify ─────────────────────────────────────────────────────────
def verify_chromadb() -> None:
    print("\n Running verification test ...")

    client     = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)
    model      = SentenceTransformer(EMBED_MODEL)

    test_query      = "i hate you so much"
    query_embedding = model.encode([test_query]).tolist()

    results = collection.query(
        query_embeddings = query_embedding,
        n_results        = 3,
        include          = ["documents", "metadatas", "distances"],
    )

    print(f"\n Test query: '{test_query}'")
    print(" Top 3 results:\n")

    for i in range(3):
        text      = results["documents"][0][i]
        label     = results["metadatas"][0][i]["label"]
        url       = results["metadatas"][0][i]["source_url"]
        distance  = results["distances"][0][i]
        label_str = "HATEFUL" if label == 1 else "NOT HATEFUL"

        print(f"  [{i+1}] {label_str}")
        print(f"       Text   : {text}")
        print(f"       Source : {url}")
        print(f"       Dist   : {round(distance, 4)}")
        print()

    print(" Verification complete.\n")


# ── main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    if not os.path.exists(DATA_PATH):
        print(f"\n ERROR: {DATA_PATH} not found.")
        print(" Place train.jsonl in the data/ folder.")
        exit(1)

    df = load_data(DATA_PATH)
    df = clean_dataframe(df)
    store_in_chromadb(df)
    verify_chromadb()

    print(f" Facebook ingestion complete — {len(df)} entries stored.")
    print(" Next: python ingest_twitter.py")
