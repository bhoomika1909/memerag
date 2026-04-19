"""
ingest_twitter.py
-----------------
Owner: Srijita Nath

What this file does:
    1. Loads the Twitter Hate Speech dataset (data/twitter_export.jsonl)
    2. Cleans tweet text (URLs, @mentions, RT markers removed)
    3. Generates sentence embeddings using all-MiniLM-L6-v2
    4. Stores into the SAME ChromaDB collection as Facebook data

NOTE: twitter_export.jsonl is pre-processed:
      - Labels are already binary (0 = not hateful, 1 = hateful)
      - IDs already start at 900000 (no clash with Facebook 40k-99k)
      - No label mapping needed

IMPORTANT:
    Run ingest.py FIRST to load Facebook data.
    Then run this file to ADD Twitter data to the same collection.
    Safe to run twice — skips duplicates automatically.

How to run:
    python ingest_twitter.py
"""

import json
import os
import re

import chromadb
import emoji
import pandas as pd
from sentence_transformers import SentenceTransformer

# ── settings ───────────────────────────────────────────────────────────────
TWITTER_DATA_PATH = "data/twitter_export.jsonl"  # pre-processed, already binary labels
CHROMA_PATH       = "data/chromadb"              # SAME path as ingest.py
COLLECTION_NAME   = "memes"                      # SAME collection as ingest.py
EMBED_MODEL       = "all-MiniLM-L6-v2"           # SAME model as ingest.py
MIN_TEXT_LENGTH   = 10
BATCH_SIZE        = 100


# ── step 1: load twitter data ──────────────────────────────────────────────
def load_twitter_data(file_path: str) -> pd.DataFrame:
    """
    Loads twitter_export.jsonl.
    Format: {"id": 900000, "text": "...", "label": 0}
    Labels already binary, IDs already start at 900000.
    """
    print(f"\n Loading Twitter data from {file_path} ...")

    rows = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df = df[["id", "text", "label"]]

    print(f" Loaded {len(df)} Twitter entries")
    print(f" Hateful (1)     : {df['label'].sum()}")
    print(f" Not hateful (0) : {(df['label'] == 0).sum()}")
    print(f" ID range        : {df['id'].min()} → {df['id'].max()}")

    return df


# ── step 2: clean text ─────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Cleans tweet text.
    Same base cleaning as ingest.py PLUS Twitter-specific:
    - Remove URLs
    - Remove @mentions
    - Remove RT marker
    """
    if not isinstance(text, str):
        return ""

    # Twitter-specific
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'^RT\s+', '', text)

    # same as ingest.py
    text = emoji.demojize(text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    return text


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    print("\n Cleaning tweet text ...")
    df = df.copy()
    df["text"] = df["text"].apply(clean_text)

    before = len(df)
    df = df[df["text"].str.len() >= MIN_TEXT_LENGTH]
    after  = len(df)
    print(f" Removed {before - after} entries (too short after cleaning)")
    print(f" Remaining : {after}")

    df["dataset"] = "twitter"

    # Citation URL — exact data chunk location as required by professor
    df["source_url"] = df["id"].apply(
        lambda x: f"data/twitter_export.jsonl#id={x}"
    )

    return df


# ── step 3: store into chromadb ────────────────────────────────────────────
def store_in_chromadb(df: pd.DataFrame) -> None:
    """
    Adds Twitter entries to the SAME ChromaDB collection as Facebook.
    Uses get_collection — ingest.py must run first.
    Skips duplicates — safe to run twice.
    """
    print(f"\n Connecting to ChromaDB at {CHROMA_PATH} ...")

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embed_model = SentenceTransformer(EMBED_MODEL)
    print(f" Embedding model loaded: {EMBED_MODEL}")

    try:
        collection = client.get_collection(name=COLLECTION_NAME)
        print(f" Found existing collection with {collection.count()} entries (Facebook data)")
    except Exception:
        print("\n ERROR: ChromaDB collection not found!")
        print(" Run ingest.py first, then run this script.")
        exit(1)

    print(f"\n Adding {len(df)} Twitter entries to ChromaDB ...")
    print(" This takes ~30-45 minutes. Please wait...\n")

    total   = len(df)
    added   = 0
    skipped = 0

    for start in range(0, total, BATCH_SIZE):
        batch = df.iloc[start : start + BATCH_SIZE]

        texts     = batch["text"].tolist()
        ids       = [f"twitter_{row_id}" for row_id in batch["id"].tolist()]
        metadatas = [
            {
                "label"      : int(row["label"]),
                "label_str"  : "hateful" if int(row["label"]) == 1 else "not hateful",
                "dataset"    : row["dataset"],
                "source_url" : row["source_url"],
                "meme_id"    : int(row["id"]),
                "id"         : int(row["id"]),
            }
            for _, row in batch.iterrows()
        ]

        # duplicate check — safe to re-run
        try:
            existing     = collection.get(ids=ids)
            existing_ids = set(existing["ids"])
        except Exception:
            existing_ids = set()

        new_indices = [i for i, id_ in enumerate(ids) if id_ not in existing_ids]

        if not new_indices:
            skipped += len(ids)
        else:
            new_texts     = [texts[i]     for i in new_indices]
            new_ids       = [ids[i]       for i in new_indices]
            new_metadatas = [metadatas[i] for i in new_indices]
            embeddings    = embed_model.encode(new_texts).tolist()

            collection.add(
                documents  = new_texts,
                embeddings = embeddings,
                metadatas  = new_metadatas,
                ids        = new_ids,
            )

            added   += len(new_indices)
            skipped += len(ids) - len(new_indices)

        done = min(start + BATCH_SIZE, total)
        if done % 2000 == 0 or done == total:
            print(f" Progress: {done}/{total} | Added: {added} | Skipped: {skipped}")

    print(f"\n Done!")
    print(f" Added   : {added} new Twitter entries")
    print(f" Skipped : {skipped} duplicates")
    print(f" Total in ChromaDB : {collection.count()}")


# ── step 4: verify combined collection ────────────────────────────────────
def verify_combined_collection() -> None:
    print("\n Verifying combined collection ...")

    client     = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)
    model      = SentenceTransformer(EMBED_MODEL)

    total = collection.count()
    print(f" Total entries in ChromaDB : {total}")

    fb_results = collection.get(where={"dataset": "facebook"})
    tw_results = collection.get(where={"dataset": "twitter"})

    print(f"   Facebook entries : {len(fb_results['ids'])}")
    print(f"   Twitter entries  : {len(tw_results['ids'])}")

    test_query      = "i hate immigrants"
    query_embedding = model.encode([test_query]).tolist()

    results = collection.query(
        query_embeddings = query_embedding,
        n_results        = 5,
        include          = ["documents", "metadatas", "distances"],
    )

    print(f"\n Test query: '{test_query}'")
    print(" Top 5 results:\n")

    for i in range(5):
        text     = results["documents"][0][i]
        label    = results["metadatas"][0][i]["label"]
        dataset  = results["metadatas"][0][i]["dataset"]
        url      = results["metadatas"][0][i]["source_url"]
        distance = results["distances"][0][i]
        lstr     = "HATEFUL" if label == 1 else "NOT HATEFUL"

        print(f"  [{i+1}] [{dataset.upper()}] {lstr}")
        print(f"       Text   : {text[:80]}...")
        print(f"       Source : {url}")
        print(f"       Dist   : {round(distance, 4)}")
        print()

    print(" Verification complete!\n")


# ── main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    if not os.path.exists(TWITTER_DATA_PATH):
        print(f"\n ERROR: {TWITTER_DATA_PATH} not found.")
        print(" Place twitter_export.jsonl in the data/ folder.")
        exit(1)

    df = load_twitter_data(TWITTER_DATA_PATH)
    df = clean_dataframe(df)
    store_in_chromadb(df)
    verify_combined_collection()

    print("=" * 55)
    print(" Twitter ingestion complete!")
    print(" Facebook + Twitter data are now in ChromaDB.")
    print(" Next: docker-compose up (on GCP VM)")
    print("=" * 55)
