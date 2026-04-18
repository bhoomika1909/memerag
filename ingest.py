"""
ingest.py
---------
Owner: Srijita Nath

What this file does:
    1. Loads the Facebook Hateful Memes dataset (train.jsonl)
    2. Cleans the meme text (lowercase, remove noise, handle emojis)
    3. Generates sentence embeddings using all-MiniLM-L6-v2
    4. Stores everything into ChromaDB vector database

Input:  data/train.jsonl
Output: ChromaDB collection with 8,500+ embedded meme entries

How to run:
    python ingest.py

Note: Run this once before starting the app.
      Takes ~10-15 minutes on first run.
"""

# ── imports ────────────────────────────────────────────────────────────────
import json
import os
import re

import chromadb
import emoji
import pandas as pd
from sentence_transformers import SentenceTransformer

# ── settings ───────────────────────────────────────────────────────────────
DATA_PATH       = "data/train.jsonl"       # where your dataset lives
CHROMA_PATH     = "data/chromadb"          # where ChromaDB will be saved
COLLECTION_NAME = "memes"                  # name of the ChromaDB collection
EMBED_MODEL     = "all-MiniLM-L6-v2"       # embedding model name
MIN_TEXT_LENGTH = 10                       # skip memes shorter than this


# ── step 1: load data ──────────────────────────────────────────────────────
def load_data(file_path: str) -> pd.DataFrame:
    """
    Reads the .jsonl file line by line.
    Each line is one meme entry like:
    {"id": 42953, "img": "img/42953.png", "label": 0, "text": "..."}
    Returns a pandas DataFrame.
    """
    print(f"\n Loading data from {file_path} ...")

    data = []
    with open(file_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            data.append(entry)

    df = pd.DataFrame(data)

    # keep only the columns we need — drop the image path
    df = df[["id", "text", "label"]]

    print(f" Loaded {len(df)} entries")
    print(f" Hateful (1): {df['label'].sum()}  |  Not hateful (0): {(df['label']==0).sum()}")

    return df


# ── step 2: clean text ─────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Cleans a single meme text string.
    Does 4 things:
        1. Converts emojis to text  e.g. 😂 → :face_with_tears_of_joy:
        2. Lowercases everything
        3. Removes extra whitespace
        4. Strips leading/trailing spaces
    """
    # convert emojis to readable text
    text = emoji.demojize(text)

    # lowercase
    text = text.lower()

    # remove extra whitespace (multiple spaces → single space)
    text = re.sub(r'\s+', ' ', text)

    # strip leading and trailing spaces
    text = text.strip()

    return text


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies clean_text() to every row.
    Also removes memes that are too short to classify (under 10 characters).
    Returns cleaned DataFrame.
    """
    print("\n Cleaning text ...")

    # apply cleaning to every meme text
    df["text"] = df["text"].apply(clean_text)

    # remove very short memes — can't classify without seeing the image
    before = len(df)
    df = df[df["text"].str.len() >= MIN_TEXT_LENGTH]
    after = len(df)

    print(f" Removed {before - after} entries (too short, under {MIN_TEXT_LENGTH} chars)")
    print(f" Remaining entries: {after}")

    # add dataset name — useful when you add more datasets later
    df["dataset"] = "facebook"

    # add source URL for clickable citations in the app
    # format: data/train.jsonl#id=42953
    df["source_url"] = df["id"].apply(
        lambda x: f"data/train.jsonl#id={x}"
    )

    return df


# ── step 3: store into chromadb ────────────────────────────────────────────
def store_in_chromadb(df: pd.DataFrame) -> None:
    """
    Takes the cleaned DataFrame and stores every entry into ChromaDB.

    ChromaDB automatically generates embeddings using the model we specify.
    Each entry stored has:
        - document  : the meme text (this gets embedded into a vector)
        - metadata  : label, dataset name, source URL, meme ID
        - id        : unique string ID for each entry

    After this runs, ChromaDB is saved to disk at data/chromadb/
    and can be loaded later by pipeline.py without re-running this script.
    """
    print(f"\n Setting up ChromaDB at {CHROMA_PATH} ...")

    # create a persistent ChromaDB client
    # persistent = saved to disk, not lost when script ends
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # load the sentence embedding model
    print(f" Loading embedding model: {EMBED_MODEL} ...")
    embed_model = SentenceTransformer(EMBED_MODEL)

    # create (or load if already exists) the collection
    # think of a collection like a table in a database
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # use cosine similarity for search
    )

    print(f"\n Embedding and storing {len(df)} entries into ChromaDB ...")
    print(" This will take 10-15 minutes. Please wait...\n")

    # process in batches of 100 to avoid memory issues
    batch_size = 100
    total = len(df)

    for start in range(0, total, batch_size):
        # get current batch
        batch = df.iloc[start : start + batch_size]

        # extract texts, ids, and metadata from batch
        texts     = batch["text"].tolist()
        ids       = [f"facebook_{row_id}" for row_id in batch["id"].tolist()]
        metadatas = [
            {
                "label"      : int(row["label"]),
                "dataset"    : row["dataset"],
                "source_url" : row["source_url"],
                "meme_id"    : int(row["id"])
            }
            for _, row in batch.iterrows()
        ]

        # generate embeddings for this batch
        embeddings = embed_model.encode(texts).tolist()

        # store into ChromaDB
        collection.add(
            documents  = texts,
            embeddings = embeddings,
            metadatas  = metadatas,
            ids        = ids
        )

        # show progress
        done = min(start + batch_size, total)
        print(f" Progress: {done}/{total} entries stored ({int(done/total*100)}%)")

    print(f"\n Done! {collection.count()} entries stored in ChromaDB.")
    print(f" Database saved at: {CHROMA_PATH}")


# ── step 4: verify it worked ───────────────────────────────────────────────
def verify_chromadb() -> None:
    """
    Quick sanity check after ingestion.
    Runs one test query and prints the top 3 results.
    This proves the database is working correctly.
    """
    print("\n Running verification test ...")

    client     = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)
    model      = SentenceTransformer(EMBED_MODEL)

    # test query
    test_query = "i hate you so much"
    query_embedding = model.encode([test_query]).tolist()

    results = collection.query(
        query_embeddings = query_embedding,
        n_results        = 3,
        include          = ["documents", "metadatas", "distances"]
    )

    print(f"\n Test query: '{test_query}'")
    print(" Top 3 similar memes retrieved:\n")

    for i in range(3):
        text      = results["documents"][0][i]
        label     = results["metadatas"][0][i]["label"]
        url       = results["metadatas"][0][i]["source_url"]
        distance  = results["distances"][0][i]
        label_str = "HATEFUL" if label == 1 else "NOT HATEFUL"

        print(f"  [{i+1}] {label_str}")
        print(f"       Text     : {text}")
        print(f"       Source   : {url}")
        print(f"       Distance : {round(distance, 4)}")
        print()

    print(" Verification complete — ChromaDB is working correctly!\n")


# ── main ─────────────────
if __name__ == "__main__":

    # check that the data file exists before starting
    if not os.path.exists(DATA_PATH):
        print(f"\n ERROR: Data file not found at '{DATA_PATH}'")
        print(" Please download train.jsonl and place it in the data/ folder.")
        print(" See README.md for download instructions.\n")
        exit(1)

    # run all 4 steps in order
    df = load_data(DATA_PATH)       # step 1 — load
    df = clean_dataframe(df)        # step 2 — clean
    store_in_chromadb(df)           # step 3 — store
    verify_chromadb()               # step 4 — verify

    print("=" * 50)
    print(" Ingestion complete! You can now run the app.")
    print(" Next step: streamlit run app.py")
    print("=" * 50)