"""
ingest_twitter.py
-----------------
Owner: Srijita Nath

What this file does:
    1. Loads the Twitter Hate Speech dataset (CSV format)
    2. Normalizes it to match the Facebook dataset structure exactly
    3. Cleans the tweet text (same cleaning as ingest.py)
    4. Generates sentence embeddings using all-MiniLM-L6-v2
    5. Stores everything into the SAME ChromaDB collection as Facebook data

Why this file exists:
    The Facebook dataset has ~8,500 entries which is below the professor's
    10,000 entry requirement. Adding ~25,000 Twitter entries brings the
    total to ~33,500 — well above the requirement.

Twitter dataset source (Kaggle):
    https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset

How to download:
    pip install kaggle
    kaggle datasets download -d mrmorj/hate-speech-and-offensive-language-dataset
    unzip hate-speech-and-offensive-language-dataset.zip -d data/

    OR manually download from Kaggle and place labeled_data.csv in data/ folder.

How to run:
    python ingest_twitter.py

IMPORTANT:
    Run ingest.py FIRST to load Facebook data.
    Then run this file to ADD Twitter data to the same collection.
    Do NOT run this file twice — it will skip duplicates automatically.

Twitter dataset label mapping:
    Original class column:
        0 = hate speech      → label 1 (hateful)
        1 = offensive        → label 0 (not hateful)
        2 = neither          → label 0 (not hateful)

    We map to binary to match Facebook:
        hateful     = 1
        not hateful = 0
"""

# ── imports ────────────────────────────────────────────────────────────────
import os
import re

import chromadb
import emoji
import pandas as pd
from sentence_transformers import SentenceTransformer

# ── settings ───────────────────────────────────────────────────────────────
TWITTER_DATA_PATH = "data/labeled_data.csv"    # Twitter CSV file
CHROMA_PATH       = "data/chromadb"            # same ChromaDB as ingest.py
COLLECTION_NAME   = "memes"                    # same collection as ingest.py
EMBED_MODEL       = "all-MiniLM-L6-v2"         # same model as ingest.py
MIN_TEXT_LENGTH   = 10                         # skip tweets shorter than this
BATCH_SIZE        = 100                        # process in batches


# ── step 1: load twitter data ──────────────────────────────────────────────
def load_twitter_data(file_path: str) -> pd.DataFrame:
    """
    Loads the Twitter hate speech CSV file.

    Raw CSV columns:
        Unnamed: 0  — row index (ignore)
        count       — number of crowdflower annotators
        hate_speech — votes for hate speech
        offensive_language — votes for offensive
        neither     — votes for neither
        class       — final label (0=hate, 1=offensive, 2=neither)
        tweet       — the tweet text

    We only need: class, tweet
    """
    print(f"\n Loading Twitter data from {file_path} ...")

    df = pd.read_csv(file_path)

    # keep only what we need
    df = df[["class", "tweet"]].copy()

    # rename to match Facebook structure
    df = df.rename(columns={"tweet": "text"})

    # create a unique ID for each Twitter entry
    # prefix with 90000 to avoid any overlap with Facebook IDs (which are ~42000-99000)
    df = df.reset_index(drop=True)
    df["id"] = df.index + 900000   # e.g. 900000, 900001, 900002 ...

    print(f" Loaded {len(df)} Twitter entries")
    print(f" Class distribution:")
    print(f"   Class 0 (hate speech): {(df['class'] == 0).sum()}")
    print(f"   Class 1 (offensive):   {(df['class'] == 1).sum()}")
    print(f"   Class 2 (neither):     {(df['class'] == 2).sum()}")

    return df


# ── step 2: map twitter labels to binary ──────────────────────────────────
def map_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps Twitter's 3-class labels to binary labels matching Facebook.

    Twitter original:
        0 = hate speech  → 1 (hateful)
        1 = offensive    → 0 (not hateful)
        2 = neither      → 0 (not hateful)

    We treat only explicit hate speech as hateful.
    Offensive language is treated as not hateful — this keeps
    the label consistent with Facebook's definition.
    """
    print("\n Mapping labels to binary (hate=1, not hate=0) ...")

    df["label"] = df["class"].map({
        0: 1,   # hate speech → hateful
        1: 0,   # offensive   → not hateful
        2: 0    # neither     → not hateful
    })

    # drop the original class column — no longer needed
    df = df.drop(columns=["class"])

    hateful     = df["label"].sum()
    not_hateful = (df["label"] == 0).sum()

    print(f" After mapping:")
    print(f"   Hateful (1):     {hateful}")
    print(f"   Not hateful (0): {not_hateful}")

    return df


# ── step 3: clean text ─────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Cleans a single tweet text string.
    Uses EXACTLY the same cleaning logic as ingest.py
    so both datasets are processed consistently.

    Does 4 things:
        1. Converts emojis to text  e.g. 😂 → :face_with_tears_of_joy:
        2. Lowercases everything
        3. Removes extra whitespace
        4. Strips leading/trailing spaces

    Also removes Twitter-specific noise:
        5. Removes URLs (http://... or https://...)
        6. Removes @mentions
        7. Removes RT (retweet marker)
    """
    if not isinstance(text, str):
        return ""

    # remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # remove @mentions
    text = re.sub(r'@\w+', '', text)

    # remove RT marker
    text = re.sub(r'^RT\s+', '', text)

    # convert emojis to readable text
    text = emoji.demojize(text)

    # lowercase
    text = text.lower()

    # remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # strip
    text = text.strip()

    return text


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies clean_text() to every row.
    Removes tweets that are too short after cleaning.
    Adds dataset tag and source_url to match Facebook structure.
    """
    print("\n Cleaning tweet text ...")

    df["text"] = df["text"].apply(clean_text)

    # remove empty or too-short entries
    before = len(df)
    df = df[df["text"].str.len() >= MIN_TEXT_LENGTH]
    after = len(df)

    print(f" Removed {before - after} entries (too short or empty after cleaning)")
    print(f" Remaining entries: {after}")

    # tag the dataset — this is how pipeline.py knows where each result came from
    df["dataset"] = "twitter"

    # source URL — links to the Kaggle dataset page
    # (no per-tweet URLs available unlike Facebook challenge)
    df["source_url"] = df["id"].apply(
        lambda x: f"data/labeled_data.csv#id={x}"
    )

    return df


# ── step 4: store into chromadb ────────────────────────────────────────────
def store_in_chromadb(df: pd.DataFrame) -> None:
    """
    Stores Twitter entries into the SAME ChromaDB collection as Facebook data.

    Each entry uses the prefix 'twitter_' in its ID to avoid
    any collision with Facebook entries (which use 'facebook_').

    Skips entries that already exist in ChromaDB — so you can
    safely re-run this script without creating duplicates.
    """
    print(f"\n Connecting to ChromaDB at {CHROMA_PATH} ...")

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # load the embedding model
    print(f" Loading embedding model: {EMBED_MODEL} ...")
    embed_model = SentenceTransformer(EMBED_MODEL)

    # get the existing collection — must already exist from ingest.py
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
        print(f" Found existing collection with {collection.count()} entries (Facebook data)")
    except Exception:
        print("\n ERROR: ChromaDB collection not found!")
        print(" Please run ingest.py first to load Facebook data.")
        print(" Then run this script to add Twitter data.\n")
        exit(1)

    print(f"\n Adding {len(df)} Twitter entries to ChromaDB ...")
    print(" This will take 30-45 minutes. Please wait...\n")

    total    = len(df)
    added    = 0
    skipped  = 0

    for start in range(0, total, BATCH_SIZE):
        batch = df.iloc[start : start + BATCH_SIZE]

        texts     = batch["text"].tolist()
        ids       = [f"twitter_{row_id}" for row_id in batch["id"].tolist()]
        metadatas = [
            {
                "label"      : int(row["label"]),
                "dataset"    : row["dataset"],
                "source_url" : row["source_url"],
                "meme_id"    : int(row["id"])
            }
            for _, row in batch.iterrows()
        ]

        # check which IDs already exist — avoid duplicates
        try:
            existing = collection.get(ids=ids)
            existing_ids = set(existing["ids"])
        except Exception:
            existing_ids = set()

        # filter out already-existing entries
        new_indices = [i for i, id_ in enumerate(ids) if id_ not in existing_ids]

        if not new_indices:
            skipped += len(ids)
        else:
            new_texts     = [texts[i]     for i in new_indices]
            new_ids       = [ids[i]       for i in new_indices]
            new_metadatas = [metadatas[i] for i in new_indices]

            # generate embeddings
            embeddings = embed_model.encode(new_texts).tolist()

            # store into ChromaDB
            collection.add(
                documents  = new_texts,
                embeddings = embeddings,
                metadatas  = new_metadatas,
                ids        = new_ids
            )

            added   += len(new_indices)
            skipped += len(ids) - len(new_indices)

        # show progress
        done = min(start + BATCH_SIZE, total)
        print(f" Progress: {done}/{total} processed | Added: {added} | Skipped (duplicates): {skipped}")

    print(f"\n Done!")
    print(f" Added:   {added} new Twitter entries")
    print(f" Skipped: {skipped} duplicate entries")
    print(f" Total entries in ChromaDB now: {collection.count()}")


# ── step 5: verify combined collection ────────────────────────────────────
def verify_combined_collection() -> None:
    """
    Runs a quick sanity check after ingestion.
    Shows how many Facebook vs Twitter entries are in ChromaDB.
    Also runs one test query to confirm retrieval works across both datasets.
    """
    print("\n Verifying combined collection ...")

    client     = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)
    model      = SentenceTransformer(EMBED_MODEL)

    total = collection.count()
    print(f" Total entries in ChromaDB: {total}")

    # count by dataset
    fb_results      = collection.get(where={"dataset": "facebook"})
    twitter_results = collection.get(where={"dataset": "twitter"})

    print(f"   Facebook entries: {len(fb_results['ids'])}")
    print(f"   Twitter entries:  {len(twitter_results['ids'])}")

    # test query
    test_query      = "i hate immigrants"
    query_embedding = model.encode([test_query]).tolist()

    results = collection.query(
        query_embeddings = query_embedding,
        n_results        = 5,
        include          = ["documents", "metadatas", "distances"]
    )

    print(f"\n Test query: '{test_query}'")
    print(" Top 5 results (from both datasets):\n")

    for i in range(5):
        text      = results["documents"][0][i]
        label     = results["metadatas"][0][i]["label"]
        dataset   = results["metadatas"][0][i]["dataset"]
        distance  = results["distances"][0][i]
        label_str = "HATEFUL" if label == 1 else "NOT HATEFUL"

        print(f"  [{i+1}] [{dataset.upper()}] {label_str}")
        print(f"       Text     : {text[:80]}...")
        print(f"       Distance : {round(distance, 4)}")
        print()

    print(" Verification complete — combined collection is working!\n")


# ── main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # check that Twitter data file exists
    if not os.path.exists(TWITTER_DATA_PATH):
        print(f"\n ERROR: Twitter data file not found at '{TWITTER_DATA_PATH}'")
        print(" Please download it from Kaggle:")
        print(" https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset")
        print(" Place labeled_data.csv in the data/ folder.\n")
        exit(1)

    # run all steps
    df = load_twitter_data(TWITTER_DATA_PATH)   # step 1 — load
    df = map_labels(df)                         # step 2 — map labels to binary
    df = clean_dataframe(df)                    # step 3 — clean text
    store_in_chromadb(df)                       # step 4 — store
    verify_combined_collection()                # step 5 — verify

    print("=" * 55)
    print(" Twitter ingestion complete!")
    print(" Both Facebook + Twitter data are now in ChromaDB.")
    print(" No changes needed to pipeline.py or app.py.")
    print(" Next step: streamlit run app.py")
    print("=" * 55)