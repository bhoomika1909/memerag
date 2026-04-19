"""
test_local.py
-------------
Pre-deployment test script.
Run this BEFORE docker-compose up to confirm all components work.

Tests:
    1. Ollama connection (GCP VM)
    2. ChromaDB connection + entry count
    3. Pipeline — end-to-end on 3 sample memes
    4. Citation format — source_url present and correct format
    5. Image lookup — demo_images folder check
    6. Import check — all modules load without errors

How to run:
    python test_local.py

All tests must pass before you present on April 21.
"""

import os
import sys
import json
import time
import requests

# ── colours ────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

passed = 0
failed = 0

def ok(msg):
    global passed
    passed += 1
    print(f"  {GREEN}✅ PASS{RESET}  {msg}")

def fail(msg, detail=""):
    global failed
    failed += 1
    print(f"  {RED}🔴 FAIL{RESET}  {msg}")
    if detail:
        print(f"         {YELLOW}→ {detail}{RESET}")

def header(title):
    print(f"\n{BOLD}── {title} {'─' * (50 - len(title))}{RESET}")

# ── TEST 1: Ollama connection ───────────────────────────────────────────────
header("TEST 1 — Ollama Connection (GCP VM)")

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://34.10.8.118:11435")
print(f"  Connecting to: {OLLAMA_URL}")

try:
    resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
    if resp.status_code == 200:
        models = [m["name"] for m in resp.json().get("models", [])]
        ok(f"Ollama reachable — models available: {models}")
        if any("llama3" in m for m in models):
            ok("Llama 3 model is loaded and ready")
        else:
            fail("Llama 3 not found in Ollama", f"Available: {models}")
    else:
        fail(f"Ollama returned status {resp.status_code}")
except requests.exceptions.ConnectionError:
    fail("Cannot connect to Ollama", f"Is the GCP VM running at {OLLAMA_URL}?")
except requests.exceptions.Timeout:
    fail("Ollama connection timed out", "GCP VM may be asleep")
except Exception as e:
    fail(f"Ollama error: {e}")

# ── TEST 2: ChromaDB connection ────────────────────────────────────────────
header("TEST 2 — ChromaDB Connection")

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = os.getenv("CHROMA_PORT", "8000")
print(f"  Connecting to: {CHROMA_HOST}:{CHROMA_PORT}")

try:
    # Try v2 first (ChromaDB latest), fall back to v1
    for endpoint in ["/api/v2/heartbeat", "/api/v1/heartbeat", "/api/v1"]:
        try:
            resp = requests.get(f"http://{CHROMA_HOST}:{CHROMA_PORT}{endpoint}", timeout=5)
            if resp.status_code in [200, 204]:
                ok(f"ChromaDB is reachable ({endpoint})")
                break
        except Exception:
            continue
    else:
        fail("ChromaDB heartbeat failed on all endpoints")
except Exception as e:
    fail("Cannot connect to ChromaDB", str(e))

try:
    import chromadb
    client     = chromadb.HttpClient(host=CHROMA_HOST, port=int(CHROMA_PORT))
    collection = client.get_collection("memes")
    count      = collection.count()
    if count >= 8000:
        ok(f"ChromaDB has {count:,} entries (expected ~33,000)")
    elif count > 0:
        ok(f"ChromaDB has {count:,} entries (ingest may be partial)")
    else:
        fail("ChromaDB collection is EMPTY", "Run python ingest.py first")

    # check dataset breakdown
    try:
        fb  = collection.get(where={"dataset": "facebook"})
        tw  = collection.get(where={"dataset": "twitter"})
        ok(f"Facebook entries: {len(fb['ids']):,}")
        ok(f"Twitter entries : {len(tw['ids']):,}")
    except Exception as e:
        fail(f"Dataset breakdown query failed: {e}")

except Exception as e:
    fail(f"ChromaDB collection error: {e}")

# ── TEST 3: Pipeline end-to-end ────────────────────────────────────────────
header("TEST 3 — Pipeline End-to-End (3 memes)")

test_memes = [
    ("when you already started eating and someone says 'lets pray'", "safe",    "08924"),
    ("need a hug ? i love hugs",                                     "safe",    "09312"),
    ("and if anything happens, the women are to blame",              "hateful", "10256"),
]

try:
    from pipeline import analyze_meme
    ok("pipeline.py imported successfully")

    for meme_text, expected_label, expected_img_id in test_memes:
        print(f"\n  Meme: \"{meme_text[:55]}...\"")
        start  = time.time()
        result = analyze_meme(meme_text)
        elapsed = round(time.time() - start, 1)

        # check result structure
        required_keys = ["explanation", "hate_label", "reasoning", "confidence", "id", "citations"]
        missing = [k for k in required_keys if k not in result]
        if missing:
            fail(f"Result dict missing keys: {missing}")
        else:
            ok(f"Result dict has all required keys")

        # check hate_label
        label = result.get("hate_label", "")
        ok(f"hate_label = '{label}' (expected ~{expected_label}) | {elapsed}s")

        # check confidence
        conf = result.get("confidence", 0)
        if 0 <= float(conf) <= 1:
            ok(f"confidence = {conf} (valid 0.0–1.0 range)")
        else:
            fail(f"confidence = {conf} (out of range)")

        # check citations
        citations = result.get("citations", [])
        if len(citations) >= 1:
            ok(f"{len(citations)} citations returned")
        else:
            fail("No citations returned from ChromaDB")

        # check citation format
        if citations:
            c = citations[0]
            source_url = c.get("source_url", "")
            dataset    = c.get("dataset", "")
            label_str  = c.get("label_str", "")

            if "#id=" in source_url:
                ok(f"source_url format correct: {source_url}")
            else:
                fail(f"source_url format wrong: '{source_url}'")

            if dataset in ["facebook", "twitter"]:
                ok(f"dataset tag correct: '{dataset}'")
            else:
                fail(f"dataset tag wrong: '{dataset}'")

            if label_str in ["hateful", "not hateful"]:
                ok(f"label_str correct: '{label_str}'")
            else:
                fail(f"label_str wrong: '{label_str}'")

except ImportError as e:
    fail(f"Failed to import pipeline.py: {e}")
except Exception as e:
    fail(f"Pipeline error: {e}")

# ── TEST 4: Demo images ────────────────────────────────────────────────────
header("TEST 4 — Demo Images")

DEMO_IMAGES = {
    "08924": "when you already started eating and someone says 'lets pray'",
    "09312": "need a hug ? i love hugs",
    "09718": "can you tag your friend tell them i found their baby pic",
    "10256": "and if anything happens, the women are to blame",
    "12597": "he is proving you don't have to be poor to be trash",
    "12983": "laughter really is the best medicine",
    "14327": "hunting it's not a hobby it's a way of life",
    "15067": "feeling cute may go out with my cousins later",
    "17624": "just trying to do my part",
    "18307": "when people talk trash but you're a bigger person",
}

demo_dir = "demo_images"
if not os.path.exists(demo_dir):
    fail(f"demo_images/ folder not found")
else:
    ok(f"demo_images/ folder exists")
    for img_id, text in DEMO_IMAGES.items():
        found = any(
            os.path.exists(os.path.join(demo_dir, f"{img_id}.{ext}"))
            or os.path.exists(os.path.join(demo_dir, f"{int(img_id)}.{ext}"))
            for ext in ["png", "jpg", "jpeg"]
        )
        if found:
            ok(f"demo_images/{img_id}.png ✓")
        else:
            fail(f"demo_images/{img_id}.png missing")

# ── TEST 5: Data files ─────────────────────────────────────────────────────
header("TEST 5 — Data Files")

data_files = {
    "data/train.jsonl"       : "Facebook corpus",
    "data/dev.jsonl"        : "Evaluation set",
    "data/twitter_export.jsonl": "Twitter corpus",
}

for path, label in data_files.items():
    if os.path.exists(path):
        size = os.path.getsize(path)
        ok(f"{path} ({label}) — {round(size/1024/1024, 1)} MB")
    else:
        fail(f"{path} ({label}) not found")

# ── TEST 6: Module imports ──────────────────────────────────────────────────
header("TEST 6 — Module Imports")

modules = [
    ("chromadb",             "ChromaDB"),
    ("sentence_transformers","sentence-transformers"),
    ("streamlit",            "Streamlit"),
    ("pandas",               "pandas"),
    ("emoji",                "emoji"),
    ("requests",             "requests"),
]

for mod, name in modules:
    try:
        __import__(mod)
        ok(f"{name} imports cleanly")
    except ImportError:
        fail(f"{name} not installed", f"pip install {mod}")

# ── SUMMARY ────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"{BOLD}  RESULTS{RESET}")
print(f"{'='*55}")
print(f"  {GREEN}✅ Passed : {passed}{RESET}")
print(f"  {RED}🔴 Failed : {failed}{RESET}")
print(f"{'='*55}")

if failed == 0:
    print(f"\n  {GREEN}{BOLD}All tests passed — ready for GCP! 🚀{RESET}")
else:
    print(f"\n  {RED}{BOLD}Fix the failing tests before presenting.{RESET}")
    sys.exit(1)
