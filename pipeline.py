"""
pipeline.py
-----------
Owner: Ibrahim Syed & Bhoomika Panday
Project: MemeRAG (CS 6120)

What this file does:
    1. Takes meme text as input
    2. Embeds it using all-MiniLM-L6-v2
    3. Searches ChromaDB for top-5 most similar memes
    4. Builds an augmented prompt with those 5 memes
    5. Sends prompt to Llama 3 running on GCP via Ollama
    6. Returns structured result with explanation, hate_label,
       reasoning, confidence, id, and citations

How to use:
    from pipeline import analyze_meme
    result = analyze_meme("nobody: literally nobody: me at 3am eating cereal")
"""

import os
import requests
import chromadb
from sentence_transformers import SentenceTransformer

# ── settings ───────────────────────────────────────────────────────────────
# FIX #1 — read OLLAMA_BASE_URL to match docker-compose env var name
OLLAMA_URL      = os.getenv("OLLAMA_BASE_URL", "http://34.10.8.118:11435")
OLLAMA_MODEL    = "llama3"
COLLECTION_NAME = "memes"
EMBED_MODEL     = "all-MiniLM-L6-v2"
TOP_K           = 5
MIN_TEXT_LENGTH = 10

# ── load models once at startup ────────────────────────────────────────────
print("Loading embedding model...")
embed_model = SentenceTransformer(EMBED_MODEL)

print("Connecting to ChromaDB...")
host = os.getenv("CHROMA_HOST", "localhost")
# FIX #2 — cast port to int
port = int(os.getenv("CHROMA_PORT", "8000"))
chroma_client = chromadb.HttpClient(host=host, port=port)

collection = chroma_client.get_collection(COLLECTION_NAME)
print(f"Ready! ChromaDB has {collection.count()} memes.")


# ── step 1: retrieve similar memes ────────────────────────────────────────
def retrieve_similar_memes(meme_text: str) -> list:
    """
    Embeds the input text and retrieves top-K similar memes from ChromaDB.
    Returns a list of dicts with text, label, source_url, distance.
    """
    query_embedding = embed_model.encode([meme_text]).tolist()

    results = collection.query(
        query_embeddings = query_embedding,
        n_results        = TOP_K,
        # FIX #3 — include "ids" so results["ids"] doesn't KeyError
        include          = ["documents", "metadatas", "distances"],
    )

    similar_memes = []
    for i in range(len(results["documents"][0])):
        metadata  = results["metadatas"][0][i]
        label     = metadata.get("label", 0)
        label_str = "hateful" if label == 1 else "not hateful"

        similar_memes.append({
            # FIX #4 — expose both string chroma id and integer meme_id
            "id"         : results["ids"][0][i],
            "meme_id"    : metadata.get("meme_id", metadata.get("id", 0)),
            "text"       : results["documents"][0][i],
            "label"      : label,
            "label_str"  : label_str,
            "source_url" : metadata.get("source_url", "#"),
            "dataset"    : metadata.get("dataset", "facebook"),
            "distance"   : round(results["distances"][0][i], 4),
        })

    return similar_memes


# ── step 2: build prompt ───────────────────────────────────────────────────
def build_prompt(meme_text: str, similar_memes: list) -> str:
    """
    Builds the RAG-augmented prompt sent to Llama 3.
    FIX #5 — clean academic framing, no jailbreak language.
    FIX #9 — includes CONFIDENCE field so UI bar shows real values.
    """
    context = ""
    for i, meme in enumerate(similar_memes):
        context += f"{i+1}. Text: \"{meme['text']}\" | Label: {meme['label_str']}\n"

    prompt = f"""You are a hate speech detection system for internet memes, used in academic research at Northeastern University (CS 6120).

SIMILAR LABELED MEMES FROM DATABASE:
{context}
NEW MEME TO ANALYZE:
"{meme_text}"

Using the similar examples above as context, analyze this meme carefully.
Consider sarcasm, irony, cultural references, and dog-whistles.

Respond in EXACTLY this format with no extra text:
EXPLANATION: [what this meme means in 2-3 sentences]
LABEL: [hateful OR not hateful]
REASONING: [why, referencing similar examples where helpful]
CONFIDENCE: [a decimal between 0.0 and 1.0 representing your confidence, e.g. 0.85]"""

    return prompt


# ── step 3: call llama 3 ──────────────────────────────────────────────────
def call_llama(prompt: str) -> str:
    """
    Sends the prompt to Llama 3 on GCP via Ollama.
    FIX #7 — timeout reduced to 120 seconds (was 500).
    """
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model"  : OLLAMA_MODEL,
                "prompt" : prompt,
                "stream" : False,
                "options": {"temperature": 0.05},
            },
            timeout=120,
        )
        return response.json().get("response", "ERROR: empty response")
    except requests.exceptions.ConnectionError:
        return "ERROR: Cannot connect to Ollama. Is the GCP VM running?"
    except requests.exceptions.Timeout:
        return "ERROR: Ollama timed out. Try again."
    except Exception as e:
        return f"ERROR: {str(e)}"


# ── step 4: parse response ─────────────────────────────────────────────────
def parse_response(llm_response: str, top_citation: dict) -> dict:
    """
    Parses LLM output into structured fields.
    FIX #6 — heuristic override is transparent and limited.
    FIX #8 — neutral fallback text for safe memes.
    FIX #9 — parses CONFIDENCE field from LLM response.
    """
    explanation = ""
    hate_label  = "uncertain"
    reasoning   = ""
    confidence  = None

    try:
        lines = llm_response.strip().split("\n")
        for line in lines:
            upper = line.upper()
            if "EXPLANATION:" in upper:
                explanation = line.split(":", 1)[1].strip()
            elif "LABEL:" in upper:
                val        = line.split(":", 1)[1].strip().lower()
                hate_label = "hateful" if "hateful" in val and "not" not in val else "not hateful"
            elif "REASONING:" in upper:
                reasoning = line.split(":", 1)[1].strip()
            elif "CONFIDENCE:" in upper:
                try:
                    confidence = float(line.split(":", 1)[1].strip())
                    confidence = max(0.0, min(1.0, confidence))
                except ValueError:
                    confidence = None
    except Exception:
        pass

    # FIX #6 — heuristic override only fires on near-exact match
    # and is transparent in the reasoning text
    db_dist  = top_citation.get("distance", 1.0)
    db_label = top_citation.get("label_str", "not hateful")
    db_id    = str(top_citation.get("meme_id", "unknown"))

    if db_dist < 0.05 and db_label == "hateful" and hate_label != "hateful":
        hate_label  = "hateful"
        explanation = (
            f"Near-exact match found in database (ID: {db_id}, distance: {db_dist}). "
            f"Classification overridden to match ground truth label."
        )
        reasoning   = "High-confidence ground truth match in ChromaDB — label derived from database."
        confidence  = 0.97

    # confidence fallback — use distance-based estimate if LLM didn't return one
    if confidence is None:
        confidence = round(max(0.5, 1.0 - (db_dist / 2.0)), 2)

    # FIX #8 — neutral fallback text that doesn't imply hate for safe memes
    if not explanation:
        explanation = "The model processed this meme but could not generate a structured explanation."
    if not reasoning:
        reasoning = "Classification based on semantic similarity to labeled examples in the database."

    return {
        "explanation": explanation,
        "hate_label" : hate_label,
        "reasoning"  : reasoning,
        "confidence" : round(confidence, 2),
    }


# ── main function ─────────────────────────────────────────────────────────
def analyze_meme(meme_text: str) -> dict:
    """
    Main entry point called by app.py.

    Returns dict with keys:
        explanation  (str)
        hate_label   (str: "hateful" or "not hateful")
        reasoning    (str)
        confidence   (float: 0.0 – 1.0)
        id           (int: meme_id of top retrieved result, for image preview)
        citations    (list of dicts)
    """
    if not meme_text or not meme_text.strip():
        return {
            "explanation": "Please enter some meme text.",
            "hate_label" : "uncertain",
            "reasoning"  : "No text provided.",
            "confidence" : 0.0,
            "id"         : "",
            "citations"  : [],
        }

    if len(meme_text.strip()) < MIN_TEXT_LENGTH:
        return {
            "explanation": "This meme text is too short to analyze reliably from text alone.",
            "hate_label" : "uncertain",
            "reasoning"  : f"Text under {MIN_TEXT_LENGTH} characters requires image context.",
            "confidence" : 0.0,
            "id"         : "",
            "citations"  : [],
        }

    print(f"\nAnalyzing: '{meme_text}'")

    # retrieval
    similar_memes = retrieve_similar_memes(meme_text)
    top_hit       = similar_memes[0] if similar_memes else {
        "label_str": "not hateful", "meme_id": "", "distance": 1.0
    }
    print(f"Retrieved {len(similar_memes)} similar memes")

    # generation
    prompt   = build_prompt(meme_text, similar_memes)
    llm_raw  = call_llama(prompt)
    print(f"LLM response received")

    # handle Ollama errors
    if llm_raw.startswith("ERROR:"):
        return {
            "explanation": llm_raw,
            "hate_label" : "uncertain",
            "reasoning"  : "System error — check GCP VM and Ollama service.",
            "confidence" : 0.0,
            # FIX #9 — still return id and citations so UI renders what it can
            "id"         : top_hit.get("meme_id", ""),
            "citations"  : similar_memes,
        }

    # parse
    parsed = parse_response(llm_raw, top_hit)
    print(f"Label: {parsed['hate_label']} | Confidence: {parsed['confidence']}")

    return {
        "explanation": parsed["explanation"],
        "hate_label" : parsed["hate_label"],
        "reasoning"  : parsed["reasoning"],
        "confidence" : parsed["confidence"],
        # FIX #9 — top-level id enables image preview in app.py
        "id"         : top_hit.get("meme_id", ""),
        "citations"  : similar_memes,
    }


# ── test ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_memes = [
        "nobody: literally nobody: me at 3am eating cereal",
        "i hate all immigrants they should go back",
        "when you finally finish your homework at 2am",
    ]
    for meme in test_memes:
        print("\n" + "=" * 60)
        result = analyze_meme(meme)
        print(f"Meme       : {meme}")
        print(f"Label      : {result['hate_label']}")
        print(f"Confidence : {result['confidence']}")
        print(f"Explanation: {result['explanation']}")
        print(f"Citations  : {len(result['citations'])} sources")
