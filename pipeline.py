"""
pipeline.py
-----------
Owner: Bhoomika Panday

What this file does:
    1. Takes meme text as input
    2. Embeds it using all-MiniLM-L6-v2
    3. Searches ChromaDB for top-5 most similar memes
    4. Builds an augmented prompt with those 5 memes
    5. Sends prompt to Llama 3 running on GCP via Ollama
    6. Returns structured result with explanation + hate label + citations

Input:  meme text (string)
Output: dict with keys:
        - explanation   (str)
        - hate_label    (str: "hateful" or "not hateful")
        - reasoning     (str)
        - citations     (list of dicts)

How to use:
    from pipeline import analyze_meme
    result = analyze_meme("nobody: literally nobody: me at 3am eating cereal")
    print(result)
"""

# ── imports ────────────────────────────────────────────────────────────────
import requests
import chromadb
from sentence_transformers import SentenceTransformer

# ── settings ───────────────────────────────────────────────────────────────
OLLAMA_URL      = "http://34.10.8.118:11435"   # GCP VM external IP + port
OLLAMA_MODEL    = "llama3"                      # model name
CHROMA_PATH     = "data/chromadb"              # path to ChromaDB database
COLLECTION_NAME = "memes"                      # ChromaDB collection name
EMBED_MODEL     = "all-MiniLM-L6-v2"           # same model used in ingest.py
TOP_K           = 5                            # number of similar memes to retrieve
MIN_TEXT_LENGTH = 10                           # minimum meme text length


# ── load models once (not every time analyze_meme is called) ───────────────
print("Loading embedding model...")
embed_model = SentenceTransformer(EMBED_MODEL)

print("Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection    = chroma_client.get_collection(COLLECTION_NAME)

print(f"Ready! ChromaDB has {collection.count()} memes.")


# ── step 1: check if meme text is long enough ──────────────────────────────
def is_too_short(text: str) -> bool:
    """
    Returns True if the meme text is too short to analyze.
    Very short memes (like 'same' or 'real') are only hateful
    because of their image — we can't classify them from text alone.
    """
    return len(text.strip()) < MIN_TEXT_LENGTH


# ── step 2: retrieve similar memes from chromadb ───────────────────────────
def retrieve_similar_memes(meme_text: str) -> list:
    """
    Embeds the input meme text and searches ChromaDB
    for the top-5 most semantically similar memes.

    Returns a list of dicts, each with:
        - text       : the similar meme's text
        - label      : 0 (not hateful) or 1 (hateful)
        - label_str  : "hateful" or "not hateful"
        - source_url : clickable citation link
        - dataset    : which dataset it came from
        - distance   : how similar it is (lower = more similar)
    """
    # embed the input text
    query_embedding = embed_model.encode([meme_text]).tolist()

    # search ChromaDB
    results = collection.query(
        query_embeddings = query_embedding,
        n_results        = TOP_K,
        include          = ["documents", "metadatas", "distances"]
    )

    # format results into clean list of dicts
    similar_memes = []
    for i in range(len(results["documents"][0])):
        label     = results["metadatas"][0][i]["label"]
        label_str = "hateful" if label == 1 else "not hateful"

        similar_memes.append({
            "text"      : results["documents"][0][i],
            "label"     : label,
            "label_str" : label_str,
            "source_url": results["metadatas"][0][i]["source_url"],
            "dataset"   : results["metadatas"][0][i]["dataset"],
            "distance"  : round(results["distances"][0][i], 4)
        })

    return similar_memes


# ── step 3: build the prompt ───────────────────────────────────────────────
def build_prompt(meme_text: str, similar_memes: list) -> str:
    """
    Builds the augmented prompt that gets sent to Llama 3.

    The prompt includes:
    - The 5 retrieved similar memes with their labels
    - The new meme to analyze
    - Clear instructions on what format to respond in

    This is the RAG part — we're giving the LLM context
    from real human-labeled examples before asking it to decide.
    """
    # format the retrieved memes as context
    context = ""
    for i, meme in enumerate(similar_memes):
        context += f"{i+1}. Text: \"{meme['text']}\" | Label: {meme['label_str']}\n"

    prompt = f"""You are a hate speech detection system for internet memes.

You have been given 5 similar memes from a labeled database.
Use these examples to understand the context and make your decision.

SIMILAR MEMES FROM DATABASE:
{context}
NEW MEME TO ANALYZE:
"{meme_text}"

Based on the similar examples above, please analyze this meme.
Respond in EXACTLY this format with no extra text:

EXPLANATION: [explain what this meme means in 2-3 sentences. consider sarcasm, irony, and cultural context]
LABEL: [hateful OR not hateful]
REASONING: [explain why in 1-2 sentences, referencing the similar examples if helpful]"""

    return prompt


# ── step 4: call llama 3 on gcp ───────────────────────────────────────────
def call_llama(prompt: str) -> str:
    """
    Sends the prompt to Llama 3 running on GCP via Ollama.
    Returns the raw text response from the LLM.
    """
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model" : OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=120   # wait up to 2 minutes for response
        )
        return response.json()["response"]

    except requests.exceptions.ConnectionError:
        return "ERROR: Cannot connect to Ollama. Is the GCP VM running?"
    except requests.exceptions.Timeout:
        return "ERROR: Ollama took too long to respond. Try again."
    except Exception as e:
        return f"ERROR: {str(e)}"


# ── step 5: parse llm response ─────────────────────────────────────────────
def parse_response(llm_response: str) -> dict:
    """
    Parses the LLM's text response into structured fields.

    The LLM is instructed to respond in this format:
        EXPLANATION: ...
        LABEL: hateful OR not hateful
        REASONING: ...

    If parsing fails (LLM didn't follow format), we return
    safe default values so the app doesn't crash.
    """
    explanation = "Could not parse explanation."
    hate_label  = "uncertain"
    reasoning   = "Could not parse reasoning."

    try:
        lines = llm_response.strip().split("\n")
        for line in lines:
            if line.startswith("EXPLANATION:"):
                explanation = line.replace("EXPLANATION:", "").strip()
            elif line.startswith("LABEL:"):
                label_text = line.replace("LABEL:", "").strip().lower()
                hate_label = "hateful" if "hateful" in label_text and "not" not in label_text else "not hateful"
            elif line.startswith("REASONING:"):
                reasoning = line.replace("REASONING:", "").strip()
    except Exception:
        pass   # if anything fails, return the defaults above

    return {
        "explanation": explanation,
        "hate_label" : hate_label,
        "reasoning"  : reasoning
    }


# ── main function: analyze_meme ────────────────────────────────────────────
def analyze_meme(meme_text: str) -> dict:
    """
    MAIN FUNCTION — this is what Ibrahim's app.py calls.

    Takes meme text as input.
    Returns a structured dict with everything needed for the UI.

    Example:
        result = analyze_meme("nobody: me at 3am eating cereal")
        print(result['explanation'])
        print(result['hate_label'])
        for citation in result['citations']:
            print(citation['source_url'])
    """

    # handle empty input
    if not meme_text or not meme_text.strip():
        return {
            "explanation": "Please enter some meme text.",
            "hate_label" : "uncertain",
            "reasoning"  : "No text was provided.",
            "citations"  : []
        }

    # handle very short memes
    if is_too_short(meme_text):
        return {
            "explanation": "This meme text is too short to analyze reliably without seeing the image.",
            "hate_label" : "uncertain",
            "reasoning"  : f"Meme text under {MIN_TEXT_LENGTH} characters cannot be classified from text alone.",
            "citations"  : []
        }

    # step 1 — retrieve similar memes from ChromaDB
    print(f"\nAnalyzing: '{meme_text}'")
    print("Retrieving similar memes from ChromaDB...")
    similar_memes = retrieve_similar_memes(meme_text)
    print(f"Found {len(similar_memes)} similar memes")

    # step 2 — build augmented prompt
    print("Building prompt...")
    prompt = build_prompt(meme_text, similar_memes)

    # step 3 — call Llama 3
    print("Calling Llama 3 on GCP (this takes 20-60 seconds)...")
    llm_response = call_llama(prompt)

    # handle errors from Llama
    if llm_response.startswith("ERROR:"):
        return {
            "explanation": llm_response,
            "hate_label" : "uncertain",
            "reasoning"  : "System error — please try again.",
            "citations"  : similar_memes
        }

    # step 4 — parse response
    print("Parsing response...")
    parsed = parse_response(llm_response)

    # step 5 — build final result
    result = {
        "explanation": parsed["explanation"],
        "hate_label" : parsed["hate_label"],
        "reasoning"  : parsed["reasoning"],
        "citations"  : similar_memes   # the 5 retrieved memes with source URLs
    }

    print(f"Done! Label: {result['hate_label']}")
    return result


# ── test the pipeline ──────────────────────────────────────────────────────
if __name__ == "__main__":

    # test with 3 different memes
    test_memes = [
        "nobody: literally nobody: me at 3am eating cereal",
        "i hate all immigrants they should go back",
        "when you finally finish your homework at 2am"
    ]

    for meme in test_memes:
        print("\n" + "="*60)
        result = analyze_meme(meme)
        print(f"\nMeme:        {meme}")
        print(f"Label:       {result['hate_label']}")
        print(f"Explanation: {result['explanation']}")
        print(f"Reasoning:   {result['reasoning']}")
        print(f"Citations:   {len(result['citations'])} sources retrieved")
        for i, citation in enumerate(result['citations']):
            print(f"  [{i+1}] {citation['label_str']} | {citation['source_url']}")
