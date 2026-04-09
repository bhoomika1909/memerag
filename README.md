# MemeRAG — Meme Understanding & Hate Detection

**Course:** CS 6120 — Natural Language Processing | Northeastern University | Spring 2026

**Team:** Bhoomika Panday, Srijita Nath, Ibrahim, Yazi

---

## What This System Does

This system takes meme text as input and:

1. **Retrieves** the 5 most semantically similar memes from a database of 8,500+ human-labeled entries
2. **Explains** the meme's meaning in plain English using a locally-served LLM
3. **Classifies** the meme as hateful or not hateful with supporting reasoning
4. **Cites** every retrieved source with a clickable link back to the original entry

No external APIs are used — the LLM runs entirely on GCP via Ollama.

---

## System Pipeline

```
User Input (meme text)
         │
         ▼
  ┌──────────────┐
  │ Streamlit UI │   ← app.py
  └──────┬───────┘
         │
         ▼
  Sentence Embedding
  (all-MiniLM-L6-v2)
         │
         ▼
  ┌─────────────────────┐
  │  ChromaDB           │
  │  Vector Search      │   ← top-5 similar memes retrieved
  └──────┬──────────────┘
         │
         ▼
  Augmented Prompt
  (user meme + 5 retrieved examples + labels)
         │
         ▼
  ┌──────────────────────┐
  │  Llama 3 8B          │   ← running locally via Ollama on GCP
  └──────┬───────────────┘
         │
         ▼
  Explanation + Hate Label + Reasoning + Clickable Citations
```

---

## Approach

| Component | Naive Approach | Our Approach |
|-----------|---------------|--------------|
| Hate Detection | Keyword filter | RAG + LLM reasoning |
| Context | None | 5 retrieved similar memes |
| Hallucination risk | High | Low — grounded in retrieved examples |
| Citations | None | Clickable link per retrieved meme |
| LLM | External API | Local Llama 3 via Ollama on GCP |

---

## Why RAG?

Standard hate speech classifiers fail on memes because meme meaning is encoded in sarcasm, irony, and cultural references — not in literal words. For example:

- `"everybody loves chocolate chip cookies, even hitler"` → **NOT HATEFUL** (joke about cookies)
- `"same"` → **HATEFUL** (depends entirely on the image)

RAG addresses this by retrieving human-labeled examples similar to the input meme and giving that context to the LLM before asking it to decide. This reduces hallucination and improves accuracy on ambiguous cases.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| Vector Database | ChromaDB |
| Embedding Model | `all-MiniLM-L6-v2` (sentence-transformers) |
| LLM | Llama 3 8B via Ollama |
| Orchestration | LangChain |
| Infrastructure | Google Cloud Platform (GCP) |
| Containerization | Docker + docker-compose |
| Metadata Store | SQLite |

---

## Data

We use the **Facebook Hateful Memes Dataset** (Kiela et al., 2020) — 8,500 meme text entries with binary hate labels, specifically designed for meme hate detection.

| Property | Value |
|----------|-------|
| Total entries | 8,500 |
| Not hateful (label = 0) | 5,450 — 64.1% |
| Hateful (label = 1) | 3,050 — 35.9% |
| Average text length | 62 characters |
| Shortest meme | 4 characters |
| Longest meme | 433 characters |
| Source | https://hatefulmemeschallenge.com |

Each entry in the dataset:
```json
{
  "id": 42953,
  "img": "img/42953.png",
  "label": 0,
  "text": "its their character not their color that matters"
}
```

> **Note:** The `img` field is not used. MemeRAG is a text-only system. Only `id`, `text`, and `label` are ingested into ChromaDB.

### Class Imbalance

The dataset has a 64/36 split. A naive classifier that always predicts "not hateful" would score 64% accuracy — which is why we use **macro F1-score** as our primary metric, not accuracy.

### How to Download the Data

**Option A — Official challenge page:**
1. Go to https://hatefulmemeschallenge.com
2. Request free access (approved instantly)
3. Download and unzip
4. Place `train.jsonl` inside the `data/` folder

**Option B — Kaggle:**
```bash
pip install kaggle
kaggle datasets download -d parthplc/facebook-hateful-meme-dataset
unzip facebook-hateful-meme-dataset.zip -d data/
```

---

## Project Structure

```
memerag/
├── ingest.py            # data cleaning + ChromaDB ingestion      (Srijita)
├── pipeline.py          # LangChain RAG chain + Llama 3 prompt    (Bhoomika)
├── app.py               # Streamlit frontend + citations UI        (Ibrahim)
├── evaluate.py          # F1, precision, recall, confusion matrix  (Yazi)
├── docker-compose.yml   # one-command setup
├── requirements.txt     # all dependencies
├── data/
│   └── train.jsonl      # Facebook dataset — download separately (see above)
└── README.md
```

---

## Setup

### Option 1 — Docker (Recommended)

```bash
# Step 1 — clone the repo
git clone https://github.com/bhoomika1909/memerag.git
cd memerag

# Step 2 — add data file
# place train.jsonl inside the data/ folder

# Step 3 — start everything
docker-compose up
```

Open your browser → `http://localhost:8501`

---

### Option 2 — Manual

**Step 1 — Clone**
```bash
git clone https://github.com/YOUR_USERNAME/memerag.git
cd memerag
```

**Step 2 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 3 — Install Ollama and pull Llama 3**
```bash
# install from https://ollama.ai/download
ollama pull llama3
ollama serve
```

**Step 4 — Place data**
```
Copy train.jsonl into the data/ folder
```

**Step 5 — Ingest data into ChromaDB**
```bash
python ingest.py
```
> Embeds all 8,500 entries and stores them in ChromaDB. Takes ~10–15 minutes on first run.

**Step 6 — Run the app**
```bash
streamlit run app.py
```

Open your browser → `http://localhost:8501`

---

## Run Tests

```bash
python -m pytest tests/ -v
```

---

## Example Output

**Input:**
```
nobody: literally nobody: me at 3am eating cereal
```

**Output:**
```
Explanation:
This meme uses the "nobody: literally nobody:" format to highlight a 
relatable late-night habit. The humor comes from the contradiction 
between knowing you should be asleep and doing something mundane anyway.

Hate Label:  NOT HATEFUL

Reasoning:
No targeted group, slur, or harmful stereotype present. This is 
self-deprecating humor about a universal human experience.

Sources:
[1] "me at 3am when i have to wake up at 6"           → facebook #12453
[2] "nobody: me at 2am: let's reorganize my whole room" → facebook #67823
[3] "diet starts tomorrow. also me at midnight: snacks" → facebook #34109
[4] "i'll sleep early tonight. me at 3am:"             → facebook #89201
[5] "literally no one: me: rewatching the same show"   → facebook #44382
```

---

## Evaluation Results

| Metric | With RAG | Without RAG (LLM only) |
|--------|----------|------------------------|
| F1 Score (macro) | TBD | TBD |
| Precision | TBD | TBD |
| Recall | TBD | TBD |
| Avg. Response Time | TBD | TBD |

> Results will be updated after evaluation is complete in Week 2.

---

## Known Limitations

- **Very short memes:** 27 entries (0.3% of dataset) contain fewer than 10 characters and cannot be classified from text alone. The system returns an "insufficient context" message for these.
- **Image dependency:** Memes like `"same"` or `"real"` are only hateful because of their image. Text-only analysis cannot catch these — this is an acknowledged scope limitation.
- **English only:** Optimized for English meme text only.
- **Multimodal support:** Adding image analysis is listed as a future direction (LLaVA or similar).

---

## Team Responsibilities

| Member | Module | Key Files |
|--------|--------|-----------|
| **Bhoomika Panday** | RAG Pipeline & LLM | `pipeline.py` `README.md` |
| **Srijita Nath** | Data & ChromaDB | `ingest.py`,|
| **Ibrahim** | Frontend & Docker | `app.py`, `docker-compose.yml` |
| **Yazi** | Evaluation & Report | `evaluate.py` |

---

## Live Demo

Available at: `http://YOUR_GCP_IP:8501`

> Server is active on **presentation day: April 21, 2026**.
> A recorded backup demo is available if the live endpoint is temporarily unavailable.

---

## References

- Kiela et al. (2020) — [The Hateful Memes Challenge](https://arxiv.org/abs/2005.04790)
- Lewis et al. (2020) — [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- Basile et al. (2019) — HatEval: SemEval Task 5
- Facebook Hateful Memes Dataset — https://hatefulmemeschallenge.com
- ChromaDB Documentation — https://docs.trychroma.com
- Ollama — https://ollama.ai
- LangChain — https://python.langchain.com

---

*CS 6120 NLP · Spring 2026 · Northeastern University Silicon Valley Campus*
