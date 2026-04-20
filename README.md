# MemeRAG — Meme Understanding & Hate Detection

**Course:** CS 6120 — Natural Language Processing | Northeastern University | Spring 2026  
**Team:** Bhoomika Panday · Srijita Nath · Syed Ibrahim Saleem · Yazi

---

## What This System Does

MemeRAG takes meme text as input and:
- Retrieves the 5 most semantically similar memes from a database of **33,000 labeled entries** (Facebook + Twitter) 
- Explains the meme's meaning in plain English using **Llama 3** running locally on GCP
- Classifies the meme as **hateful or not hateful** with supporting reasoning
- Cites every retrieved source with a **clickable link** back to the original dataset entry

No external APIs — the LLM runs entirely on GCP via Ollama.

---

## System Architecture

```
User Input (meme text)
        │
        ▼
  Streamlit UI (app.py)
        │
        ▼
  Sentence Embedding (all-MiniLM-L6-v2)
        │
        ▼
  ChromaDB Vector Search → top-5 similar memes (Facebook + Twitter)
        │
        ▼
  Augmented Prompt → Llama 3 8B via Ollama on GCP
        │
        ▼
  Explanation + Hate Label + Reasoning + Clickable Citations
```

---

## Datasets

| Dataset | Entries | Source |
|---------|---------|--------|
| Facebook Hateful Memes (Kiela et al., 2020) | 8,473 | [Kaggle](https://www.kaggle.com/datasets/parthplc/facebook-hateful-meme-dataset) |
| Twitter Hate Speech (Davidson et al., 2017) | 24,527 | [Kaggle](https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset) |
| **Total corpus** | **33,000** | — |
| Evaluation set (`dev.jsonl`) | 500 | Facebook dev set |

---

## Project Structure

```
memerag/
├── app.py                  # Streamlit UI
├── pipeline.py             # RAG chain + Llama 3 prompt
├── ingest.py               # Facebook data → ChromaDB
├── ingest_twitter.py       # Twitter data → ChromaDB
├── evaluate.py             # F1, precision, recall, confusion matrix
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .dockerignore
├── README.md
├── data/
│   ├── train.jsonl         # Facebook corpus (download separately)
│   ├── dev.jsonl          # Evaluation set (never ingested)
│   ├── labeled_data.csv    # Twitter corpus (download separately)
│   └── chromadb/           # Populated by ingest.py + ingest_twitter.py
└── demo_images/            # 10 demo meme images
```

---

---

## Setup & Run (GCP — Recommended)

### Step 1 — Clone the repo

```bash
git clone https://github.com/bhoomika1909/memerag.git
cd memerag
```

### Step 2 — Download datasets

**Facebook dataset:**
```bash
pip install kaggle
kaggle datasets download -d parthplc/facebook-hateful-meme-dataset
unzip facebook-hateful-meme-dataset.zip -d data/
mv data/data/train.jsonl data/train.jsonl
mv data/data/dev.jsonl data/dev.jsonl
```

**Twitter dataset:**
```bash
kaggle datasets download -d mrmorj/hate-speech-and-offensive-language-dataset
unzip hate-speech-and-offensive-language-dataset.zip -d data/
# This creates data/labeled_data.csv
# ingest_twitter.py reads labeled_data.csv and processes it into twitter_export.jsonl
```

### Step 3 — Build the ChromaDB database (run once)

```bash
python ingest.py             # Facebook: ~8,473 entries, ~10-15 min
python ingest_twitter.py     # Twitter: ~24,527 entries, ~30-45 min
```

> **IMPORTANT:** ChromaDB data is stored in `data/chromadb/` (no underscore). Do not rename this folder.

### Step 4 — Start Ollama and the app

```bash
ollama pull llama3
ollama serve &
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

### Step 5 — Open the app
http://YOUR_GCP_EXTERNAL_IP:8501

> **GCP firewall rules required:** ports `8501` (Streamlit), `11434` (Ollama)

---

## Evaluation

```bash
python evaluate.py              # Full 500-sample evaluation (~3-8 hours on CPU)
python evaluate.py --sample 50  # Quick test with 50 samples (~1 hour on CPU)
```

Outputs: F1 score (macro), precision, recall, accuracy, confusion matrix.  
Evaluation set: `data/dev.jsonl` — 500 human-labeled entries, never ingested into ChromaDB.

> **Note:** Make sure Ollama is running (`ollama serve &`) before running evaluation, otherwise all predictions will default to "not hateful".

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| Vector Database | ChromaDB (PersistentClient) |
| Embedding Model | all-MiniLM-L6-v2 |
| LLM | Llama 3 8B via Ollama |
| Infrastructure | Google Cloud Platform (e2-standard-4) |
| Containerization | Docker + docker-compose |

---

## Contributions

| Member | Module | Files |
|--------|--------|-------|
| Bhoomika Panday | RAG Pipeline, LLM Prompt Engineering, ChromaDB Setup, GCP Deployment | `pipeline.py`, `ingest.py` |
| Srijita Nath | Data Ingestion, Twitter Data Transformation, GCP VM Setup, Pipeline Debugging & Error Correction, Prompt Engineering, Evaluation Design, README | `ingest.py`, `ingest_twitter.py`, `README.md`, `data/twitter_eval.jsonl` |
| Syed Ibrahim Saleem | Streamlit Frontend, Docker, GCP Deployment, Report Writing | `app.py`, `docker-compose.yml`, `Dockerfile` |
| Yazi | Evaluation Framework, F1/Precision/Recall Metrics, Report Writing | `evaluate.py` |

---

## References

- Kiela et al. (2020) — The Hateful Memes Challenge
- Davidson et al. (2017) — Automated Hate Speech Detection
- Lewis et al. (2020) — Retrieval-Augmented Generation
- ChromaDB: https://docs.trychroma.com
- Ollama: https://ollama.ai
