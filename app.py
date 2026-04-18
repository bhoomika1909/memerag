"""
app.py
------
Owner: TBD

What this file does:
    1. Streamlit web interface for MemeRAG
    2. Takes meme text input from the user
    3. Calls pipeline.py to get the analysis
    4. Displays explanation, hate label badge, reasoning
    5. Shows 5 retrieved similar memes with clickable citation links

How to run:
    streamlit run app.py

Then open browser at: http://localhost:8501
"""


import streamlit as st
from pipeline import analyze_meme

# ── page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MemeRAG — Hate Detection",
    page_icon="🔍",
    layout="centered"
)

# ── custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background-color: #0f0f0f;
    }

    .stApp {
        background-color: #0f0f0f;
        color: #f0f0f0;
    }

    .title-block {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }

    .title-block h1 {
        font-family: 'Space Mono', monospace;
        font-size: 2.4rem;
        color: #ffffff;
        letter-spacing: -1px;
        margin-bottom: 0.2rem;
    }

    .title-block p {
        color: #888;
        font-size: 0.95rem;
        font-weight: 300;
    }

    .label-hateful {
        background-color: #ff3b3b;
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 4px;
        font-family: 'Space Mono', monospace;
        font-size: 1.1rem;
        font-weight: 700;
        letter-spacing: 1px;
        display: inline-block;
        margin: 0.5rem 0;
    }

    .label-not-hateful {
        background-color: #00c853;
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 4px;
        font-family: 'Space Mono', monospace;
        font-size: 1.1rem;
        font-weight: 700;
        letter-spacing: 1px;
        display: inline-block;
        margin: 0.5rem 0;
    }

    .label-uncertain {
        background-color: #ff9800;
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 4px;
        font-family: 'Space Mono', monospace;
        font-size: 1.1rem;
        font-weight: 700;
        letter-spacing: 1px;
        display: inline-block;
        margin: 0.5rem 0;
    }

    .result-box {
        background-color: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }

    .result-box h4 {
        color: #888;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
        font-family: 'Space Mono', monospace;
    }

    .result-box p {
        color: #e0e0e0;
        font-size: 0.95rem;
        line-height: 1.6;
        margin: 0;
    }

    .citation-card {
        background-color: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-left: 3px solid #444;
        border-radius: 6px;
        padding: 0.9rem 1.2rem;
        margin: 0.5rem 0;
    }

    .citation-card.hateful {
        border-left-color: #ff3b3b;
    }

    .citation-card.not-hateful {
        border-left-color: #00c853;
    }

    .citation-text {
        color: #e0e0e0;
        font-size: 0.9rem;
        margin-bottom: 0.4rem;
        font-style: italic;
    }

    .citation-meta {
        font-family: 'Space Mono', monospace;
        font-size: 0.72rem;
        color: #666;
    }

    .citation-source {
        font-family: 'Space Mono', monospace;
        font-size: 0.72rem;
        color: #4a9eff;
        word-break: break-all;
    }

    .divider {
        border: none;
        border-top: 1px solid #2a2a2a;
        margin: 1.5rem 0;
    }

    .section-header {
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        color: #555;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin: 1.5rem 0 0.8rem 0;
    }

    .stTextArea textarea {
        background-color: #1a1a1a !important;
        color: #f0f0f0 !important;
        border: 1px solid #333 !important;
        border-radius: 6px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
    }

    .stTextArea textarea:focus {
        border-color: #555 !important;
        box-shadow: none !important;
    }

    .stButton > button {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: none !important;
        border-radius: 4px !important;
        font-family: 'Space Mono', monospace !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
        letter-spacing: 1px !important;
        padding: 0.6rem 2rem !important;
        width: 100% !important;
        transition: opacity 0.2s !important;
    }

    .stButton > button:hover {
        opacity: 0.85 !important;
    }

    .example-btn {
        background: none;
        border: 1px solid #333;
        color: #888;
        padding: 0.3rem 0.8rem;
        border-radius: 4px;
        font-size: 0.8rem;
        cursor: pointer;
        font-family: 'Inter', sans-serif;
        margin: 0.2rem;
        display: inline-block;
    }

    .footer {
        text-align: center;
        color: #444;
        font-size: 0.75rem;
        font-family: 'Space Mono', monospace;
        margin-top: 3rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# ── header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-block">
    <h1>🔍 MemeRAG</h1>
    <p>Meme Understanding & Hate Detection · RAG + Llama 3 · CS 6120 NLP · Northeastern University</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ── example memes ──────────────────────────────────────────────────────────
st.markdown("<p class='section-header'>Try an example</p>", unsafe_allow_html=True)

examples = [
    "nobody: literally nobody: me at 3am eating cereal",
    "i hate all immigrants they should go back",
    "when you finally finish your homework at 2am",
    "women belong in the kitchen not the workplace",
]

cols = st.columns(len(examples))
for i, example in enumerate(examples):
    if cols[i].button(f"#{i+1}", key=f"ex_{i}"):
        st.session_state["meme_input"] = example

# ── input area ─────────────────────────────────────────────────────────────
st.markdown("<p class='section-header'>Enter meme text</p>", unsafe_allow_html=True)

meme_text = st.text_area(
    label="meme text",
    label_visibility="collapsed",
    placeholder="Type or paste meme text here...",
    height=100,
    key="meme_input"
)

analyze_clicked = st.button("ANALYZE MEME →")

# ── results ────────────────────────────────────────────────────────────────
if analyze_clicked:
    if not meme_text or not meme_text.strip():
        st.warning("Please enter some meme text first.")
    else:
        with st.spinner("Retrieving similar memes and calling Llama 3... (20-60 seconds)"):
            result = analyze_meme(meme_text)

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("<p class='section-header'>Analysis Result</p>", unsafe_allow_html=True)

        # ── hate label badge ───────────────────────────────────────────────
        label = result.get("hate_label", "uncertain").lower()

        if label == "hateful":
            st.markdown('<div class="label-hateful">⚠ HATEFUL</div>', unsafe_allow_html=True)
        elif label == "not hateful":
            st.markdown('<div class="label-not-hateful">✓ NOT HATEFUL</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="label-uncertain">? UNCERTAIN</div>', unsafe_allow_html=True)

        # ── explanation ────────────────────────────────────────────────────
        st.markdown(f"""
        <div class="result-box">
            <h4>Explanation</h4>
            <p>{result.get("explanation", "No explanation available.")}</p>
        </div>
        """, unsafe_allow_html=True)

        # ── reasoning ──────────────────────────────────────────────────────
        st.markdown(f"""
        <div class="result-box">
            <h4>Reasoning</h4>
            <p>{result.get("reasoning", "No reasoning available.")}</p>
        </div>
        """, unsafe_allow_html=True)

        # ── citations ──────────────────────────────────────────────────────
        citations = result.get("citations", [])
        if citations:
            st.markdown("<hr class='divider'>", unsafe_allow_html=True)
            st.markdown("<p class='section-header'>Retrieved Sources — top 5 similar memes from database</p>", unsafe_allow_html=True)

            for i, c in enumerate(citations):
                label_str  = c.get("label_str", "unknown")
                text       = c.get("text", "")
                source_url = c.get("source_url", "")
                dataset    = c.get("dataset", "unknown")
                distance   = c.get("distance", 0)
                card_class = "hateful" if c.get("label") == 1 else "not-hateful"
                label_badge = "HATEFUL" if c.get("label") == 1 else "NOT HATEFUL"

                st.markdown(f"""
                <div class="citation-card {card_class}">
                    <div class="citation-text">"{text}"</div>
                    <div class="citation-meta">
                        [{i+1}] {label_badge} &nbsp;·&nbsp; {dataset.upper()} &nbsp;·&nbsp; similarity: {round(1 - distance, 4)}
                    </div>
                    <div class="citation-source">📎 {source_url}</div>
                </div>
                """, unsafe_allow_html=True)

# ── footer ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    MemeRAG · CS 6120 NLP · Spring 2026 · Northeastern University Silicon Valley Campus<br>
    Bhoomika Panday · Srijita Nath · Ibrahim · Yazi
</div>
""", unsafe_allow_html=True)
