"""
app.py
------
Owner: Syed Ibrahim Saleem

MemeRAG — Streamlit UI (Professional Edition)

What this file does:
1. Streamlit web interface for MemeRAG
2. Takes meme text input from the user
3. Calls pipeline.py to retrieve similar memes and run Llama 3
4. Displays explanation, hate label badge, confidence score
5. Shows 5 retrieved evidence tiles with source citations

How to run:
    streamlit run app.py
Then open browser at:  http://localhost:8501
For GCP deployment:    http://34.10.8.118:8501
"""

import os
import re
import time
import hashlib
import streamlit as st
from pipeline import analyze_meme as run_pipeline

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MemeRAG",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">

<style>
:root {
  --bg-0: #0a0b0f;
  --bg-1: #0d1015;
  --bg-2: #111418;
  --bg-3: #13161c;
  --border-1: #1a1d24;
  --border-2: #1f2937;
  --border-3: #374151;
  --ink-0: #f9fafb;
  --ink-1: #e5e7eb;
  --ink-2: #d1d5db;
  --ink-3: #9ca3af;
  --ink-4: #6b7280;
  --ink-5: #4b5563;
  --indigo: #6366f1;
  --indigo-hover: #4f46e5;
  --indigo-soft: rgba(99, 102, 241, 0.1);
  --indigo-border: rgba(99, 102, 241, 0.3);
  --indigo-text: #a5b4fc;
  --violet: #8b5cf6;
  --violet-text: #c4b5fd;
  --emerald: #10b981;
  --emerald-soft: rgba(16, 185, 129, 0.08);
  --emerald-border: rgba(16, 185, 129, 0.3);
  --rose: #f43f5e;
  --rose-soft: rgba(244, 63, 94, 0.08);
  --rose-border: rgba(244, 63, 94, 0.3);
  --amber: #f59e0b;
}

html, body, [class*="css"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, system-ui, sans-serif !important;
  color: var(--ink-1);
}

.stApp {
  background: var(--bg-0) !important;
  position: relative;
}

.stApp::before {
  content: '';
  position: fixed;
  top: -200px; right: -200px;
  width: 600px; height: 600px;
  background: radial-gradient(circle, rgba(99, 102, 241, 0.08), transparent 60%);
  pointer-events: none; z-index: 0;
  animation: float1 18s ease-in-out infinite;
}
.stApp::after {
  content: '';
  position: fixed;
  bottom: -200px; left: -200px;
  width: 500px; height: 500px;
  background: radial-gradient(circle, rgba(139, 92, 246, 0.06), transparent 60%);
  pointer-events: none; z-index: 0;
  animation: float2 20s ease-in-out infinite;
}
@keyframes float1 { 0%,100% { transform: translate(0, 0); } 50% { transform: translate(-50px, 50px); } }
@keyframes float2 { 0%,100% { transform: translate(0, 0); } 50% { transform: translate(50px, -50px); } }
@keyframes shimmer { 0%   { background-position: -200% 0; } 100% { background-position: 200% 0; } }
@keyframes pulse { 0%,100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.5; transform: scale(0.8); } }
@keyframes slidein { from { opacity: 0; transform: translateY(8px); } to   { opacity: 1; transform: translateY(0); } }

.block-container {
  max-width: 1280px !important;
  padding: 1rem 2rem 1rem !important;
  position: relative; z-index: 1;
}

#MainMenu, footer, header, .stDeployButton { visibility: hidden; display: none; }

/* Trim default Streamlit vertical gaps */
[data-testid="stVerticalBlock"] { gap: 0.5rem !important; }
[data-testid="stHorizontalBlock"] { gap: 0.75rem !important; }

/* top bar */
.top-bar {
  display: flex; justify-content: space-between; align-items: center;
  padding-bottom: 18px; border-bottom: 1px solid var(--border-1);
  margin-bottom: 32px;
}
.brand { display: flex; align-items: center; gap: 12px; }
.brand-mark {
  width: 32px; height: 32px;
  background: linear-gradient(135deg, #6366f1, #8b5cf6);
  border-radius: 8px;
  display: flex; align-items: center; justify-content: center;
  font-weight: 600; font-size: 14px; color: white;
  box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
}
.brand-name { font-size: 15px; font-weight: 600; color: var(--ink-0); letter-spacing: -0.01em; }
.brand-sub  { font-size: 11px; color: var(--ink-4); }

.top-pills { display: flex; gap: 8px; align-items: center; }
.status-pill {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 5px 12px;
  background: var(--emerald-soft);
  border: 1px solid var(--emerald-border);
  border-radius: 6px;
  font-size: 11px; color: var(--emerald); font-weight: 500;
}
.status-dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--emerald);
  animation: pulse 2s ease-in-out infinite;
  box-shadow: 0 0 8px rgba(16, 185, 129, 0.6);
}
.course-pill {
  padding: 5px 12px;
  border: 1px solid var(--border-2);
  border-radius: 6px;
  font-size: 11px; color: var(--ink-3);
}

/* hero */
.hero {
  display: grid;
  grid-template-columns: 1.3fr 1fr;
  gap: 32px;
  margin-bottom: 28px;
  align-items: center;
}
@media (max-width: 900px) { .hero { grid-template-columns: 1fr; } }
.hero-badge {
  display: inline-flex; align-items: center; gap: 8px;
  padding: 4px 12px;
  background: var(--indigo-soft);
  border: 1px solid var(--indigo-border);
  border-radius: 999px;
  margin-bottom: 18px;
  font-size: 11px; color: var(--indigo-text);
  letter-spacing: 0.05em; font-weight: 500;
}
.hero-badge-dot { width: 4px; height: 4px; background: var(--indigo); border-radius: 50%; }
.hero-title {
  font-size: clamp(30px, 4.2vw, 40px);
  font-weight: 600;
  color: var(--ink-0);
  margin: 0 0 14px;
  line-height: 1.1;
  letter-spacing: -0.03em;
}
.hero-title-accent {
  background: linear-gradient(135deg, #6366f1, #8b5cf6, #ec4899);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.hero-sub { font-size: 14px; color: var(--ink-3); max-width: 500px; line-height: 1.6; margin: 0; }

/* perf card — trimmed sizing so labels never wrap */
.perf-card {
  background: linear-gradient(135deg, var(--bg-2), var(--bg-1));
  border: 1px solid var(--border-2);
  border-radius: 14px;
  padding: 16px;
  position: relative;
  overflow: hidden;
}
.perf-card::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.6), transparent);
  background-size: 200% 100%;
  animation: shimmer 3s linear infinite;
}
.perf-header {
  font-size: 10px; color: var(--ink-4);
  letter-spacing: 0.12em; text-transform: uppercase;
  margin-bottom: 12px;
  display: flex; justify-content: space-between; align-items: center;
  font-family: 'JetBrains Mono', monospace;
}
.perf-live { color: var(--emerald); display: flex; align-items: center; gap: 4px; }
.perf-live-dot {
  width: 4px; height: 4px; background: var(--emerald); border-radius: 50%;
  animation: pulse 1.5s ease-in-out infinite;
}
.perf-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.stat-tile {
  padding: 10px 12px;
  background: var(--bg-1);
  border: 1px solid var(--border-1);
  border-radius: 10px;
  transition: all 0.2s ease;
  min-width: 0;
}
.stat-tile:hover { border-color: var(--border-3); background: var(--bg-3); }
.stat-label {
  font-size: 9px; color: var(--ink-4);
  margin-bottom: 3px; letter-spacing: 0.1em;
  font-family: 'JetBrains Mono', monospace;
}
.stat-value {
  font-size: 20px; font-weight: 600; color: var(--ink-0);
  letter-spacing: -0.02em;
  line-height: 1.1;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.stat-note {
  font-size: 10px; margin-top: 2px; color: var(--ink-3);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.stat-note-up { color: var(--emerald); }

/* section labels */
.section-label {
  font-size: 11px; font-weight: 500;
  color: var(--ink-3);
  letter-spacing: 0.08em; text-transform: uppercase;
  display: flex; align-items: center; gap: 8px;
}
.section-icon {
  width: 16px; height: 16px;
  border-radius: 4px;
  display: inline-flex; align-items: center; justify-content: center;
  font-size: 9px;
}
.section-icon-indigo { background: var(--indigo-soft); color: var(--indigo-text); }
.section-icon-violet { background: rgba(139, 92, 246, 0.15); color: var(--violet-text); }

/* cards */
.card {
  background: var(--bg-1);
  border: 1px solid var(--border-1);
  border-radius: 14px;
  padding: 18px;
  margin-bottom: 12px;
  transition: all 0.2s ease;
  position: relative;
  height: 100%;
}
.card:hover { border-color: var(--border-2); }
.card-header {
  display: flex; justify-content: space-between; align-items: center;
  margin-bottom: 12px;
}
.card-meta { font-size: 10px; color: var(--ink-4); font-family: 'JetBrains Mono', monospace; }
.card-verdict { position: relative; overflow: hidden; animation: slidein 0.4s ease-out 0.1s both; }
.card-verdict::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, var(--emerald), var(--indigo));
}
.card-verdict-hateful::before { background: linear-gradient(90deg, var(--rose), var(--amber)) !important; }

/* textarea */
.stTextArea textarea {
  font-family: 'Inter', sans-serif !important;
  font-size: 14px !important;
  color: var(--ink-1) !important;
  background: var(--bg-0) !important;
  border: 1px solid var(--border-3) !important;
  border-radius: 10px !important;
  padding: 12px 14px !important;
  box-shadow: none !important;
  transition: all 0.2s !important;
  caret-color: var(--indigo) !important;
}
.stTextArea textarea::placeholder { color: var(--ink-5) !important; }
.stTextArea textarea:focus {
  border-color: var(--indigo) !important;
  box-shadow: 0 0 0 3px var(--indigo-soft) !important;
  background: var(--bg-1) !important;
}
.stTextArea label { display: none !important; }

/* default streamlit buttons = preset chips — full-width, same size */
.stButton > button {
  font-family: 'Inter', sans-serif !important;
  font-weight: 500 !important;
  font-size: 12px !important;
  background: transparent !important;
  color: var(--ink-3) !important;
  border: 1px solid var(--border-2) !important;
  border-radius: 8px !important;
  padding: 8px 12px !important;
  transition: all 0.15s ease !important;
  box-shadow: none !important;
  height: 36px !important;
}
.stButton > button:hover {
  border-color: var(--border-3) !important;
  color: var(--ink-1) !important;
  background: var(--bg-2) !important;
}
.stButton > button:active { transform: scale(0.98) !important; }
.stButton > button:focus { outline: none !important; }

/* primary analyze button */
.analyze-btn .stButton > button {
  width: 100% !important;
  background: var(--indigo) !important;
  color: white !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  letter-spacing: 0.01em !important;
  padding: 10px 20px !important;
  border-radius: 10px !important;
  border: none !important;
  height: 42px !important;
  box-shadow: 0 4px 12px rgba(99, 102, 241, 0.25) !important;
  transition: all 0.15s ease !important;
}
.analyze-btn .stButton > button:hover {
  background: var(--indigo-hover) !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 8px 24px rgba(99, 102, 241, 0.35) !important;
  color: white !important;
}
.analyze-btn .stButton > button:active { transform: translateY(0) !important; }

/* thinking bar */
.thinking-card {
  background: var(--bg-1);
  border: 1px solid var(--indigo-border);
  border-radius: 14px;
  padding: 18px;
  margin-bottom: 12px;
  animation: slidein 0.3s ease-out;
  position: relative; overflow: hidden;
}
.thinking-card::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent, var(--indigo), transparent);
  background-size: 200% 100%;
  animation: shimmer 1.5s linear infinite;
}
.thinking-title {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px; font-weight: 600;
  letter-spacing: 0.15em; text-transform: uppercase;
  color: var(--indigo-text);
  margin-bottom: 14px;
  display: flex; align-items: center; gap: 10px;
}
.thinking-dot {
  width: 8px; height: 8px;
  border-radius: 50%;
  background: var(--indigo);
  box-shadow: 0 0 8px var(--indigo);
  animation: pulse 1s ease-in-out infinite;
}
.stages { display: flex; flex-direction: column; gap: 8px; }
.stage { display: flex; align-items: center; gap: 12px; }
.stage-icon { font-size: 13px; width: 20px; text-align: center; }
.stage-done .stage-label   { font-size: 12px; font-weight: 500; color: var(--emerald); flex: 1; }
.stage-active .stage-label { font-size: 12px; font-weight: 500; color: var(--indigo-text); flex: 1; }
.stage-wait .stage-label   { font-size: 12px; font-weight: 500; color: var(--ink-5); flex: 1; }
.stage-check { font-size: 12px; margin-left: auto; }

.prog-track {
  height: 4px;
  background: var(--border-1);
  border-radius: 999px;
  overflow: hidden;
  margin-top: 14px;
}
.prog-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--indigo), var(--violet));
  border-radius: 999px;
  transition: width 0.4s ease;
}
.prog-footer {
  display: flex; justify-content: space-between;
  margin-top: 6px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px; font-weight: 500;
  color: var(--ink-4);
}
.prog-footer span:last-child { color: var(--indigo-text); }

/* verdict badge */
.verdict-badge {
  display: inline-flex; align-items: center; gap: 10px;
  padding: 7px 14px;
  border-radius: 999px;
  margin-bottom: 14px;
  font-size: 13px; font-weight: 500;
  letter-spacing: 0.01em;
  animation: slidein 0.4s ease-out;
}
.verdict-dot { width: 8px; height: 8px; border-radius: 50%; }
.badge-safe {
  background: var(--emerald-soft);
  border: 1px solid var(--emerald-border);
  color: var(--emerald);
}
.badge-safe .verdict-dot { background: var(--emerald); box-shadow: 0 0 10px rgba(16, 185, 129, 0.6); }
.badge-hateful {
  background: var(--rose-soft);
  border: 1px solid var(--rose-border);
  color: var(--rose);
}
.badge-hateful .verdict-dot { background: var(--rose); box-shadow: 0 0 10px rgba(244, 63, 94, 0.6); }

.conf-wrap {
  border-top: 1px solid var(--border-1);
  padding-top: 12px;
  margin-top: 14px;
}
.conf-header { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px; }
.conf-label { font-size: 11px; color: var(--ink-3); letter-spacing: 0.05em; }
.conf-value { font-size: 17px; font-weight: 600; color: var(--ink-0); letter-spacing: -0.02em; }
.conf-value-suffix { font-size: 11px; color: var(--ink-3); font-weight: 400; }
.conf-track { height: 5px; background: var(--border-1); border-radius: 999px; overflow: hidden; position: relative; }
.conf-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--indigo), var(--emerald));
  border-radius: 999px;
  position: relative;
  animation: fillin 1s cubic-bezier(0.34, 1.56, 0.64, 1);
}
.conf-fill::after {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; bottom: 0;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
  background-size: 200% 100%;
  animation: shimmer 2s linear infinite;
}
.conf-fill-hateful { background: linear-gradient(90deg, var(--rose), var(--amber)) !important; }
@keyframes fillin { from { width: 0 !important; } }

.card-body { font-size: 13px; line-height: 1.7; color: var(--ink-2); margin: 0; }

.empty-state {
  min-height: 240px;
  display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  text-align: center; gap: 12px;
  border: 1px dashed var(--border-2);
  border-radius: 14px;
  background: var(--bg-1);
  padding: 32px 20px;
}
.empty-glyph {
  width: 48px; height: 48px;
  border-radius: 10px;
  background: var(--indigo-soft);
  border: 1px solid var(--indigo-border);
  display: flex; align-items: center; justify-content: center;
  color: var(--indigo-text);
  font-size: 18px;
}
.empty-title { font-size: 15px; font-weight: 500; color: var(--ink-1); }
.empty-sub { font-size: 12px; color: var(--ink-3); max-width: 340px; line-height: 1.6; }

/* evidence gallery */
.evidence-header {
  display: flex; justify-content: space-between; align-items: center;
  margin: 20px 0 12px;
}
.evidence-meta { font-size: 11px; color: var(--ink-4); }

.ev-media {
  height: 80px; border-radius: 8px;
  margin-bottom: 10px;
  position: relative; overflow: hidden;
  border: 1px solid var(--border-1);
  display: flex; align-items: center; justify-content: center;
}
.ev-media-gen {
  color: rgba(255,255,255,0.9);
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px; font-weight: 600;
  letter-spacing: 0.1em;
}
.ev-media-gen::before {
  content: '';
  position: absolute; inset: 0;
  background: radial-gradient(circle at 30% 30%, rgba(255,255,255,0.18), transparent 55%);
}
.ev-media-twitter {
  background: linear-gradient(135deg, #1e293b, #334155);
  flex-direction: column; gap: 3px;
  color: var(--ink-3);
}

.evidence-tile {
  background: var(--bg-1);
  border: 1px solid var(--border-1);
  border-radius: 12px;
  padding: 10px;
  transition: all 0.2s ease;
  animation: slidein 0.4s ease-out both;
  display: flex;
  flex-direction: column;
  height: 100%;
}
.evidence-tile:hover { border-color: var(--border-3); transform: translateY(-2px); }

.ev-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
.ev-dataset { font-family: 'JetBrains Mono', monospace; font-size: 9px; font-weight: 600; letter-spacing: 0.1em; }
.ev-dataset-fb { color: var(--indigo); }
.ev-dataset-tw { color: var(--violet); }
.ev-tag { font-size: 9px; font-weight: 600; letter-spacing: 0.05em; padding: 2px 6px; border-radius: 4px; }
.ev-tag-safe { color: var(--emerald); background: var(--emerald-soft); }
.ev-tag-hateful { color: var(--rose); background: var(--rose-soft); }

.ev-text {
  font-size: 11px; color: var(--ink-3);
  line-height: 1.5; margin-bottom: 8px;
  min-height: 44px;
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
.ev-footer {
  display: flex; justify-content: space-between; align-items: center;
  padding-top: 6px; border-top: 1px solid var(--border-1);
  margin-top: auto;
}
.ev-dist { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: var(--ink-4); }
.ev-rank { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: var(--ink-4); }

.ev-citation {
  font-family: 'JetBrains Mono', monospace;
  font-size: 9px;
  color: var(--indigo-text);
  background: rgba(99, 102, 241, 0.06);
  border: 1px solid rgba(99, 102, 241, 0.15);
  border-radius: 4px;
  padding: 4px 6px;
  margin-top: 6px;
  word-break: break-all;
  line-height: 1.3;
  letter-spacing: 0.02em;
}

/* source meme placeholder — fixed centering */
.source-placeholder {
  height: 180px;
  border-radius: 10px;
  border: 1px solid var(--border-1);
  display: flex; align-items: center; justify-content: center;
  flex-direction: column; gap: 10px;
  padding: 20px; text-align: center;
  position: relative; overflow: hidden;
}
.source-placeholder::before {
  content: '';
  position: absolute; inset: 0;
  background: radial-gradient(circle at 30% 30%, rgba(255,255,255,0.15), transparent 60%);
}
.source-placeholder-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  letter-spacing: 0.15em;
  color: rgba(255,255,255,0.75);
  text-transform: uppercase;
  position: relative; z-index: 1;
}
.source-placeholder-text {
  font-size: 13px;
  color: rgba(255,255,255,0.95);
  font-style: italic;
  max-width: 85%;
  line-height: 1.5;
  position: relative; z-index: 1;
}

.stImage > img { border-radius: 8px !important; border: 1px solid var(--border-1) !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
PRESETS = [
    "when you already started eating and someone says 'lets pray'",
    "need a hug ? i love hugs",
    "when people talk trash but you're a bigger person",
    "and if anything happens, the women are to blame that's right, it's their fault, definitely",
]
PRESET_LABELS = ["Let's pray", "Need a hug", "Bigger person", "Hate test"]

DEMO_IMAGES_DIR = "demo_images"

STAGES = [
    ("◈", "Embedding input..."),
    ("◇", "Retrieving similar memes..."),
    ("◆", "Querying Llama 3..."),
    ("◉", "Generating results..."),
]

PLACEHOLDER_GRADIENTS = [
    ("#4c1d95", "#6366f1"),
    ("#1e40af", "#6366f1"),
    ("#065f46", "#10b981"),
    ("#7c2d12", "#f59e0b"),
    ("#831843", "#ec4899"),
    ("#1e293b", "#475569"),
    ("#581c87", "#a855f7"),
    ("#134e4a", "#14b8a6"),
]


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def gradient_for_id(meme_id: str):
    if not meme_id:
        meme_id = "default"
    h = int(hashlib.md5(str(meme_id).encode()).hexdigest(), 16)
    return PLACEHOLDER_GRADIENTS[h % len(PLACEHOLDER_GRADIENTS)]


def thinking_bar_html(active_step: int) -> str:
    pct = int((active_step / len(STAGES)) * 100)
    stage_rows = ""
    for i, (icon, label) in enumerate(STAGES):
        if i < active_step:
            cls, check = "stage-done", "✓"
        elif i == active_step:
            cls, check = "stage-active", '<span style="color:var(--indigo-text);animation:pulse 1s infinite">◐</span>'
        else:
            cls, check = "stage-wait", ""
        stage_rows += f"""
        <div class="stage {cls}">
          <div class="stage-icon">{icon}</div>
          <div class="stage-label">{label}</div>
          <div class="stage-check">{check}</div>
        </div>"""
    step_label = f"Step {active_step + 1} / {len(STAGES)}"
    foot_right = "processing..." if active_step < len(STAGES) - 1 else "finalizing..."
    return f"""
    <div class="thinking-card">
      <div class="thinking-title">
        <div class="thinking-dot"></div> Llama 3 is analyzing
      </div>
      <div class="stages">{stage_rows}</div>
      <div class="prog-track">
        <div class="prog-fill" style="width:{pct}%"></div>
      </div>
      <div class="prog-footer"><span>{step_label}</span><span>{foot_right}</span></div>
    </div>"""


def try_load_image(meme_id: str):
    if not meme_id:
        return None
    clean_id = re.sub(r'[^0-9]', '', str(meme_id))
    if not clean_id:
        return None
    for candidate in [clean_id, clean_id.zfill(5)]:
        for ext in ("png", "jpg", "jpeg", "webp"):
            path = os.path.join(DEMO_IMAGES_DIR, f"{candidate}.{ext}")
            if os.path.exists(path):
                return path
    return None


def extract_id_from_source_url(source_url: str) -> str:
    if not source_url:
        return ""
    try:
        return source_url.split("id=")[1]
    except (IndexError, AttributeError):
        return ""


def html_escape(s):
    return (str(s).replace("&", "&amp;")
                  .replace("<", "&lt;")
                  .replace(">", "&gt;"))


def render_source_placeholder(meme_id: str, text: str, dataset: str = "facebook"):
    """Render a vertically-centered source meme placeholder."""
    c1, c2 = gradient_for_id(meme_id)
    preview_text = (text[:100] + "…") if len(text) > 100 else text
    label = "twitter · text-only" if dataset == "twitter" else f"text-only · id {meme_id or 'N/A'}"
    prefix = '<span style="font-size: 22px; color: rgba(255,255,255,0.9); margin-bottom: 4px;">𝕏</span>' if dataset == "twitter" else ""

    st.markdown(f"""
    <div class="source-placeholder" style="background: linear-gradient(135deg, {c1}, {c2});">
      {prefix}
      <span class="source-placeholder-label">{label}</span>
      <span class="source-placeholder-text">"{html_escape(preview_text)}"</span>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────────────────────────────────────
if "meme_text" not in st.session_state:
    st.session_state["meme_text"] = ""
if "result" not in st.session_state:
    st.session_state.result = None
if "run_effects" not in st.session_state:
    st.session_state.run_effects = False


def apply_preset(text: str):
    st.session_state["meme_text"] = text
    st.session_state.result = None


# ──────────────────────────────────────────────────────────────────────────────
# TOP BAR
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-bar">
  <div class="brand">
    <div class="brand-mark">M</div>
    <div>
      <div class="brand-name">MemeRAG</div>
      <div class="brand-sub">Meme understanding & hate detection</div>
    </div>
  </div>
  <div class="top-pills">
    <div class="status-pill">
      <div class="status-dot"></div>
      Llama 3 · online
    </div>
    <div class="course-pill">CS 6120 · Spring 2026</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# HERO SECTION
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div>
    <div class="hero-badge">
      <span class="hero-badge-dot"></span>
      Retrieval-Augmented Generation
    </div>
    <h1 class="hero-title">
      Understand memes. <span class="hero-title-accent">Detect hate.</span>
    </h1>
    <p class="hero-sub">
      Retrieves the 5 most semantically similar memes from 33,000 labeled
      entries, then uses Llama 3 to explain meaning and classify content
      with verifiable cited evidence.
    </p>
  </div>
  <div class="perf-card">
    <div class="perf-header">
      <span>Performance</span>
      <span class="perf-live"><span class="perf-live-dot"></span>LIVE</span>
    </div>
    <div class="perf-grid">
      <div class="stat-tile">
        <div class="stat-label">CORPUS</div>
        <div class="stat-value">33,000</div>
        <div class="stat-note stat-note-up">indexed</div>
      </div>
      <div class="stat-tile">
        <div class="stat-label">F1 SCORE</div>
        <div class="stat-value">0.73</div>
        <div class="stat-note">macro avg</div>
      </div>
      <div class="stat-tile">
        <div class="stat-label">LATENCY</div>
        <div class="stat-value">~2s</div>
        <div class="stat-note">per query</div>
      </div>
      <div class="stat-tile">
        <div class="stat-label">MODEL</div>
        <div class="stat-value">Llama 3</div>
        <div class="stat-note">8B · local</div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# INPUT CARD — textarea, presets (full-width row), analyze button
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="card" style="padding: 20px;">
  <div class="card-header">
    <div class="section-label">
      <span class="section-icon section-icon-indigo">✎</span>
      Meme text
    </div>
    <div class="card-meta">Paste a caption or try a preset</div>
  </div>
""", unsafe_allow_html=True)

meme_input = st.text_area(
    label="meme_text_area",
    placeholder="Paste or type meme text here…",
    height=90,
    key="meme_text",
    label_visibility="collapsed",
)

# Preset chips row — stretched via use_container_width
pcols = st.columns(4)
for i, (short_lbl, full_text) in enumerate(zip(PRESET_LABELS, PRESETS)):
    with pcols[i]:
        st.button(
            short_lbl,
            key=f"preset_{i}",
            use_container_width=True,
            on_click=apply_preset,
            args=(full_text,),
        )

# Embedded-via note + analyze button on SAME row
footer_left, footer_right = st.columns([1.3, 1])
with footer_left:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:8px; height:42px;
                font-size:11px; color:var(--ink-4); padding-top:4px;">
      <span style="color:var(--indigo); font-size:10px;">◆</span>
      <span>Embedded via all-MiniLM-L6-v2</span>
    </div>
    """, unsafe_allow_html=True)

with footer_right:
    st.markdown('<div class="analyze-btn">', unsafe_allow_html=True)
    analyze_clicked = st.button(
        "Analyze meme  →",
        use_container_width=True,
        key="analyze_btn",
    )
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # /card

# ──────────────────────────────────────────────────────────────────────────────
# PIPELINE CALL
# ──────────────────────────────────────────────────────────────────────────────
if analyze_clicked and meme_input.strip():
    st.session_state.result = None
    st.session_state.run_effects = True

    bar_slot = st.empty()
    bar_slot.markdown(thinking_bar_html(0), unsafe_allow_html=True)
    time.sleep(0.6)
    bar_slot.markdown(thinking_bar_html(1), unsafe_allow_html=True)
    time.sleep(0.5)
    bar_slot.markdown(thinking_bar_html(2), unsafe_allow_html=True)

    try:
        result = run_pipeline(meme_input.strip())
        st.session_state.result = result
    except Exception as e:
        bar_slot.empty()
        st.session_state.result = {
            "explanation": f"Could not reach Llama 3: {e}",
            "hate_label": "uncertain",
            "reasoning": "Check that the GCP VM is running and Ollama is serving.",
            "confidence": 0.0,
            "id": "",
            "citations": [],
        }

    bar_slot.markdown(thinking_bar_html(3), unsafe_allow_html=True)
    time.sleep(0.4)
    bar_slot.empty()

# ──────────────────────────────────────────────────────────────────────────────
# RESULTS — 3 equal-height columns
# Source meme (left) | Classification (middle) | Explanation (right)
# ──────────────────────────────────────────────────────────────────────────────
if not st.session_state.result:
    st.markdown("""
    <div class="empty-state">
      <div class="empty-glyph">◆</div>
      <div class="empty-title">Enter a meme and click Analyze</div>
      <div class="empty-sub">
        Results will show the retrieved source meme, an explanation of its
        meaning, a classification verdict, and the top 5 evidence citations
        from ChromaDB.
      </div>
    </div>""", unsafe_allow_html=True)
else:
    r = st.session_state.result
    citations_list = r.get("citations", [])
    top_dataset = citations_list[0].get("dataset", "facebook") if citations_list else "facebook"
    top_source_url = citations_list[0].get("source_url", "") if citations_list else ""
    top_text = citations_list[0].get("text", "") if citations_list else ""
    img_id = extract_id_from_source_url(top_source_url)
    img_path = try_load_image(img_id) if top_dataset == "facebook" else None

    # Verdict values
    label_raw = str(r.get("hate_label", "not hateful")).lower()
    is_hateful = ("hate" in label_raw) and ("not" not in label_raw)
    badge_cls = "badge-hateful" if is_hateful else "badge-safe"
    badge_text = "Hateful" if is_hateful else "Not hateful"
    verdict_card_cls = "card-verdict card-verdict-hateful" if is_hateful else "card-verdict"
    conf_fill_cls = "conf-fill-hateful" if is_hateful else ""
    reasoning = r.get("rationale", r.get("reasoning", "No reasoning returned."))

    raw_conf = r.get("confidence", 0.75)
    try:
        raw_conf = float(raw_conf)
        confidence = int(raw_conf * 100) if raw_conf <= 1.0 else int(raw_conf)
    except (TypeError, ValueError):
        confidence = 75
    confidence = max(0, min(100, confidence))

    explanation = r.get("explanation", "No explanation returned.")

    # ── 3-column results row ──
    col_source, col_verdict, col_explain = st.columns([1, 1, 1.1], gap="small")

    with col_source:
        st.markdown(f"""
        <div class="card">
          <div class="card-header">
            <div class="section-label">
              <span class="section-icon section-icon-indigo">◆</span>
              Source meme
            </div>
            <div class="card-meta">{html_escape(top_dataset)} · id {html_escape(img_id or 'N/A')}</div>
          </div>
        """, unsafe_allow_html=True)
        if img_path:
            st.image(img_path, use_container_width=True)
        else:
            render_source_placeholder(img_id, top_text, top_dataset)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_verdict:
        st.markdown(f"""
        <div class="card {verdict_card_cls}">
          <div class="card-header">
            <div class="section-label">
              <span class="section-icon section-icon-indigo">◎</span>
              Classification
            </div>
            <div class="card-meta">Llama 3 · 8B</div>
          </div>
          <div class="verdict-badge {badge_cls}">
            <div class="verdict-dot"></div>{badge_text}
          </div>
          <p class="card-body" style="font-size:12.5px; margin-bottom: 10px;">
            {html_escape(reasoning)}
          </p>
          <div class="conf-wrap">
            <div class="conf-header">
              <span class="conf-label">Confidence</span>
              <span class="conf-value">{confidence}<span class="conf-value-suffix">%</span></span>
            </div>
            <div class="conf-track">
              <div class="conf-fill {conf_fill_cls}" style="width:{confidence}%;"></div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

    with col_explain:
        st.markdown(f"""
        <div class="card">
          <div class="card-header">
            <div class="section-label">
              <span class="section-icon section-icon-indigo">✦</span>
              Explanation
            </div>
            <div class="card-meta">Generated by Llama 3</div>
          </div>
          <p class="card-body">{html_escape(explanation)}</p>
        </div>""", unsafe_allow_html=True)

    # Celebratory effects
    if st.session_state.get("run_effects"):
        if is_hateful:
            st.snow()
        else:
            st.balloons()
        st.session_state.run_effects = False

    # ──────────────────────────────────────────────────────────────────────────
    # EVIDENCE GALLERY — 5 aligned tiles
    # ──────────────────────────────────────────────────────────────────────────
    sources = citations_list
    if sources:
        st.markdown(f"""
        <div class="evidence-header">
          <div class="section-label">
            <span class="section-icon section-icon-violet">◈</span>
            Retrieved evidence
          </div>
          <div class="evidence-meta">
            {len(sources[:5])} nearest neighbors · ChromaDB
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Evidence tile grid
        img_cols = st.columns(len(sources[:5]), gap="small")
        for idx, (col, s) in enumerate(zip(img_cols, sources[:5])):
            with col:
                ev_url = s.get("source_url", "#")
                ev_dataset = s.get("dataset", "facebook")
                ev_text = s.get("text", "—")
                ev_dist = s.get("distance", "N/A")
                ev_id = extract_id_from_source_url(ev_url)
                ev_img = try_load_image(ev_id) if ev_dataset == "facebook" else None

                label_raw = s.get("label_str", s.get("label", 0))
                try:
                    is_ev_hate = int(label_raw) == 1
                except (ValueError, TypeError):
                    is_ev_hate = ("hate" in str(label_raw).lower()
                                  and "not" not in str(label_raw).lower())

                tag_cls = "ev-tag-hateful" if is_ev_hate else "ev-tag-safe"
                tag_text = "hateful" if is_ev_hate else "safe"
                dataset_upper = ev_dataset.upper()
                dataset_cls = "ev-dataset-tw" if ev_dataset == "twitter" else "ev-dataset-fb"
                ev_text_trunc = ev_text[:90] + ("…" if len(ev_text) > 90 else "")

                rank_label = f"#{idx + 1}"

                # Build tile HTML
                if ev_img:
                    # Real image case — render tile header separately around st.image
                    st.markdown('<div class="evidence-tile">', unsafe_allow_html=True)
                    st.image(ev_img, use_container_width=True)
                    st.markdown(f"""
                      <div class="ev-header" style="margin-top: 10px;">
                        <span class="ev-dataset {dataset_cls}">{dataset_upper}</span>
                        <span class="ev-tag {tag_cls}">{tag_text}</span>
                      </div>
                      <div class="ev-text">"{html_escape(ev_text_trunc)}"</div>
                      <div class="ev-footer">
                        <span class="ev-dist">dist: {html_escape(ev_dist)}</span>
                        <span class="ev-rank">{rank_label}</span>
                      </div>
                      <div class="ev-citation" title="{html_escape(ev_url)}">{html_escape(ev_url)}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Styled placeholder
                    if ev_dataset == "twitter":
                        media_html = f"""
                        <div class="ev-media ev-media-twitter">
                          <span style="font-size:16px;">𝕏</span>
                          <span style="font-family:'JetBrains Mono',monospace; font-size:9px;
                                       letter-spacing:0.1em; color:var(--ink-4);">Twitter</span>
                          <span style="font-family:'JetBrains Mono',monospace; font-size:9px;
                                       color:var(--ink-5);">id · {html_escape(ev_id)}</span>
                        </div>"""
                    else:
                        c1, c2 = gradient_for_id(ev_id)
                        media_html = f"""
                        <div class="ev-media ev-media-gen" style="
                            background: linear-gradient(135deg, {c1}, {c2});">
                          id · {html_escape(ev_id or 'N/A')}
                        </div>"""

                    st.markdown(f"""
                    <div class="evidence-tile">
                      {media_html}
                      <div class="ev-header">
                        <span class="ev-dataset {dataset_cls}">{dataset_upper}</span>
                        <span class="ev-tag {tag_cls}">{tag_text}</span>
                      </div>
                      <div class="ev-text">"{html_escape(ev_text_trunc)}"</div>
                      <div class="ev-footer">
                        <span class="ev-dist">dist: {html_escape(ev_dist)}</span>
                        <span class="ev-rank">{rank_label}</span>
                      </div>
                      <div class="ev-citation" title="{html_escape(ev_url)}">{html_escape(ev_url)}</div>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="card" style="margin-top: 20px;">
          <div class="section-label">
            <span class="section-icon section-icon-violet">◈</span>
            Retrieved evidence
          </div>
          <p class="card-body" style="margin-top: 12px; color: var(--ink-4);">
            No retrieved sources returned from the pipeline.
          </p>
        </div>""", unsafe_allow_html=True)
