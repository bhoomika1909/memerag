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
import os
import re
import time
import streamlit as st
from pipeline import analyze_meme as run_pipeline


# PAGE CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MemeRAG",
    page_icon="😂",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# FONTS + GLOBAL CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<link rel="stylesheet"
      href="https://cdn.jsdelivr.net/npm/@fontsource/fredoka@5/700.min.css">
<link rel="stylesheet"
      href="https://cdn.jsdelivr.net/npm/@fontsource/nunito@5/400.min.css">
<link rel="stylesheet"
      href="https://cdn.jsdelivr.net/npm/@fontsource/nunito@5/700.min.css">

<style>
:root {
    --cream:#FFFBF0; --charcoal:#1a1a1a;
    --mint:#B8F0D8;  --mint-deep:#0A5E3F;
    --peach:#FFB5A7; --peach-deep:#7B1200;
    --sky:#C8E6FF;   --sky-deep:#0A3A6E;
    --lav:#E8D8FF;   --lav-deep:#3A0A6E;
    --amber:#FFE4A0; --amber-deep:#6B4000;
    --border:3px solid #1a1a1a;
    --border-thin:2px solid #1a1a1a;
    --radius:14px; --radius-sm:8px;
    --shadow:4px 4px 0 #1a1a1a;
    --shadow-sm:2px 2px 0 #1a1a1a;
}
html, body, [class*="css"] { font-family:'Nunito',sans-serif !important; }
.stApp {
    background-color:var(--cream) !important;
    background-image:radial-gradient(circle,rgba(26,26,26,.04) 1px,transparent 1px);
    background-size:28px 28px;
}
.block-container { max-width:1200px !important; padding:1.5rem 2rem 4rem !important; }
.stApp::before {
    content:"😂 💀 🔥 👀 😭 💯 🤡 🫠 😤 🧠 💅 😈 🤯 😑 🙃 🤌 🫣 😩 🤣 😔";
    position:fixed; top:0; left:0; right:0; bottom:0;
    font-size:26px; line-height:3.6; letter-spacing:2.2rem;
    opacity:.045; pointer-events:none; z-index:0;
    word-break:break-all; overflow:hidden; padding:2rem;
    transform:rotate(-6deg) scale(1.15);
}
#MainMenu, footer, header { visibility:hidden; }
.stDeployButton { display:none; }

/* ── navbar ── */
.meme-nav {
    background:linear-gradient(110deg,#1a1a2e 0%,#2d1b4e 30%,#1a2e2e 60%,#2e1a1a 100%);
    border-radius:var(--radius); padding:.85rem 1.4rem; margin-bottom:1.5rem;
    border:var(--border); display:flex; align-items:center; justify-content:space-between;
    position:relative; overflow:hidden; box-shadow:var(--shadow);
}
.meme-nav::before {
    content:''; position:absolute; top:0; left:0; right:0; height:1px;
    background:linear-gradient(90deg,transparent,rgba(255,209,102,.5),
        rgba(126,207,179,.5),rgba(255,181,200,.5),transparent);
}
.logo-cluster { display:flex; align-items:center; gap:14px; }
.emoji-col    { display:flex; flex-direction:column; gap:3px; font-size:15px; line-height:1.4; }
.logo-text    { display:flex; flex-direction:column; gap:3px; }
.logo-wordmark {
    font-family:'Fredoka',sans-serif;
    font-size:26px; font-weight:700; letter-spacing:.5px; line-height:1;
}
.c1{color:#FFD166}.c2{color:#FF8FAB}.c3{color:#7ECFB3}.c4{color:#FF9F5A}
.csep{color:rgba(255,255,255,.2);margin:0 3px}
.c5{color:#C5A3FF}.c6{color:#7DD9F7}.c7{color:#FFD166}
.logo-tagline {
    font-size:10px; font-weight:700; color:rgba(255,255,255,.4);
    letter-spacing:.08em; text-transform:uppercase;
}
.nav-pills { display:flex; gap:7px; flex-wrap:wrap; }
.nav-pill {
    font-size:11px; font-weight:700; padding:4px 12px;
    border-radius:20px; border:var(--border-thin); font-family:'Nunito',sans-serif;
}
.pill-a{background:#FFD166;color:#1a1a1a}
.pill-b{background:#7ECFB3;color:#1a1a1a}
.pill-c{background:#FFB5C8;color:#1a1a1a}

/* ── section label ── */
.section-label {
    font-family:'Fredoka',sans-serif; font-size:12px; font-weight:700;
    letter-spacing:.08em; text-transform:uppercase; color:#1a1a1a; margin-bottom:8px;
}

/* ── cards ── */
.card {
    border:var(--border); border-radius:var(--radius);
    padding:1.1rem 1.2rem; margin-bottom:12px;
    backdrop-filter:blur(6px); box-shadow:var(--shadow);
}
.card-explain  { background:rgba(200,220,255,.42); }
.card-verdict  { background:rgba(184,240,216,.42); }
.card-hateful  { background:rgba(255,181,167,.42); }
.card-empty    { background:rgba(255,255,255,.35); }

/* ── textarea ── */
.stTextArea textarea {
    font-family:'Nunito',sans-serif !important; font-size:14px !important;
    color:#1a1a1a !important; background:rgba(255,251,240,.92) !important;
    border:var(--border) !important; border-radius:var(--radius-sm) !important;
    padding:10px 14px !important; box-shadow:none !important; resize:vertical !important;
}
.stTextArea textarea:focus {
    border-color:#FF8C42 !important; box-shadow:3px 3px 0 #FF8C42 !important;
}
.stTextArea label { display:none !important; }

/* ── buttons ── */
.stButton > button {
    font-family:'Nunito',sans-serif !important; font-weight:700 !important;
    border:var(--border-thin) !important; border-radius:20px !important;
    transition:transform .1s, box-shadow .1s !important;
    box-shadow:var(--shadow-sm) !important;
}
.stButton > button:hover  { transform:translate(-1px,-1px) !important; box-shadow:3px 3px 0 #1a1a1a !important; }
.stButton > button:active { transform:translate(2px,2px) !important; box-shadow:none !important; }
.stButton > button:focus  { outline:none !important; box-shadow:var(--shadow-sm) !important; }

[data-testid="column"]:nth-child(1) .stButton > button
    { background:rgba(255,209,102,.8)!important; color:#1a1a1a!important; }
[data-testid="column"]:nth-child(2) .stButton > button
    { background:rgba(255,181,200,.8)!important; color:#1a1a1a!important; }
[data-testid="column"]:nth-child(3) .stButton > button
    { background:rgba(126,207,179,.8)!important; color:#1a1a1a!important; }
[data-testid="column"]:nth-child(4) .stButton > button
    { background:rgba(197,163,255,.8)!important; color:#1a1a1a!important; }

/* ── analyze button ── */
.analyze-btn .stButton > button {
    width:100% !important; background:#FF6B9D !important; color:#fff !important;
    font-family:'Fredoka',sans-serif !important; font-size:17px !important;
    letter-spacing:.5px !important; padding:12px !important;
    border-radius:var(--radius-sm) !important; border:var(--border) !important;
    box-shadow:4px 4px 0 #1a1a1a !important;
}
.analyze-btn .stButton > button:hover  { background:#e85d8d !important; transform:translate(-1px,-1px) !important; box-shadow:5px 5px 0 #1a1a1a !important; }
.analyze-btn .stButton > button:active { transform:translate(3px,3px) !important; box-shadow:1px 1px 0 #1a1a1a !important; }

/* ── thinking bar ── */
.thinking-card {
    background:rgba(26,26,26,.9); border:var(--border);
    border-radius:var(--radius); padding:1.1rem 1.2rem;
    margin-bottom:12px; box-shadow:var(--shadow);
}
.thinking-title {
    font-family:'Fredoka',sans-serif; font-size:13px; font-weight:700;
    color:#FFD166; margin-bottom:12px; display:flex; align-items:center; gap:8px;
}
.thinking-dot {
    width:8px; height:8px; border-radius:50%; background:#FF6B9D;
    display:inline-block;
    animation:tpulse 1s infinite;
}
@keyframes tpulse {
    0%,100%{opacity:1;transform:scale(1)}
    50%{opacity:.35;transform:scale(.65)}
}
.stages { display:flex; flex-direction:column; gap:8px; }
.stage  { display:flex; align-items:center; gap:10px; }
.stage-icon { font-size:15px; width:22px; text-align:center; }
.stage-done   .stage-label { font-size:12px; font-weight:700; color:#7ECFB3; flex:1; }
.stage-active .stage-label { font-size:12px; font-weight:700; color:#FFD166; flex:1; }
.stage-wait   .stage-label { font-size:12px; font-weight:700; color:rgba(255,255,255,.22); flex:1; }
.stage-check { font-size:13px; margin-left:auto; }
.prog-track {
    height:8px; background:rgba(255,255,255,.08);
    border:1.5px solid rgba(255,255,255,.15);
    border-radius:20px; overflow:hidden; margin-top:12px;
}
.prog-fill {
    height:100%; border-radius:20px;
    background:linear-gradient(90deg,#7ECFB3,#FFD166);
    transition:width .4s ease;
}
.prog-footer {
    display:flex; justify-content:space-between; margin-top:4px;
    font-size:10px; font-weight:700; color:rgba(255,255,255,.35);
}
.prog-footer span:last-child { color:#FFD166; }

/* ── toast ── */
.toast {
    display:inline-flex; align-items:center; gap:8px;
    background:#1a1a1a; border:2.5px solid #7ECFB3;
    border-radius:12px; padding:8px 14px;
    box-shadow:var(--shadow); margin-bottom:10px;
    animation:tslide .45s ease-out;
}
.toast-hateful { border-color:#FFB5A7 !important; }
@keyframes tslide {
    from{opacity:0;transform:translateY(-10px)}
    to{opacity:1;transform:translateY(0)}
}
.toast-text { font-family:'Fredoka',sans-serif; font-size:13px; font-weight:700; color:#7ECFB3; }
.toast-text-hateful { color:#FFB5A7 !important; }
.toast-sub  { font-size:10px; color:rgba(255,255,255,.45); font-weight:700; }

/* ── verdict badge ── */
.verdict-badge {
    display:inline-flex; align-items:center; gap:8px;
    padding:7px 16px; border-radius:20px; border:var(--border);
    margin-bottom:10px; box-shadow:var(--shadow-sm);
    font-family:'Fredoka',sans-serif; font-size:15px; font-weight:700; color:#1a1a1a;
}
.verdict-dot { width:10px; height:10px; border-radius:50%; background:#1a1a1a; flex-shrink:0; }
.badge-safe    { background:var(--mint); }
.badge-hateful { background:var(--peach); }

/* ── confidence bar ── */
.conf-wrap   { margin-top:12px; }
.conf-header {
    display:flex; justify-content:space-between;
    font-size:11px; font-weight:700; color:var(--mint-deep); margin-bottom:5px;
}
.conf-track {
    height:9px; background:rgba(255,255,255,.6);
    border:var(--border-thin); border-radius:20px; overflow:hidden;
}
.conf-fill { height:100%; background:var(--mint-deep); border-radius:20px; }

/* ── body text ── */
.text-sky   { font-size:13px; line-height:1.75; color:var(--sky-deep);  }
.text-green { font-size:13px; line-height:1.75; color:var(--mint-deep); }
.text-peach { font-size:13px; line-height:1.75; color:var(--peach-deep); }
</style>
""", unsafe_allow_html=True)


# CONSTANTS & HELPERS
# ──────────────────────────────────────────────────────────────────────────────
PRESETS = [
    "when you already started eating and someone says 'lets pray'",
    "need a hug ? i love hugs",
    "when people talk trash but you're a bigger person",
    "and if anything happens, the women are to blame that's right, it's their fault, definitely",
]
PRESET_LABELS = ["🐹 Let's Pray", "🐕 Need a Hug", "😎 Bigger Person", "⚠️ Hate Test"]
DEMO_IMAGES_DIR = "demo_images"

STAGES = [
    ("🧠", "Embedding your meme..."),
    ("🔍", "Retrieving similar memes..."),
    ("🤖", "Asking Llama 3..."),
    ("📋", "Generating results..."),
]

def thinking_bar_html(active_step: int) -> str:
    """Render the dark animated thinking bar."""
    pct = int((active_step / len(STAGES)) * 100)
    stage_rows = ""
    for i, (icon, label) in enumerate(STAGES):
        if i < active_step:
            cls, check = "stage-done",   "✅"
        elif i == active_step:
            cls, check = "stage-active", '<span style="color:#FFD166;animation:tpulse 1s infinite">⏳</span>'
        else:
            cls, check = "stage-wait",   ""
        stage_rows += f"""
        <div class="stage {cls}">
          <div class="stage-icon">{icon}</div>
          <div class="stage-label">{label}</div>
          <div class="stage-check">{check}</div>
        </div>"""
    step_label = f"step {active_step + 1} of {len(STAGES)}"
    foot_right = "almost there 👀" if active_step < len(STAGES) - 1 else "finalizing..."
    return f"""
    <div class="thinking-card">
      <div class="thinking-title">
        <div class="thinking-dot"></div> Llama 3 is analyzing...
      </div>
      <div class="stages">{stage_rows}</div>
      <div class="prog-track">
        <div class="prog-fill" style="width:{pct}%"></div>
      </div>
      <div class="prog-footer"><span>{step_label}</span><span>{foot_right}</span></div>
    </div>"""

def toast_html(is_hateful: bool, confidence: int) -> str:
    """Render the toast notification that slides in after classification."""
    if is_hateful:
        return f"""
        <div class="toast toast-hateful">
          <div style="font-size:16px">⚠️</div>
          <div>
            <div class="toast-text toast-text-hateful">Hateful · {confidence}% confidence</div>
            <div class="toast-sub">classification complete</div>
          </div>
        </div>"""
    return f"""
    <div class="toast">
      <div style="font-size:16px">✅</div>
      <div>
        <div class="toast-text">Not hateful · {confidence}% confidence 💯</div>
        <div class="toast-sub">classification complete</div>
      </div>
    </div>"""

def try_load_image(meme_id: str):
    """
    Tries to find the image in demo_images.
    Handles 'facebook_78034' by checking for '78034.png' and '078034.png'.
    """
    if not meme_id: return None
    
    clean_id = re.sub(r'[^0-9]', '', str(meme_id))
    
    candidates = [
        f"{clean_id}.png", 
        f"{clean_id.zfill(5)}.png", 
        f"{meme_id}.png"
    ]
    
    for ext in ("png", "jpg", "jpeg", "webp"):
        for filename in candidates:
            path = os.path.join(DEMO_IMAGES_DIR, filename.replace(".png", f".{ext}"))
            if os.path.exists(path):
                return path
    return None


# SESSION STATE
# ──────────────────────────────────────────────────────────────────────────────
if "meme_input_area" not in st.session_state:
    st.session_state.meme_input_area = ""
if "result" not in st.session_state:
    st.session_state.result = None
if "run_effects" not in st.session_state:
    st.session_state.run_effects = False


# NAVBAR
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="meme-nav">
  <div class="logo-cluster">
    <div class="emoji-col"><div>😂 🔥</div><div>💀 👀</div></div>
    <div class="logo-text">
      <div class="logo-wordmark">
        <span class="c1">M</span><span class="c2">e</span>
        <span class="c3">m</span><span class="c4">e</span>
        <span class="csep">·</span>
        <span class="c5">R</span><span class="c6">A</span><span class="c7">G</span>
      </div>
      <div class="logo-tagline">meme understanding · hate detection · cs 6120</div>
    </div>
    <div class="emoji-col"><div>😭 🤡</div><div>💯 🫠</div></div>
  </div>
  <div class="nav-pills">
    <span class="nav-pill pill-a">8,500 memes</span>
    <span class="nav-pill pill-b">Llama 3 · local</span>
    <span class="nav-pill pill-c">F1 · 0.73</span>
  </div>
</div>
""", unsafe_allow_html=True)


# INPUT
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">📝 Drop your meme text</div>', unsafe_allow_html=True)

# Callback function to safely update the text box
def apply_preset(text):
    st.session_state.meme_input_area = text
    st.session_state.result = None

input_left, input_right = st.columns([1.6, 1], gap="medium")

with input_left:
    meme_input = st.text_area(
        label="meme_text_area",
        placeholder="paste meme text here…",
        height=110,
        key="meme_input_area",
        label_visibility="collapsed",
    )

with input_right:
    pcols = st.columns(4)
    for i, (pc, short_lbl, full_text) in enumerate(zip(pcols, PRESET_LABELS, PRESETS)):
        with pc:
            
            st.button(
                short_lbl, 
                key=f"preset_{i}", 
                use_container_width=True,
                on_click=apply_preset,
                args=(full_text,)
            )
            
    st.markdown('<div class="analyze-btn">', unsafe_allow_html=True)
    analyze_clicked = st.button(
        "🔍 Analyze this meme →",
        use_container_width=True,
        key="analyze_btn",
    )
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# LOGIC PIPELINE
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
        st.error(f"⚠️ Pipeline Connection Error: {e}")
        st.stop()

    bar_slot.markdown(thinking_bar_html(3), unsafe_allow_html=True)
    time.sleep(0.4)
    bar_slot.empty()


# SPLIT-SCREEN RESULTS
# ──────────────────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1.1, 1], gap="large")

# ── LEFT ─────────────────────────────────────────────────────────────────────
with left_col:
    if not st.session_state.result:
        st.markdown("""
        <div class="card card-empty" style="
            min-height:260px;display:flex;flex-direction:column;
            align-items:center;justify-content:center;
            text-align:center;gap:12px;">
          <div style="font-size:44px;letter-spacing:.4rem;">🧠 🔍 💬</div>
          <div style="font-family:'Fredoka',sans-serif;font-size:19px;
                      font-weight:700;color:#1a1a1a;">Enter a meme and hit Analyze</div>
          <div style="font-size:13px;color:#999;font-weight:700;
                      max-width:300px;line-height:1.6;">
            Retrieves the 5 most similar labeled memes from 8,500 entries,
            then asks Llama 3 to explain and classify.
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown('<div class="section-label">🖼️ Meme Image / Ground Truth</div>', unsafe_allow_html=True)

        citations = st.session_state.result.get("citations", [])
        meme_id = str(citations[0].get("id", "")) if citations else ""
        img_path = try_load_image(meme_id)
        
        if img_path:
            st.image(img_path, use_column_width=True)
        else:
            raw_text = citations[0].get("text", meme_input) if citations else meme_input
            dataset_src = citations[0].get("dataset", "train.jsonl") if citations else "train.jsonl"
            
            st.markdown(f"""
            <div style="border: var(--border); border-radius: var(--radius); padding: 20px; background: #fff; box-shadow: var(--shadow-sm); margin-bottom: 15px;">
                <div style="display: inline-block; background: var(--amber); padding: 4px 12px; border-radius: 20px; font-size: 11px; font-weight: 700; border: var(--border-thin); margin-bottom: 12px;">
                    📄 Dataset Text Citation
                </div>
                <div style="font-family: 'Courier New', monospace; font-size: 16px; font-weight: 700; color: #1a1a1a; margin-bottom: 12px; line-height: 1.4;">
                    "{raw_text}"
                </div>
                <div style="font-size: 12px; color: #666; border-top: 1px dashed #ccc; padding-top: 8px;">
                    <strong>Source Data:</strong> <code>{dataset_src}</code> &nbsp;|&nbsp; 
                    <strong>ID:</strong> <code>{meme_id.replace('facebook_', '')}</code>
                </div>
            </div>
            """, unsafe_allow_html=True)

        explanation = st.session_state.result.get("explanation", "No explanation returned.")
        st.markdown(f"""
        <div class="card card-explain" style="margin-top:15px;">
          <div class="section-label">🧠 What this meme means</div>
          <div class="text-sky">{explanation}</div>
        </div>""", unsafe_allow_html=True)

# ── RIGHT ────────────────────────────────────────────────────────────────────
with right_col:
    if st.session_state.result:
        r = st.session_state.result
        label_raw   = str(r.get("hate_label", "not hateful")).lower()
        is_hateful  = ("hate" in label_raw) and ("not" not in label_raw)
        badge_cls   = "badge-hateful" if is_hateful else "badge-safe"
        badge_emoji = "⚠️" if is_hateful else "✅"
        badge_text  = "Hateful" if is_hateful else "Not hateful"
        verdict_cls = "card-hateful" if is_hateful else "card-verdict"
        body_cls    = "text-peach"   if is_hateful else "text-green"

        reasoning  = r.get("rationale", r.get("reasoning", "No reasoning returned."))
        
        raw_conf = r.get("confidence", 75)
        confidence = int(raw_conf) if float(raw_conf) > 1.0 else int(float(raw_conf) * 100)
        confidence = max(0, min(100, confidence)) 

        st.markdown(toast_html(is_hateful, confidence), unsafe_allow_html=True)

        st.markdown(f"""
        <div class="card {verdict_cls}">
          <div class="section-label">🔍 Hate classification</div>
          <div class="verdict-badge {badge_cls}">
            <div class="verdict-dot"></div>{badge_emoji} {badge_text}
          </div>
          <div class="{body_cls}">{reasoning}</div>
          <div class="conf-wrap">
            <div class="conf-header"><span>Confidence</span><span>{confidence}%</span></div>
            <div class="conf-track">
              <div class="conf-fill" style="width:{confidence}%;"></div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

        if st.session_state.get("run_effects"):
            if is_hateful:
                st.snow()
            else:
                st.balloons()
            st.session_state.run_effects = False

st.markdown("<br><br>", unsafe_allow_html=True)


# RETRIEVED EVIDENCE & CITATIONS (THE "AUDIT TRAIL")
# ──────────────────────────────────────────────────────────────────────────────
if st.session_state.result:
    sources = st.session_state.result.get("citations", [])

    if sources:
        st.markdown(f"""
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
            <div class="section-label" style="margin:0;">📌 Retrieved Evidence Gallery</div>
            <div style="font-size:11px;font-weight:700;color:#7A5FAF;">
              Top {len(sources[:5])} vectors retrieved from ChromaDB
            </div>
          </div>
        """, unsafe_allow_html=True)

        img_cols = st.columns(len(sources[:5]))
        for col, s in zip(img_cols, sources[:5]):
            with col:
                ev_id = str(s.get("id", "")).replace("facebook_", "")
                ev_img_path = try_load_image(ev_id)
                
                if ev_img_path:
                    st.image(ev_img_path, use_column_width=True)
                else:
                    st.markdown(f"""
                    <div style="border: 2px dashed #ccc; border-radius: 8px; height: 100%; min-height: 100px; display: flex; align-items: center; justify-content: center; background: rgba(255,255,255,0.4); text-align: center;">
                        <span style="font-size: 10px; font-weight: 700; color: #888;">No Image<br>ID: {ev_id}</span>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("<hr style='border-color: rgba(0,0,0,0.1); margin: 25px 0 15px 0;'>", unsafe_allow_html=True)
        st.markdown('<div class="section-label" style="font-size: 11px;">📄 Formal Dataset Citations</div>', unsafe_allow_html=True)

        for idx, s in enumerate(sources[:5]):
            ev_id = str(s.get("id", "")).replace("facebook_", "")
            ev_text = s.get("text", "Text missing from database")
            ev_dataset = s.get("dataset", "train.jsonl") 
            ev_label = s.get("label_str", "unknown").upper()
            ev_dist = s.get("distance", "N/A")
            
            tag_color = "#FFB5A7" if "HATEFUL" in ev_label and "NOT" not in ev_label else "#B8F0D8"

            st.markdown(f"""
            <div style="border: 1px solid #ddd; border-left: 4px solid #7A5FAF; border-radius: 6px; padding: 12px; margin-bottom: 10px; background: rgba(255,255,255,0.8);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <div style="font-size: 12px; font-weight: 700; color: #4A1D8F;">Citation {idx + 1}</div>
                    <div style="display: flex; gap: 8px; align-items: center;">
                        <span style="font-size: 10px; font-weight: 700; background: #eee; padding: 2px 6px; border-radius: 4px;">Dist: {ev_dist}</span>
                        <span style="font-size: 10px; font-weight: 700; background: {tag_color}; padding: 2px 6px; border-radius: 4px; color: #1a1a1a;">{ev_label}</span>
                    </div>
                </div>
                <div style="font-family: 'Courier New', monospace; font-size: 13px; font-weight: 600; color: #1a1a1a; background: #f9f9f9; padding: 8px; border-radius: 4px; border: 1px solid #eee; margin-bottom: 8px;">
                    "{ev_text}"
                </div>
                <div style="font-size: 11px; font-weight: 700; color: #666;">
                    <strong>Dataset Source:</strong> <code>{ev_dataset}</code> &nbsp;|&nbsp; <strong>Meme ID:</strong> <code>{ev_id}</code>
                </div>
            </div>
            """, unsafe_allow_html=True)
