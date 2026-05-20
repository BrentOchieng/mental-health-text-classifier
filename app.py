import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import html
import re
import plotly.graph_objects as go

# 1. Page Configuration
st.set_page_config(
    page_title="MindContext AI",
    page_icon="",
    layout="centered"
)

# 2. Inject Custom CSS for Premium UI Styling
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Lora:wght@400;600;700&family=Source+Sans+3:wght@300;400;500;600&display=swap');

/* ── Root Palette ─────────────────────────────────────────────────
   Deep warm espresso backgrounds, sage green accents, soft
   terracotta for crisis signals, dusty slate for depression,
   amber for anxiety. Dark theme — like a quiet evening study,
   calm and steady, not cold or stark.
──────────────────────────────────────────────────────────────── */
:root {
    --bg-base:        #1A1815;   /* deep warm espresso */
    --bg-card:        #232018;   /* slightly lifted card surface */
    --bg-sidebar:     #161411;   /* deeper sidebar well */

    --sage-deep:      #5C825A;   /* primary sage — brightened for dark bg */
    --sage-mid:       #7AA876;   /* mid sage — hover, accents */
    --sage-soft:      #3A5238;   /* muted sage — borders on dark */
    --sage-mist:      #1F2E1E;   /* very dark sage — tag bg */

    --stone-ink:      #EDE8E1;   /* warm off-white for headings */
    --stone-mid:      #A89F96;   /* muted warm for body text */
    --stone-light:    #6A6259;   /* captions, labels */

    --amber:          #C97B3A;   /* anxiety — lifted for dark */
    --amber-soft:     #2A1F12;
    --slate:          #7A8499;   /* depression — lifted for dark */
    --slate-soft:     #1A1D26;
    --terracotta:     #C45A42;   /* suicidal / crisis — lifted for dark */
    --terracotta-soft:#271210;
    --normal:         #5C825A;   /* normal — same as sage */
    --normal-soft:    #1F2E1E;

    --radius-card:    16px;
    --radius-btn:     12px;
    --shadow-card:    0 2px 16px rgba(0, 0, 0, 0.35);
    --shadow-hover:   0 6px 28px rgba(0, 0, 0, 0.50);

    --font-display:   'Lora', Georgia, serif;
    --font-body:      'Source Sans 3', 'Helvetica Neue', sans-serif;
}

/* ── Global Reset ─────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: var(--font-body) !important;
    color: var(--stone-ink);
}

/* ── App Background ───────────────────────────────────────────── */
.stApp {
    background-color: var(--bg-base);
    background-image:
        radial-gradient(ellipse at 18% 12%, rgba(92,130,90,0.10) 0%, transparent 55%),
        radial-gradient(ellipse at 85% 80%, rgba(196,90,66,0.06) 0%, transparent 50%);
}

/* ── Header Container ─────────────────────────────────────────── */
.header-wrap {
    background: var(--bg-card);
    border: 1px solid var(--sage-soft);
    border-radius: var(--radius-card);
    padding: 2.4rem 2rem 2rem;
    text-align: center;
    margin-bottom: 1.8rem;
    box-shadow: var(--shadow-card);
    position: relative;
    overflow: hidden;
}

.header-wrap::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
    background: linear-gradient(90deg, var(--sage-deep), var(--sage-mid), var(--sage-soft));
}

.main-title {
    font-family: var(--font-display);
    font-size: 2.5rem;
    font-weight: 700;
    color: var(--sage-deep);
    margin-bottom: 0.5rem;
    letter-spacing: -0.3px;
    line-height: 1.2;
}

.subtitle {
    font-family: var(--font-body);
    font-size: 1rem;
    font-weight: 400;
    color: var(--stone-mid);
    line-height: 1.6;
    max-width: 480px;
    margin: 0 auto;
    letter-spacing: 0.01em;
}

.header-badge {
    display: inline-block;
    background: var(--sage-mist);
    color: var(--sage-deep);
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    border: 1px solid var(--sage-soft);
    border-radius: 100px;
    padding: 0.28rem 0.9rem;
    margin-bottom: 1rem;
}

/* ── Divider ──────────────────────────────────────────────────── */
hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--sage-soft), transparent);
    margin: 1.6rem 0;
}

/* ── Textarea ─────────────────────────────────────────────────── */
textarea {
    font-family: var(--font-body) !important;
    border-radius: var(--radius-card) !important;
    border: 1.5px solid var(--sage-soft) !important;
    background-color: #2A2620 !important;
    color: var(--stone-ink) !important;
    font-size: 0.97rem !important;
    padding: 1rem 1.1rem !important;
    line-height: 1.65 !important;
    box-shadow: inset 0 1px 4px rgba(0,0,0,0.20) !important;
    transition: border-color 0.2s ease !important;
}

textarea:focus {
    border-color: var(--sage-mid) !important;
    box-shadow: 0 0 0 3px rgba(107,143,98,0.12) !important;
}

/* ── Input Label ──────────────────────────────────────────────── */
label[data-testid="stWidgetLabel"] > div > p {
    font-family: var(--font-body) !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    color: var(--stone-mid) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
    margin-bottom: 0.4rem !important;
}

/* ── Primary Button ───────────────────────────────────────────── */
.stButton > button {
    font-family: var(--font-body) !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    background: var(--sage-deep) !important;
    color: #FDFBF8 !important;
    border: none !important;
    border-radius: var(--radius-btn) !important;
    padding: 0.78rem 1.2rem !important;
    letter-spacing: 0.02em !important;
    transition: all 0.22s ease !important;
    box-shadow: 0 2px 8px rgba(74,103,65,0.25) !important;
}

.stButton > button:hover {
    background: var(--sage-mid) !important;
    box-shadow: 0 6px 20px rgba(74,103,65,0.30) !important;
    transform: translateY(-1px) !important;
}

.stButton > button:active {
    transform: translateY(0px) !important;
    box-shadow: 0 2px 6px rgba(74,103,65,0.20) !important;
}

/* ── Metric Card ──────────────────────────────────────────────── */
.metric-box {
    background: var(--bg-card);
    padding: 1.6rem 1.8rem;
    border-radius: var(--radius-card);
    box-shadow: var(--shadow-card);
    text-align: center;
    border-left: 5px solid var(--sage-deep);
    margin-bottom: 1.5rem;
    transition: box-shadow 0.22s ease, transform 0.22s ease;
}

.metric-box:hover {
    box-shadow: var(--shadow-hover);
    transform: translateY(-2px);
}

.metric-title {
    font-family: var(--font-body);
    font-size: 0.75rem;
    text-transform: uppercase;
    color: var(--stone-light);
    letter-spacing: 0.10em;
    margin-bottom: 0.55rem;
    font-weight: 600;
}

.metric-value {
    font-family: var(--font-display);
    font-size: 1.9rem;
    font-weight: 700;
    line-height: 1.1;
}

/* ── Section Heading ──────────────────────────────────────────── */
h3, .section-title {
    font-family: var(--font-display) !important;
    color: var(--stone-ink) !important;
    font-weight: 600 !important;
    font-size: 1.15rem !important;
    margin-bottom: 0.8rem !important;
}

/* ── Sidebar ──────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background-color: var(--bg-sidebar) !important;
    border-right: 1px solid var(--sage-soft) !important;
}

section[data-testid="stSidebar"] * {
    color: var(--stone-ink) !important;
}

section[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: var(--font-display) !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    color: var(--sage-deep) !important;
    border-bottom: 1px solid var(--sage-soft);
    padding-bottom: 0.4rem;
    margin-bottom: 0.8rem !important;
}

section[data-testid="stSidebar"] .stAlert {
    background: var(--sage-mist) !important;
    border: 1px solid var(--sage-soft) !important;
    border-left: 3px solid var(--sage-deep) !important;
    border-radius: 10px !important;
    color: var(--stone-mid) !important;
    font-size: 0.875rem !important;
    line-height: 1.6 !important;
}

section[data-testid="stSidebar"] .stSuccess {
    background: var(--sage-mist) !important;
    border: 1px solid var(--sage-soft) !important;
    border-left: 3px solid var(--sage-deep) !important;
    border-radius: 10px !important;
    color: var(--sage-deep) !important;
    font-size: 0.875rem !important;
}

section[data-testid="stSidebar"] small,
section[data-testid="stSidebar"] caption,
section[data-testid="stSidebar"] .stCaption {
    color: var(--stone-light) !important;
    font-size: 0.78rem !important;
    font-style: italic;
}

/* ── Warning / Alert Overrides ────────────────────────────────── */
.stAlert {
    border-radius: 12px !important;
    font-family: var(--font-body) !important;
}

.stWarning {
    background: #FFF8ED !important;
    border-left-color: var(--amber) !important;
    color: var(--stone-ink) !important;
}

/* ── Spinner ──────────────────────────────────────────────────── */
.stSpinner > div {
    border-top-color: var(--sage-mid) !important;
}

/* ── Plotly Chart Container ───────────────────────────────────── */
[data-testid="stPlotlyChart"] {
    background: var(--bg-card);
    border: 1px solid var(--sage-soft);
    border-radius: var(--radius-card);
    padding: 0.6rem;
    box-shadow: var(--shadow-card);
}

/* ── Scrollbar (subtle) ───────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--sage-soft); border-radius: 8px; }
::-webkit-scrollbar-thumb:hover { background: var(--sage-mid); }

</style>
""", unsafe_allow_html=True)

# ── Header Container ───────────────────────────────────────────────────────────
st.markdown("""
<div class="header-wrap">
    <div class="header-badge">Mental Health NLP · Deep Learning</div>
    <div class="main-title">MindContext AI</div>
    <div class="subtitle">
        A contextual text classifier for mental health signal detection —
        grounded in empathy, guided by data.
    </div>
</div>
""", unsafe_allow_html=True)

# 3. Cached Model Loading Pipeline (Production-Grade Fine-Tuned Model)
@st.cache_resource
def load_pipeline():
    # Points to a highly accurate public BERT model fine-tuned on exactly 4 mental health classes
    model_path = "ourafla/mental-health-bert-finetuned" 
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, 
        num_labels=4,
        ignore_mismatched_sizes=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

try:
    tokenizer, model, device = load_pipeline()
    st.sidebar.markdown("### System Status")
    st.sidebar.success("Model Status: Online & Loaded")
except Exception as e:
    st.sidebar.error("Model Loading Error. Verify your configuration files are in the main folder.")
    st.stop()

# 4. New Sidebar Content: Clean System Guidelines
st.sidebar.markdown("---")
st.sidebar.markdown("### Usage Guidelines")
st.sidebar.info("""
**Best Practices for Analysis:**
* **Natural Language:** Type or paste complete sentences. The deep learning model relies on grammatical context and sentence structures.
* **Character Length:** Ideal inputs range between 10 to 200 words for optimal attention-matrix processing.
* **Objective Evaluation:** The dashboard outputs probability distributions across four separate psychological linguistic contexts.
""")
st.sidebar.markdown("---")
st.sidebar.caption("*All string analyses are processed locally on the system memory pipeline.*")

# 5. Dedicated Text Cleaning Rule
def clean_input_text(text):
    text = html.unescape(str(text))
    text = re.sub(r'<[^>]+>', ' ', text)
    text = " ".join(text.split())
    return text

# 6. Core User Input Area
user_input = st.text_area(
    "Enter text context below to compute deep learning inferences:",
    placeholder="Type out statements or text sequences here to analyse context...",
    height=160
)

# 7. Execution and Logic
if st.button("Run Live Prediction Analytics", type="primary", use_container_width=True):
    if user_input.strip() == "":
        st.warning("Input text cannot be left blank.")
    else:
        with st.spinner("Processing text sequences..."):
            cleaned_text = clean_input_text(user_input).lower()
            
            # --- DEFINITIVE SAFETY BYPASS OVERRIDE PATTERNS ---
            safety_keywords = [
                r"\bkill\s+myself\b", 
                r"\bend\s+my\s+life\b", 
                r"\bsuicide\b", 
                r"\bcommit\s+suicide\b"
            ]
            
            is_safety_override = any(re.search(pattern, cleaned_text) for pattern in safety_keywords)
            
            class_labels = ['Anxiety', 'Depression', 'Normal', 'Suicidal']
            
            if is_safety_override:
                # Direct string override bypass logic
                predicted_label = 'Suicidal'
                highest_confidence = 100.0
                probabilities = [0.0, 0.0, 0.0, 1.0] # Hardcode array matrix for chart
                crisis_prob = 1.0
            else:
                # Fallback to standard BERT Deep Learning Model Inference
                inputs = tokenizer(clean_input_text(user_input), truncation=True, padding='max_length', max_length=256, return_tensors="pt")
                inputs = {key: val.to(device) for key, val in inputs.items()}
                
                model.eval()
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    probabilities = F.softmax(logits, dim=-1).squeeze().cpu().numpy()
                
                predicted_id = torch.argmax(logits, dim=-1).item()
                predicted_label = class_labels[predicted_id]
                highest_confidence = probabilities[predicted_id] * 100
                crisis_prob = probabilities[3]
            
            st.markdown("### Classification Analytics Dashboard")
            
            # ── Refined semantic palette aligned with mental health tone ──
            theme_colors = {
                'Normal': {
                    'hex': '#4A6741',      # sage green — calm, grounded
                    'bg':  '#EBF2E7'
                },
                'Anxiety': {
                    'hex': '#B5692A',      # warm amber — restless, alert
                    'bg':  '#F5E6D5'
                },
                'Depression': {
                    'hex': '#5A6278',      # dusty slate — muted, withdrawn
                    'bg':  '#E3E6EF'
                },
                'Suicidal': {
                    'hex': '#B04A36',      # terracotta — urgent, grounding
                    'bg':  '#F5E0DA'
                }
            }

            active_color = theme_colors[predicted_label]['hex']
            
            # Primary Metric Box Display
            st.markdown(f"""
                <div class="metric-box" style="border-left-color: {active_color}; background-color: {theme_colors[predicted_label]['bg']};">
                    <div class="metric-title">Identified Primary Psychological Context</div>
                    <div class="metric-value" style="color: {active_color};">{predicted_label} ({highest_confidence:.1f}%)</div>
                </div>
            """, unsafe_allow_html=True)
            
            # Interactive Plotly Distribution Chart
            fig = go.Figure(go.Bar(
                x=[p * 100 for p in probabilities],
                y=class_labels,
                orientation='h',
                marker=dict(
                    color=[theme_colors[lbl]['hex'] for lbl in class_labels],
                    opacity=0.88,
                    line=dict(color='rgba(44, 41, 37, 0.10)', width=1)
                ),
                text=[f"{p*100:.1f}%" for p in probabilities],
                textposition='outside',
                textfont=dict(family="Source Sans 3, sans-serif", size=12, color="#A89F96"),
                hoverinfo='x'
            ))
            
            fig.update_layout(
                title=dict(
                    text="Confidence Distribution",
                    font=dict(family="Lora, Georgia, serif", size=14, color="#A89F96"),
                    x=0.02
                ),
                xaxis=dict(
                    title="Probability Score (%)",
                    title_font=dict(family="Source Sans 3, sans-serif", size=11, color="#A09890"),
                    range=[0, 118],
                    showgrid=True,
                    gridcolor='#2E2B26',
                    tickfont=dict(family="Source Sans 3, sans-serif", size=11, color="#6A6259"),
                    zeroline=False
                ),
                yaxis=dict(
                    autorange="reversed",
                    tickfont=dict(family="Source Sans 3, sans-serif", size=12, color="#EDE8E1"),
                ),
                margin=dict(l=20, r=30, t=44, b=24),
                height=260,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Crisis Warning Output Flag
            if crisis_prob >= 0.25:
                st.markdown("""
                    <div style="
                        background-color: #FDF3F1;
                        border: 1px solid #D9A89F;
                        border-left: 5px solid #B04A36;
                        padding: 1.1rem 1.3rem;
                        border-radius: 12px;
                        margin-top: 1rem;
                        font-family: 'Source Sans 3', sans-serif;
                    ">
                        <span style="font-weight: 700; color: #8B2E1F; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.06em;">
                            ⚠ Critical Risk Signal Detected
                        </span><br><br>
                        <span style="color: #4A2020; font-size: 0.94rem; line-height: 1.65;">
                            High-severity mental health signals have been identified in this input.
                            The crisis confidence score registers at <b style="color:#B04A36;">{:.1f}%</b>.
                            Please ensure emergency helpline resources and support contacts are made immediately visible to the user.
                        </span>
                    </div>
                """.format(crisis_prob * 100), unsafe_allow_html=True)
