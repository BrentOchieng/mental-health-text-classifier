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
    page_icon="🧠",
    layout="centered"
)

# 2. Premium Human-Centered Mental Health UI Styling
st.markdown("""
<style>

/* ===== GLOBAL APP ===== */
.stApp {
    background: linear-gradient(180deg, #F4F7F5 0%, #EEF2EF 100%);
    color: #1F2937;
}

/* Remove Streamlit Default Padding Top */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* ===== CUSTOM CONTAINERS ===== */

/* Hero Container */
.hero-container {
    background: rgba(255,255,255,0.78);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(214, 222, 218, 0.85);
    padding: 2rem;
    border-radius: 26px;
    margin-bottom: 1.3rem;
    box-shadow: 0 10px 30px rgba(44, 62, 80, 0.05);
}

/* Main Title */
.main-title {
    font-size: 3rem;
    font-weight: 800;
    color: #163A2B;
    text-align: center;
    margin-bottom: 0.4rem;
    letter-spacing: -0.8px;
}

/* Subtitle */
.subtitle {
    font-size: 1.05rem;
    color: #5B6B63;
    text-align: center;
    line-height: 1.8;
    font-weight: 400;
}

/* Secondary Info Container */
.info-container {
    background: rgba(255,255,255,0.82);
    border: 1px solid #E4ECE7;
    border-radius: 22px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 6px 18px rgba(44, 62, 80, 0.03);
}

/* Divider */
hr {
    border: none;
    height: 1px;
    background: #DCE6DF;
    margin-top: 1.5rem;
    margin-bottom: 1.5rem;
}

/* ===== METRIC CARD ===== */
.metric-box {
    background: rgba(255,255,255,0.85);
    padding: 1.7rem;
    border-radius: 22px;
    text-align: center;
    border: 1px solid #E5ECE7;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 24px rgba(44, 62, 80, 0.05);
    transition: all 0.25s ease;
}

.metric-box:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 28px rgba(44, 62, 80, 0.08);
}

.metric-title {
    font-size: 0.78rem;
    text-transform: uppercase;
    color: #6B7280;
    letter-spacing: 0.08em;
    margin-bottom: 0.7rem;
    font-weight: 700;
}

.metric-value {
    font-size: 2rem;
    font-weight: 800;
}

/* ===== TEXT AREA ===== */
textarea {
    border-radius: 20px !important;
    border: 1px solid #D6E2DA !important;
    background-color: rgba(255,255,255,0.92) !important;
    color: #1F2937 !important;
    font-size: 1rem !important;
    padding: 1rem !important;
    line-height: 1.8 !important;
    box-shadow: 0 3px 10px rgba(0,0,0,0.02);
}

textarea:focus {
    border: 1px solid #5C8D76 !important;
    box-shadow: 0 0 0 3px rgba(92, 141, 118, 0.12) !important;
}

/* Label Text */
label {
    color: #31443A !important;
    font-weight: 600 !important;
}

/* ===== BUTTONS ===== */
.stButton > button {
    background: linear-gradient(
        135deg,
        #4E7A67,
        #3E6655
    );
    color: white;
    border: none;
    border-radius: 16px;
    padding: 0.85rem 1rem;
    font-weight: 600;
    font-size: 1rem;
    transition: all 0.25s ease;
    box-shadow: 0 6px 18px rgba(78, 122, 103, 0.18);
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 10px 24px rgba(78, 122, 103, 0.24);
    background: linear-gradient(
        135deg,
        #5B8B76,
        #486F5F
    );
}

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #E8EFEA 0%, #DFE8E2 100%);
    border-right: 1px solid #D5DED8;
}

/* Sidebar Text */
section[data-testid="stSidebar"] * {
    color: #274839 !important;
}

/* Sidebar Headers */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #173728 !important;
}

/* ===== ALERTS ===== */
.stAlert {
    border-radius: 16px;
    border: 1px solid #E8E8E8;
}

/* ===== PLOTLY CHART ===== */
[data-testid="stPlotlyChart"] {
    background: rgba(255,255,255,0.82);
    border-radius: 22px;
    padding: 1rem;
    border: 1px solid #E5ECE7;
    box-shadow: 0 6px 20px rgba(44, 62, 80, 0.04);
}

/* ===== SPINNER ===== */
.stSpinner > div {
    border-top-color: #4E7A67 !important;
}

/* ===== CAPTION ===== */
.caption-text {
    color: #6B7280;
    font-size: 0.9rem;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# HERO SECTION
st.markdown("""
<div class="hero-container">
    <div class="main-title">MindContext AI</div>
    <div class="subtitle">
        Advanced Text-Based Mental Health Context Classifier
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-container">
    <div class="caption-text">
        AI-powered contextual mental health text analysis designed with a calm, human-centered interface.
    </div>
</div>
""", unsafe_allow_html=True)

# 3. Cached Model Loading Pipeline (Production-Grade Fine-Tuned Model)
@st.cache_resource
def load_pipeline():
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

    st.sidebar.markdown("## System Status")
    st.sidebar.success("Model Status: Online & Loaded")

except Exception as e:
    st.sidebar.error(
        "Model Loading Error. Verify your configuration files are in the main folder."
    )
    st.stop()

# 4. Sidebar Content
st.sidebar.markdown("---")

st.sidebar.markdown("## Usage Guidelines")

st.sidebar.info("""
### Best Practices for Analysis

- Use complete and natural sentences.
- Ideal inputs range between 10–200 words.
- The model evaluates contextual psychological language patterns.
- Results represent probabilistic contextual classifications.
""")

st.sidebar.markdown("---")

st.sidebar.caption(
    "All analyses are processed locally within the system inference pipeline."
)

# 5. Text Cleaning
def clean_input_text(text):
    text = html.unescape(str(text))
    text = re.sub(r'<[^>]+>', ' ', text)
    text = " ".join(text.split())
    return text

# 6. Input Area
user_input = st.text_area(
    "Enter text context below to compute deep learning inferences:",
    placeholder="Type statements, emotions, conversations, or psychological text patterns here for contextual analysis...",
    height=180
)

# 7. Main Execution
if st.button(
    "Run Live Prediction Analytics",
    type="primary",
    use_container_width=True
):

    if user_input.strip() == "":
        st.warning("Access Denied: Input text cannot be left blank.")

    else:

        with st.spinner("Processing contextual language patterns..."):

            cleaned_text = clean_input_text(user_input).lower()

            # SAFETY OVERRIDE
            safety_keywords = [
                r"\bkill\s+myself\b",
                r"\bend\s+my\s+life\b",
                r"\bsuicide\b",
                r"\bcommit\s+suicide\b"
            ]

            is_safety_override = any(
                re.search(pattern, cleaned_text)
                for pattern in safety_keywords
            )

            class_labels = ['Anxiety', 'Depression', 'Normal', 'Suicidal']

            if is_safety_override:

                predicted_label = 'Suicidal'
                highest_confidence = 100.0
                probabilities = [0.0, 0.0, 0.0, 1.0]
                crisis_prob = 1.0

            else:

                inputs = tokenizer(
                    clean_input_text(user_input),
                    truncation=True,
                    padding='max_length',
                    max_length=256,
                    return_tensors="pt"
                )

                inputs = {
                    key: val.to(device)
                    for key, val in inputs.items()
                }

                model.eval()

                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits

                    probabilities = F.softmax(
                        logits,
                        dim=-1
                    ).squeeze().cpu().numpy()

                predicted_id = torch.argmax(logits, dim=-1).item()

                predicted_label = class_labels[predicted_id]

                highest_confidence = probabilities[predicted_id] * 100

                crisis_prob = probabilities[3]

            st.markdown("## Classification Analytics Dashboard")

            # HUMAN-CENTERED COLOR THEMES
            theme_colors = {

                'Normal': {
                    'hex': '#4E8B70',
                    'bg': '#EEF7F1'
                },

                'Anxiety': {
                    'hex': '#C48A4A',
                    'bg': '#FBF5ED'
                },

                'Depression': {
                    'hex': '#6E738B',
                    'bg': '#F3F4F8'
                },

                'Suicidal': {
                    'hex': '#C96C5B',
                    'bg': '#FBF0ED'
                }
            }

            active_color = theme_colors[predicted_label]['hex']

            # METRIC CARD
            st.markdown(f"""
                <div class="metric-box"
                     style="
                        border-left: 6px solid {active_color};
                        background-color: {theme_colors[predicted_label]['bg']};
                     ">
                    <div class="metric-title">
                        Identified Primary Psychological Context
                    </div>

                    <div class="metric-value"
                         style="color: {active_color};">
                        {predicted_label} ({highest_confidence:.1f}%)
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # PLOTLY CHART
            fig = go.Figure(go.Bar(

                x=[p * 100 for p in probabilities],

                y=class_labels,

                orientation='h',

                marker=dict(
                    color=[
                        theme_colors[lbl]['hex']
                        for lbl in class_labels
                    ],

                    line=dict(
                        color='rgba(0,0,0,0.12)',
                        width=1
                    )
                ),

                text=[f"{p*100:.1f}%" for p in probabilities],

                textposition='outside',

                hoverinfo='x'
            ))

            fig.update_layout(

                title=dict(
                    text="Confidence Distribution Matrix",
                    font=dict(
                        size=15,
                        color="#374151"
                    )
                ),

                xaxis=dict(
                    title="Probability Score (%)",
                    range=[0, 115],
                    showgrid=True,
                    gridcolor='#E7ECE9'
                ),

                yaxis=dict(
                    autorange="reversed",
                    tickfont=dict(
                        size=12
                    )
                ),

                margin=dict(
                    l=20,
                    r=20,
                    t=45,
                    b=20
                ),

                height=280,

                plot_bgcolor='rgba(0,0,0,0)',

                paper_bgcolor='rgba(0,0,0,0)'
            )

            st.plotly_chart(fig, use_container_width=True)

            # CRISIS ALERT
            if crisis_prob >= 0.25:

                st.markdown(f"""
                    <div style="
                        background-color: #FFF7ED;
                        border-left: 6px solid #D97706;
                        padding: 1.2rem;
                        border-radius: 16px;
                        margin-top: 1rem;
                        border: 1px solid #F6D7B8;
                    ">

                        <span style="
                            font-weight: 700;
                            color: #B45309;
                            font-size: 1rem;
                        ">
                            Critical Risk Safeguard Alert
                        </span>

                        <br><br>

                        <span style="
                            color: #7C4A10;
                            font-size: 0.96rem;
                            line-height: 1.7;
                        ">
                            High-severity mental health signals detected.
                            Crisis analysis confidence baseline:
                            <b>{crisis_prob * 100:.1f}%</b>.

                            Ensure emergency support resources
                            remain immediately accessible.
                        </span>

                    </div>
                """, unsafe_allow_html=True)
