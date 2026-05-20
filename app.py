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

# 2. Updated Premium & Human-Centered CSS
st.markdown("""
<style>
/* Overall App - Soft, Calming, Human Feel */
.stApp {
    background: linear-gradient(180deg, #F8FAFC 0%, #F3F7F7 100%);
    color: #1F2937;
}

/* Main content spacing */
.block-container {
    padding-top: 1.6rem;
    padding-bottom: 2rem;
}

/* Title Container */
.title-container {
    background: linear-gradient(135deg, #F0F7F6 0%, #EAF4F3 100%);
    padding: 1.8rem 1.5rem;
    border-radius: 22px;
    text-align: center;
    margin-bottom: 1.2rem;
    border: 1px solid #D3E6E4;
    box-shadow: 0 10px 28px rgba(15, 118, 110, 0.08);
}

/* Main Title */
.main-title {
    font-size: 3rem;
    font-weight: 800;
    color: #0F766E;
    margin-bottom: 0.35rem;
    letter-spacing: -1px;
    line-height: 1.05;
}

/* Subtitle */
.subtitle {
    font-size: 1.1rem;
    color: #475569;
    margin: 0;
    line-height: 1.5;
    font-weight: 500;
}

/* Metric Cards */
.metric-box {
    background: rgba(255, 255, 255, 0.92);
    padding: 1.8rem 1.6rem;
    border-radius: 20px;
    box-shadow: 0 10px 26px rgba(15, 23, 42, 0.06);
    text-align: center;
    border: 1px solid #E2E8F0;
    margin-bottom: 1.4rem;
    transition: all 0.25s ease;
    backdrop-filter: blur(6px);
}

.metric-box:hover {
    transform: translateY(-3px);
    box-shadow: 0 16px 34px rgba(15, 23, 42, 0.09);
}

/* Metric Titles */
.metric-title {
    font-size: 0.9rem;
    text-transform: uppercase;
    color: #64748B;
    letter-spacing: 0.6px;
    margin-bottom: 0.55rem;
    font-weight: 700;
}

/* Metric Values */
.metric-value {
    font-size: 2.2rem;
    font-weight: 800;
    line-height: 1.1;
}

/* Text Area */
textarea {
    border-radius: 18px !important;
    border: 1.8px solid #CBD5E1 !important;
    background-color: #FFFFFF !important;
    color: #1F2937 !important;
    font-size: 1.02rem !important;
    padding: 1.15rem !important;
    line-height: 1.65 !important;
    box-shadow: inset 0 1px 2px rgba(15, 23, 42, 0.04);
}

textarea:focus {
    border-color: #0F766E !important;
    box-shadow: 0 0 0 3px rgba(15, 118, 110, 0.14) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #0F766E 0%, #115E59 100%);
    color: white;
    border: none;
    border-radius: 16px;
    padding: 0.95rem 1.5rem;
    font-weight: 700;
    font-size: 1.02rem;
    box-shadow: 0 8px 20px rgba(15, 118, 110, 0.22);
    transition: all 0.25s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 14px 26px rgba(15, 118, 110, 0.28);
    filter: brightness(1.02);
}

.stButton > button:active {
    transform: translateY(0px);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #F4F7F7 0%, #EEF4F3 100%);
    border-right: 1px solid #D9E6E4;
}

section[data-testid="stSidebar"] * {
    color: #334155;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #0F766E;
}

/* Sidebar widgets */
section[data-testid="stSidebar"] [data-testid="stNotificationContent"],
section[data-testid="stSidebar"] [data-testid="stInfoBox"],
section[data-testid="stSidebar"] .stAlert {
    border-radius: 16px;
}

/* Plotly Chart */
[data-testid="stPlotlyChart"] {
    background: white;
    border-radius: 20px;
    padding: 1rem;
    box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
    border: 1px solid #EEF2F7;
}

/* Text input label */
[data-testid="stTextArea"] label {
    color: #334155 !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# Title & Subtitle in dedicated containers
st.markdown("""
<div class="title-container">
    <div class="main-title">MindContext AI</div>
    <div class="subtitle">Advanced Text-Based Mental Health Context Classifier</div>
</div>
""", unsafe_allow_html=True)

# Model Loading
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
    st.sidebar.markdown("### System Status")
    st.sidebar.success("Model Status: Online & Loaded")
except Exception as e:
    st.sidebar.error("Model Loading Error. Please check your setup.")
    st.stop()

# Sidebar Guidelines
st.sidebar.markdown("---")
st.sidebar.markdown("### Usage Guidelines")
st.sidebar.info("""
**Best Practices for Analysis:**
* Natural Language: Use complete sentences
* Best length: 10 to 200 words
* Results show probability across four contexts
""")
st.sidebar.markdown("---")
st.sidebar.caption("*All analyses are processed locally*")

# Text Cleaning
def clean_input_text(text):
    text = html.unescape(str(text))
    text = re.sub(r'<[^>]+>', ' ', text)
    text = " ".join(text.split())
    return text

# User Input
user_input = st.text_area(
    "Enter text context below to compute deep learning inferences:",
    placeholder="Type out statements or text sequences here to analyze context...",
    height=160
)

# Main Logic
if st.button("Run Live Prediction Analytics", type="primary", use_container_width=True):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing text..."):
            cleaned_text = clean_input_text(user_input).lower()

            safety_keywords = [
                r"\bkill\s+myself\b",
                r"\bend\s+my\s+life\b",
                r"\bsuicide\b",
                r"\bcommit\s+suicide\b"
            ]

            is_safety_override = any(re.search(pattern, cleaned_text) for pattern in safety_keywords)

            class_labels = ['Anxiety', 'Depression', 'Normal', 'Suicidal']

            if is_safety_override:
                predicted_label = 'Suicidal'
                highest_confidence = 100.0
                probabilities = [0.0, 0.0, 0.0, 1.0]
                crisis_prob = 1.0
            else:
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

            theme_colors = {
                'Normal': {'hex': '#1F9D74', 'bg': '#ECF8F3'},
                'Anxiety': {'hex': '#C58B2E', 'bg': '#FFF7E8'},
                'Depression': {'hex': '#64748B', 'bg': '#F6F8FA'},
                'Suicidal': {'hex': '#D64545', 'bg': '#FEF1F1'}
            }
            active_color = theme_colors[predicted_label]['hex']

            # Primary Metric Box
            st.markdown(f"""
                <div class="metric-box" style="border-left: 7px solid {active_color}; background-color: {theme_colors[predicted_label]['bg']};">
                    <div class="metric-title">Primary Psychological Context</div>
                    <div class="metric-value" style="color: {active_color};">{predicted_label}</div>
                    <div style="font-size: 1.2rem; color: #475569; margin-top: 0.3rem; font-weight: 500;">
                        {highest_confidence:.1f}% confidence
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Fixed Plotly Chart
            fig = go.Figure(go.Bar(
                x=[p * 100 for p in probabilities],
                y=class_labels,
                orientation='h',
                marker=dict(
                    color=[theme_colors[lbl]['hex'] for lbl in class_labels],
                    line=dict(color='rgba(0, 0, 0, 0.08)', width=1)
                ),
                text=[f"{p*100:.1f}%" for p in probabilities],
                textposition='outside'
            ))

            fig.update_layout(
                title="Confidence Distribution",
                xaxis=dict(title="Probability (%)", range=[0, 115], gridcolor='#E2E8F0'),
                yaxis=dict(autorange="reversed", tickfont=dict(size=13, weight="600")),
                margin=dict(l=20, r=30, t=50, b=20),
                height=280,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#334155")
            )

            st.plotly_chart(fig, use_container_width=True)

            # Crisis Warning
            if crisis_prob >= 0.25:
                st.markdown(f"""
                    <div style="background-color: #FFF1F1; border-left: 6px solid #D64545; padding: 1.2rem; border-radius: 16px; margin-top: 1.5rem; box-shadow: 0 8px 18px rgba(214, 69, 69, 0.08);">
                        <span style="font-weight: bold; color: #A61B1B;">⚠️ Critical Risk Alert</span><br>
                        <span style="color: #7F1D1D;">
                            High-severity signals detected. Crisis probability: <b>{crisis_prob*100:.1f}%</b>.<br>
                            Please reach out to a trusted person or professional helpline immediately.
                        </span>
                    </div>
                """, unsafe_allow_html=True)
