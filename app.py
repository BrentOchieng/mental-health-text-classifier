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

# 2. Inject Custom CSS for Bright, Clinical, Mental Health-Focused UI
st.markdown("""
<style>
/* App Background - Bright Soft Mint */
.stApp { background-color: #F8FDFB; }

/* Main Title - Dark Forest Green for professional contrast */
.main-title {
    font-size: 2.8rem;
    font-weight: 800;
    color: #145A32;
    text-align: center;
    margin-bottom: 0.5rem;
}

/* Subtitle Container - Bright Clean White */
.subtitle-container {
    background-color: #FFFFFF;
    padding: 1.5rem;
    border-radius: 16px;
    border: 1px solid #DDEBE3;
    margin-bottom: 2rem;
    box-shadow: 0 4px 12px rgba(20, 90, 50, 0.05);
}

.subtitle {
    font-size: 1.2rem;
    color: #2D4A3D;
    text-align: center;
    font-weight: 500;
}

/* Chart Container Wrapper */
.chart-container {
    background-color: #FFFFFF;
    padding: 1.5rem;
    border-radius: 16px;
    border: 1px solid #DDEBE3;
    box-shadow: 0 4px 12px rgba(20, 90, 50, 0.08);
    margin-top: 1rem;
}

/* Metric Cards */
.metric-box {
    background: #FFFFFF;
    padding: 1.6rem;
    border-radius: 18px;
    box-shadow: 0 4px 12px rgba(20, 90, 50, 0.08);
    text-align: center;
    border-left: 6px solid #145A32;
    margin-bottom: 1.5rem;
}

.metric-title {
    font-size: 0.82rem;
    text-transform: uppercase;
    color: #145A32;
    letter-spacing: 0.09em;
    margin-bottom: 0.5rem;
    font-weight: 600;
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #145A32;
}

/* Text Area */
textarea {
    border-radius: 12px !important;
    border: 1px solid #A8C6B9 !important;
    background-color: #FFFFFF !important;
    color: #1A3C2E !important;
    font-size: 1.1rem !important;
    padding: 1rem !important;
}

/* Sidebar */
section[data-testid="stSidebar"] { background-color: #E8F3EF; }
section[data-testid="stSidebar"] * { color: #145A32 !important; }

/* Buttons */
.stButton > button {
    background: #145A32;
    color: white;
    border-radius: 10px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">MindContext AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-container"><div class="subtitle">Advanced Text-Based Mental Health Context Classifier</div></div>', unsafe_allow_html=True)

# 3. Cached Model Loading
@st.cache_resource
def load_pipeline():
    model_path = "ourafla/mental-health-bert-finetuned" 
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=4, ignore_mismatched_sizes=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

try:
    tokenizer, model, device = load_pipeline()
    st.sidebar.markdown("### System Status")
    st.sidebar.success("Model Status: Online & Loaded")
except Exception:
    st.sidebar.error("Model Loading Error.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown("### Usage Guidelines")
st.sidebar.info("""
* **Natural Language:** Type complete sentences.
* **Length:** 10 to 200 words recommended.
* **Evaluation:** Outputs probability distributions.
""")

def clean_input_text(text):
    text = html.unescape(str(text))
    text = re.sub(r'<[^>]+>', ' ', text)
    return " ".join(text.split())

st.markdown("<h4 style='color:#145A32;'>Enter text context below to compute deep learning inferences:</h4>", unsafe_allow_html=True)
user_input = st.text_area("", placeholder="Type out statements or text sequences here to analyze context...", height=160)

# 7. Execution and Logic
if st.button("Run Live Prediction Analytics ", type="primary", use_container_width=True):
    if user_input.strip() == "":
        st.warning(" Access Denied: Input text cannot be left blank.")
    else:
        with st.spinner("Processing text sequences..."):
            cleaned_text = clean_input_text(user_input).lower()
            
            safety_keywords = [r"\bkill\s+myself\b", r"\bend\s+my\s+life\b", r"\bsuicide\b", r"\bcommit\s+suicide\b"]
            is_safety_override = any(re.search(pattern, cleaned_text) for pattern in safety_keywords)
            class_labels = ['Anxiety', 'Depression', 'Normal', 'Suicidal']
            
            if is_safety_override:
                predicted_label, highest_confidence, probabilities, crisis_prob = 'Suicidal', 100.0, [0.0, 0.0, 0.0, 1.0], 1.0
            else:
                inputs = tokenizer(cleaned_text, truncation=True, padding='max_length', max_length=256, return_tensors="pt").to(device)
                model.eval()
                with torch.no_grad():
                    logits = model(**inputs).logits
                    probabilities = F.softmax(logits, dim=-1).squeeze().cpu().numpy()
                predicted_id = probabilities.argmax()
                predicted_label, highest_confidence, crisis_prob = class_labels[predicted_id], probabilities[predicted_id] * 100, probabilities[3]
            
            st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-title">Identified Primary Psychological Context</div>
                    <div class="metric-value">{predicted_label} ({highest_confidence:.1f}%)</div>
                </div>
            """, unsafe_allow_html=True)
            
            # Chart Rendering inside Container
            fig = go.Figure(go.Bar(
                x=[p * 100 for p in probabilities], y=class_labels, orientation='h',
                marker_color='#145A32'
            ))
            fig.update_layout(height=260, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=20, r=20, t=20, b=20))
            
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("<p style='color:#145A32; font-weight:bold;'>Confidence Distribution Array Matrix</p>", unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            if crisis_prob >= 0.25:
                st.warning(f"Critical Risk Safeguard Alert: High-severity signals detected ({crisis_prob*100:.1f}%). Ensure emergency resources are visible.")
