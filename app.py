<FILE file_path="/home/workdir/attachments/app.py" size="10088 bytes">
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
    background-color: #F8FAFC;
}

/* Main Container Styling */
.main-container {
    background: white;
    border-radius: 24px;
    padding: 2.5rem 2rem;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.06);
    margin-bottom: 2rem;
}

/* Title Container */
.title-container {
    background: linear-gradient(135deg, #E0F2FE, #F0F9FF);
    padding: 2rem 1.5rem;
    border-radius: 20px;
    text-align: center;
    margin-bottom: 1.5rem;
    border: 1px solid #BAE6FD;
    box-shadow: 0 4px 15px rgba(14, 165, 233, 0.08);
}

/* Main Title */
.main-title {
    font-size: 3.1rem;
    font-weight: 800;
    color: #0F766E;
    margin-bottom: 0.4rem;
    letter-spacing: -1.2px;
    text-shadow: 0 2px 4px rgba(15, 118, 110, 0.1);
}

/* Subtitle */
.subtitle {
    font-size: 1.65rem;
    color: #334155;
    margin: 0;
    line-height: 1.4;
    font-weight: 500;
}

/* Divider */
hr {
    border: none;
    height: 1px;
    background: linear-gradient(to right, transparent, #94A3B8, transparent);
    margin: 2rem 0;
}

/* Metric Cards - Soft & Empathetic */
.metric-box {
    background: white;
    padding: 2rem 1.8rem;
    border-radius: 20px;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.07);
    text-align: center;
    border: 1px solid #E2E8F0;
    margin-bottom: 1.8rem;
    transition: all 0.3s ease;
}

.metric-box:hover {
    transform: translateY(-4px);
    box-shadow: 0 20px 35px rgba(0, 0, 0, 0.1);
    border-color: #67E8F9;
}

/* Metric Titles */
.metric-title {
    font-size: 0.95rem;
    text-transform: uppercase;
    color: #64748B;
    letter-spacing: 0.5px;
    margin-bottom: 0.6rem;
    font-weight: 600;
}

/* Metric Values */
.metric-value {
    font-size: 2.35rem;
    font-weight: 700;
    line-height: 1.1;
}

/* Text Area */
textarea {
    border-radius: 18px !important;
    border: 2px solid #CBD5E1 !important;
    background-color: #FFFFFF !important;
    color: #1E2937 !important;
    font-size: 1.05rem !important;
    padding: 1.25rem !important;
    line-height: 1.7 !important;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.03);
}

textarea:focus {
    border-color: #14B8A6 !important;
    box-shadow: 0 0 0 3px rgba(20, 184, 166, 0.15) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #14B8A6, #0F766E);
    color: white;
    border: none;
    border-radius: 16px;
    padding: 0.95rem 1.5rem;
    font-weight: 600;
    font-size: 1.05rem;
    box-shadow: 0 6px 20px rgba(20, 184, 166, 0.25);
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 25px rgba(20, 184, 166, 0.35);
}

/* Sidebar - Gentle & Supportive */
section[data-testid="stSidebar"] {
    background-color: #F1F5F9;
    border-right: 1px solid #E2E8F0;
}

section[data-testid="stSidebar"] * {
    color: #334155;
}

/* Sidebar Headers */
section[data-testid="stSidebar"] h1, 
section[data-testid="stSidebar"] h2, 
section[data-testid="stSidebar"] h3 {
    color: #0F766E;
}

/* Alerts */
.stAlert {
    border-radius: 16px;
    border: none;
}

/* Plotly Chart */
[data-testid="stPlotlyChart"] {
    background: white;
    border-radius: 20px;
    padding: 1rem;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.06);
    border: 1px solid #F1F5F9;
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

# 3. Cached Model Loading Pipeline (unchanged)
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
    st.sidebar.error("Model Loading Error. Verify your configuration files are in the main folder.")
    st.stop()

# 4. Sidebar Content (minimal visual update)
st.sidebar.markdown("---")
st.sidebar.markdown("### Usage Guidelines")
st.sidebar.info("""
**Best Practices for Analysis:**
* **Natural Language:** Type or paste complete sentences. 
* **Character Length:** Ideal inputs range between 10 to 200 words.
* **Objective Evaluation:** Results show probability distribution across four psychological contexts.
""")
st.sidebar.markdown("---")
st.sidebar.caption("*All analyses are processed locally in your browser.*")

# 5. Text Cleaning (unchanged)
def clean_input_text(text):
    text = html.unescape(str(text))
    text = re.sub(r'<[^>]+>', ' ', text)
    text = " ".join(text.split())
    return text

# 6. User Input
user_input = st.text_area(
    "Enter text context below to compute deep learning inferences:",
    placeholder="Type out statements or text sequences here to analyze context...",
    height=160
)

# 7. Execution and Logic (only visual updates below)
if st.button("Run Live Prediction Analytics", type="primary", use_container_width=True):
    if user_input.strip() == "":
        st.warning("Access Denied: Input text cannot be left blank.")
    else:
        with st.spinner("Processing text sequences..."):
            cleaned_text = clean_input_text(user_input).lower()
            
            # --- DEFINITIVE SAFETY BYPASS OVERRIDE PATTERNS --- (unchanged)
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
            
            # Updated Theme Colors - More Human, Calming & Empathetic
            theme_colors = {
                'Normal': {
                    'hex': '#10B981',
                    'bg': '#ECFDF5'
                },
                'Anxiety': {
                    'hex': '#F59E0B',
                    'bg': '#FFFBEB'
                },
                'Depression': {
                    'hex': '#64748B',
                    'bg': '#F8FAFC'
                },
                'Suicidal': {
                    'hex': '#EF4444',
                    'bg': '#FEF2F2'
                }
            }
            active_color = theme_colors[predicted_label]['hex']
            
            # Primary Metric Box
            st.markdown(f"""
                <div class="metric-box" style="border-left: 7px solid {active_color}; background-color: {theme_colors[predicted_label]['bg']};">
                    <div class="metric-title">Primary Psychological Context</div>
                    <div class="metric-value" style="color: {active_color};">{predicted_label}</div>
                    <div style="font-size: 1.35rem; color: #475569; margin-top: 0.3rem;">
                        {highest_confidence:.1f}% confidence
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Interactive Plotly Chart (colors updated)
            fig = go.Figure(go.Bar(
                x=[p * 100 for p in probabilities],
                y=class_labels,
                orientation='h',
                marker=dict(
                    color=[theme_colors[lbl]['hex'] for lbl in class_labels],
                    line=dict(color='rgba(0, 0, 0, 0.1)', width=1)
                ),
                text=[f"{p*100:.1f}%" for p in probabilities],
                textposition='outside',
                hoverinfo='x'
            ))
            
            fig.update_layout(
                title=dict(text="Confidence Distribution", font=dict(size=15, color="#334155")),
                xaxis=dict(title="Probability (%)", range=[0, 115], showgrid=True, gridcolor='#E2E8F0'),
                yaxis=dict(autorange="reversed", tickfont=dict(size=13, weight="600")),
                margin=dict(l=20, r=30, t=40, b=20),
                height=280,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Crisis Warning
            if crisis_prob >= 0.25:
                st.markdown(f"""
                    <div style="background-color: #FEF3F2; border-left: 6px solid #EF4444; padding: 1.25rem; border-radius: 16px; margin-top: 1.5rem;">
                        <span style="font-weight: bold; color: #B91C1C;">⚠️ Critical Risk Alert</span><br>
                        <span style="color: #991B1B; font-size: 1rem;">
                            High-severity signals detected. Crisis probability: <b>{crisis_prob*100:.1f}%</b>.<br>
                            Please reach out to a trusted person or professional helpline immediately.
                        </span>
                    </div>
                """, unsafe_allow_html=True)
</FILE>
