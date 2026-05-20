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

    /* Global App Background */
    .stApp {
        background-color: #F5F7FA;
    }

    /* Main Title */
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: #2E4057;
        text-align: center;
        margin-bottom: 0.4rem;
        letter-spacing: -0.5px;
    }

    /* Subtitle */
    .subtitle {
        font-size: 1.05rem;
        color: #5F6C7B;
        text-align: center;
        margin-bottom: 2rem;
        line-height: 1.6;
    }

    /* Metric Cards */
    .metric-box {
        background: #FFFFFF;
        padding: 1.6rem;
        border-radius: 18px;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
        text-align: center;
        border-left: 6px solid #7C9EB2;
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
    }

    .metric-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(15, 23, 42, 0.10);
    }

    .metric-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        color: #7B8794;
        letter-spacing: 0.08em;
        margin-bottom: 0.4rem;
        font-weight: 600;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }

    /* Text Area */
    textarea {
        border-radius: 14px !important;
        border: 1px solid #D6DEE6 !important;
        background-color: #FFFFFF !important;
        color: #2D3748 !important;
        font-size: 1rem !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6C8EBF, #89A8C9);
        color: white;
        border: none;
        border-radius: 14px;
        padding: 0.75rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: scale(1.01);
        box-shadow: 0 8px 20px rgba(108, 142, 191, 0.25);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #EDF2F7;
    }

    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">MindContext AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Advanced Text-Based Mental Health Context Classifier</div>', unsafe_allow_html=True)
st.markdown("---")

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
    placeholder="Type out statements or text sequences here to analyze context...",
    height=160
)

# 7. Execution and Logic
if st.button("Run Live Prediction Analytics ", type="primary", use_container_width=True):
    if user_input.strip() == "":
        st.warning(" Access Denied: Input text cannot be left blank.")
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
            
            theme_colors = {
    'Normal': {
        'hex': '#4C956C',
        'bg': '#EDF7F1'
    },

    'Anxiety': {
        'hex': '#C08457',
        'bg': '#FFF6EE'
    },

    'Depression': {
        'hex': '#7B6D8D',
        'bg': '#F5F1FA'
    },

    'Suicidal': {
        'hex': '#A65D57',
        'bg': '#FFF1F0'
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
                    line=dict(color='rgba(0, 0, 0, 0.15)', width=1)
                ),
                text=[f"{p*100:.1f}%" for p in probabilities],
                textposition='outside',
                hoverinfo='x'
            ))
            
            fig.update_layout(
                title=dict(text="Confidence Distribution Array Matrix", font=dict(size=14, color="#4B5563")),
                xaxis=dict(title="Probability Score (%)", range=[0, 115], showgrid=True, gridcolor='#E5E7EB'),
                yaxis=dict(autorange="reversed", tickfont=dict(size=12, weight="bold")),
                margin=dict(l=20, r=20, t=40, b=20),
                height=260,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Crisis Warning Output Flag
            if crisis_prob >= 0.25:
                st.markdown("""
                    <div style="background-color: #FFFBEB; border-left: 6px solid #D97706; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                        <span style="font-weight: bold; color: #B45309;"> Critical Risk Safeguard Alert:</span><br>
                        <span style="color: #78350F; font-size: 0.95rem;">
                            High-severity mental health signals detected. Security tracking yields a crisis score baseline of <b>{:.1f}%</b>. 
                            Ensure emergency helpline resources are made visible.
                        </span>
                    </div>
                """.format(crisis_prob * 100), unsafe_allow_html=True)

