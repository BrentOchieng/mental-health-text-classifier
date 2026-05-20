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

/* Main App Background */
.stApp {
    background-color: #E7F0F8;
}

/* Main Title */
.main-title {
    font-size: 2.9rem;
    font-weight: 800;
    color: #0f5132;
    text-align: center;
    margin-bottom: 0.3rem;
    letter-spacing: -0.5px;
}

/* Subtitle */
.subtitle {
    font-size: 1.9rem;
    color: #0F5132;
    text-align: center;
    margin-bottom: 2rem;
    line-height: 1.7;
}

/* Divider */
hr {
    border: none;
    height: 1px;
    background: #0f5132;
}

/* Metric Cards */
.metric-box {
    background: #FFFFFF;
    padding: 1.6rem;
    border-radius: 18px;
    box-shadow: 0 8px 24px rgba(47, 93, 80, 0.06);
    text-align: center;
    border-left: 6px solid #7AA38D;
    margin-bottom: 1.5rem;
    transition: all 0.25s ease;
}

.metric-box:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 28px rgba(47, 93, 80, 0.10);
}

/* Metric Titles */
.metric-title {
    font-size: 0.82rem;
    text-transform: uppercase;
    color: #0F5132;
    letter-spacing: 0.09em;
    margin-bottom: 0.5rem;
    font-weight: 600;
}

/* Metric Values */
.metric-value {
    font-size: 2rem;
    font-weight: 700;
}

/* Text Area */
textarea {
    border-radius: 16px !important;
    border: 1px solid #D7E6DD !important;
    background-color: #FFFFFF !important;
    color: #0F5132 !important;
    font-size: 1rem !important;
    padding: 1rem !important;
    line-height: 1.6 !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(
        135deg,
        #0F5132,
        #0F5132
    );

    color: white;
    border: none;
    border-radius: 14px;
    padding: 0.8rem 1rem;
    font-weight: 600;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: scale(1.01);
    box-shadow: 0 8px 24px rgba(95, 158, 125, 0.25);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #C9D6CE;
}

/* Sidebar Text */
section[data-testid="stSidebar"] * {
    color: #145A32;
}

/* Alerts */
.stAlert {
    border-radius: 14px;
}

/* Plotly Chart Container */
[data-testid="stPlotlyChart"] {
    background: ;
    border-radius: 16px;
    padding: 0.5rem;
    box-shadow: 0 4px 16px rgba(0,0,0,0.03);
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
        'hex': '#4E8B70',
        'bg': '#EDF8F2'
    },

    'Anxiety': {
        'hex': '#A7B97F',
        'bg': '#F8FBF1'
    },

    'Depression': {
        'hex': '#6E8F80',
        'bg': '#F2F7F5'
    },

    'Suicidal': {
        'hex': '#8C6F5C',
        'bg': '#FAF4F1'
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

