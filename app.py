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

# 2. Inject Custom CSS for Slate Grey & Steel Blue UI Styling
st.markdown("""
    <style>
        /* Base page background using a soft, calming steel grey */
        .stApp {
            background-color: #F1F5F9;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        /* Premium minimalistic headers */
        .main-title {
            font-size: 2.8rem;
            font-weight: 800;
            color: #1E293B;
            text-align: center;
            margin-bottom: 0.25rem;
            letter-spacing: -0.025em;
        }
        .subtitle {
            font-size: 1.15rem;
            color: #64748B;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: 400;
        }
        
        /* Containers wrap cleanly inside soft slate-toned borders (No more harsh white blocks) */
        div[data-testid="stForm"], .stTextArea, div[data-testid="stMetricValue"] {
            background-color: #E2E8F0 !important;
            border: 1px solid #CBD5E1 !important;
            border-radius: 12px !important;
            box-shadow: inset 0 1px 2px 0 rgba(0, 0, 0, 0.02) !important;
        }
        
        /* Input text block refinement */
        .stTextArea textarea {
            background-color: #E2E8F0 !important;
            border: none !important;
            font-size: 16px !important;
            color: #1E293B !important;
            padding: 1rem !important;
        }
        
        /* Bespoke Slate Blue Action Button */
        .stButton>button {
            background-color: #334155 !important; /* Deep Slate Charcoal */
            color: #F8FAFC !important;
            border-radius: 8px !important;
            border: 1px solid #475569 !important;
            font-weight: 600 !important;
            font-size: 16px !important;
            padding: 0.7rem 2rem !important;
            transition: all 0.2s ease-in-out;
            box-shadow: 0 4px 6px -1px rgba(51, 65, 85, 0.12) !important;
            margin-top: 0.5rem;
        }
        
        .stButton>button:hover {
            background-color: #475569 !important; /* Lighter slate on hover */
            transform: translateY(-1px);
            box-shadow: 0 6px 10px -1px rgba(51, 65, 85, 0.2) !important;
        }
        
        /* Cleaned metric output card structure */
        .metric-box {
            padding: 1.75rem;
            border-radius: 12px;
            box-shadow: 0 4px 12px -1px rgba(0, 0, 0, 0.02);
            text-align: center;
            border: 1px solid rgba(0, 0, 0, 0.05);
            margin-bottom: 1.75rem;
            margin-top: 1rem;
        }
        .metric-title {
            font-size: 0.85rem;
            text-transform: uppercase;
            color: #475569;
            letter-spacing: 0.075em;
            font-weight: 600;
            margin-bottom: 0.4rem;
        }
        .metric-value { 
            font-size: 2rem; 
            font-weight: 800; 
            letter-spacing: -0.02em;
        }
        
        /* Sidebar styling synchronization */
        section[data-testid="stSidebar"] {
            background-color: #E2E8F0 !important;
            border-right: 1px solid #CBD5E1;
        }
        
        hr {
            margin: 2rem 0 !important;
            border-top: 1px solid #CBD5E1 !important;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">MindContext AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Advanced Text-Based Mental Health Context Classifier</div>', unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)


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

# 4. Sidebar Content: Clean System Guidelines
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
            
            # Muted clinical tones pairing with the slate gray UI interface
            theme_colors = {
                'Normal': {'hex': '#475569', 'bg': '#CBD5E1'},      # Muted Slate Grey
                'Anxiety': {'hex': '#D97706', 'bg': '#FEF3C7'},     # Warm Amber
                'Depression': {'hex': '#2563EB', 'bg': '#DBEAFE'},  # Soft Corporate Blue
                'Suicidal': {'hex': '#B91C1C', 'bg': '#FEE2E2'}    # Deep Red Alert
            }
            active_color = theme_colors[predicted_label]['hex']
            
            # Primary Metric Box Display
            st.markdown(f"""
                <div class="metric-box" style="border-left: 6px solid {active_color}; background-color: {theme_colors[predicted_label]['bg']};">
                    <div class="metric-title" style="color: {active_color if predicted_label == 'Normal' else '#475569'};">Identified Primary Psychological Context</div>
                    <div class="metric-value" style="color: {active_color};">{predicted_label} ({highest_confidence:.1f}%)</div>
                </div>
            """, unsafe_allow_html=True)
            
            # Interactive Plotly Distribution Chart (Formatted to look clean and minimalist)
            fig = go.Figure(go.Bar(
                x=[p * 100 for p in probabilities],
                y=class_labels,
                orientation='h',
                marker=dict(
                    color=[theme_colors[lbl]['hex'] for lbl in class_labels],
                    line=dict(color='rgba(255, 255, 255, 0)', width=0)
                ),
                text=[f"{p*100:.1f}%" for p in probabilities],
                textposition='outside',
                hoverinfo='x',
                textfont=dict(size=12, color='#1E293B', family='Inter')
            ))
            
            fig.update_layout(
                title=dict(text="Confidence Distribution Array Matrix", font=dict(size=13, color="#475569", family='Inter')),
                xaxis=dict(title="Probability Score (%)", range=[0, 115], showgrid=True, gridcolor='#CBD5E1', titlefont=dict(color='#475569')),
                yaxis=dict(autorange="reversed", tickfont=dict(size=13, color='#1E293B')),
                margin=dict(l=20, r=20, t=40, b=20),
                height=260,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Crisis Warning Output Flag
            if crisis_prob >= 0.25:
                st.markdown("""
                    <div style="background-color: #FCA5A5; border-left: 6px solid #B91C1C; padding: 1.25rem; border-radius: 12px; margin-top: 1.5rem; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);">
                        <span style="font-weight: 700; color: #7F1D1D; font-size: 1rem;">🛡️ Critical Risk Safeguard Alert:</span><br>
                        <span style="color: #451A03; font-size: 0.95rem; display: block; margin-top: 0.25rem; font-weight: 500;">
                            High-severity mental health signals detected. Security tracking yields a crisis score baseline of <b>{:.1f}%</b>. 
                            Ensure emergency helpline resources are made visible.
                        </span>
                    </div>
                """.format(crisis_prob * 100), unsafe_allow_html=True)