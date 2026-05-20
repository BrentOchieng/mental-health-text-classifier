import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import html
import re
import plotly.graph_objects as go

# 1. Page Configuration
st.set_page_config(page_title="MindContext AI", page_icon="🧠", layout="centered")

# 2. Inject Custom CSS for "Warm Parchment" Aesthetic
st.markdown("""
<style>
/* App Background - Soft Warm Parchment */
.stApp { background-color: ##F2F0EA; }

/* Main Title */
.main-title { font-size: 2.8rem; font-weight: 800; color: #2D241E; text-align: center; margin-bottom: 0.5rem; }

/* Subtitle Container - Warm earth tone */
.subtitle-container { 
    background-color: #EDE8E1; padding: 1.5rem; border-radius: 16px; 
    border: 1px solid #D6CDC1; margin-bottom: 2rem; box-shadow: 0 4px 12px rgba(0,0,0,0.05); 
}
.subtitle { font-size: 1.2rem; color: #5D544E; text-align: center; font-weight: 500; }

/* Containers - Clean white for high contrast data */
.chart-container, .metric-box { 
    background-color: #FFFFFF; padding: 1.5rem; border-radius: 16px; 
    border: 1px solid #D6CDC1; box-shadow: 0 4px 12px rgba(0,0,0,0.08); margin-top: 1rem; 
}

/* Sidebar Styling - Soft Brown-Green Blend */
section[data-testid="stSidebar"] { background-color: #F4F1ED; border-right: 1px solid #D6CDC1; }
section[data-testid="stSidebar"] * { color: #4A423D !important; }

/* Text Area */
textarea { 
    border-radius: 12px !important; border: 2px solid #C4B9AA !important; 
    background-color: #FFFFFF !important; color: #2D241E !important; 
}

/* Buttons */
.stButton > button { background: #5D544E; color: #FFFFFF; border-radius: 10px; font-weight: 600; border: none; }
.stButton > button:hover { background: #2D241E; }
</style>
""", unsafe_allow_html=True)

# Layout
st.markdown('<div class="main-title">MindContext AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-container"><div class="subtitle">Advanced Text-Based Mental Health Context Classifier</div></div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("### 🧠 System Status")
st.sidebar.success("Model Status: Online")
st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 Usage Guidelines")
st.sidebar.write("* **Natural Language:** Provide full, clear sentences.")
st.sidebar.write("* **Context Length:** 10–200 words.")
st.sidebar.write("* **Privacy:** All processing occurs locally.")

# Model Logic
@st.cache_resource
def load_pipeline():
    model_path = "ourafla/mental-health-bert-finetuned"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=4, ignore_mismatched_sizes=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_pipeline()

# Warm Palette for Results
theme_colors = {'Anxiety': '#D08C3F', 'Depression': '#6B6F8A', 'Normal': '#4E8B70', 'Suicidal': '#B22222'}

user_input = st.text_area("Enter text context below:", placeholder="Type statements here...", height=160)

if st.button("Run Live Prediction Analytics ", type="primary", use_container_width=True):
    if user_input.strip() == "":
        st.warning("Input text cannot be left blank.")
    else:
        with st.spinner("Analyzing..."):
            class_labels = ['Anxiety', 'Depression', 'Normal', 'Suicidal']
            inputs = tokenizer(user_input, truncation=True, padding='max_length', max_length=256, return_tensors="pt").to(device)
            with torch.no_grad():
                probs = F.softmax(model(**inputs).logits, dim=-1).squeeze().cpu().numpy()
            
            pred_idx = probs.argmax()
            pred_label, pred_color = class_labels[pred_idx], theme_colors[class_labels[pred_idx]]
            
            st.markdown(f"""
                <div class="metric-box" style="border-left: 10px solid {pred_color};">
                    <p class="metric-title" style="color: #5D544E;">Primary Psychological Context</p>
                    <h2 style="color: {pred_color};">{pred_label} ({probs[pred_idx]*100:.1f}%)</h2>
                </div>
            """, unsafe_allow_html=True)
            
            fig = go.Figure(go.Bar(
                x=probs * 100, y=class_labels, orientation='h',
                marker_color=[theme_colors[lbl] for lbl in class_labels],
                text=[f"{p*100:.1f}%" for p in probs], textposition='auto'
            ))
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=True, gridcolor='#E8E2DA', tickfont=dict(color='#5D544E')),
                yaxis=dict(tickfont=dict(color='#2D241E', weight='bold')), margin=dict(l=20, r=20, t=10, b=10), height=250
            )
            
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("<p style='color:#5D544E; font-weight:bold;'>Confidence Distribution Array Matrix</p>", unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
