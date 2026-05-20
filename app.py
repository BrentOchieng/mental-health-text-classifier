import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import html
import re
import plotly.graph_objects as go

# 1. Page Configuration
st.set_page_config(page_title="MindContext AI", page_icon="🧠", layout="centered")

# 2. Inject Custom CSS
st.markdown("""
<style>
.stApp { background-color: #F8FDFB; }
.main-title { font-size: 2.8rem; font-weight: 800; color: #145A32; text-align: center; margin-bottom: 0.5rem; }
.subtitle-container { background-color: #FFFFFF; padding: 1.5rem; border-radius: 16px; border: 1px solid #DDEBE3; margin-bottom: 2rem; box-shadow: 0 4px 12px rgba(20, 90, 50, 0.05); }
.subtitle { font-size: 1.2rem; color: #2D4A3D; text-align: center; font-weight: 500; }
.chart-container { background-color: #FFFFFF; padding: 1.5rem; border-radius: 16px; border: 1px solid #DDEBE3; box-shadow: 0 4px 12px rgba(20, 90, 50, 0.08); margin-top: 1rem; }
.metric-box { background: #FFFFFF; padding: 1.6rem; border-radius: 18px; border: 2px solid #145A32; margin-bottom: 1.5rem; text-align: center; }
.metric-title { font-size: 0.82rem; text-transform: uppercase; color: #145A32; font-weight: 800; margin-bottom: 0.5rem; }
.metric-value { font-size: 2rem; font-weight: 700; }
textarea { border-radius: 12px !important; border: 1px solid #145A32 !important; background-color: #FFFFFF !important; color: #1A3C2E !important; }
section[data-testid="stSidebar"] { background-color: #E8F3EF; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">MindContext AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-container"><div class="subtitle">Advanced Text-Based Mental Health Context Classifier</div></div>', unsafe_allow_html=True)

# 3. Model Loading
@st.cache_resource
def load_pipeline():
    model_path = "ourafla/mental-health-bert-finetuned" 
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=4, ignore_mismatched_sizes=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_pipeline()

# Theme Mapping
theme_colors = {
    'Anxiety': '#D08C3F',    # Muted Amber
    'Depression': '#6B6F8A', # Slate Blue
    'Normal': '#4E8B70',     # Sage Green
    'Suicidal': '#B22222'    # Deep Red
}

st.markdown("<h4 style='color:#145A32;'>Enter text context below to compute deep learning inferences:</h4>", unsafe_allow_html=True)
user_input = st.text_area("", placeholder="Type statements here...", height=160)

if st.button("Run Live Prediction Analytics ", type="primary", use_container_width=True):
    if user_input.strip() == "":
        st.warning("Input text cannot be left blank.")
    else:
        with st.spinner("Processing..."):
            class_labels = ['Anxiety', 'Depression', 'Normal', 'Suicidal']
            inputs = tokenizer(user_input, truncation=True, padding='max_length', max_length=256, return_tensors="pt").to(device)
            with torch.no_grad():
                probs = F.softmax(model(**inputs).logits, dim=-1).squeeze().cpu().numpy()
            
            pred_idx = probs.argmax()
            pred_label = class_labels[pred_idx]
            pred_color = theme_colors[pred_label]
            
            # Display Metric
            st.markdown(f"""
                <div class="metric-box" style="border-left: 10px solid {pred_color};">
                    <div class="metric-title">Identified Primary Psychological Context</div>
                    <div class="metric-value" style="color: {pred_color};">{pred_label} ({probs[pred_idx]*100:.1f}%)</div>
                </div>
            """, unsafe_allow_html=True)
            
            # Chart with Distinct Colors per Bar
            bar_colors = [theme_colors[lbl] for lbl in class_labels]
            
            fig = go.Figure(go.Bar(
                x=probs * 100, y=class_labels, orientation='h',
                marker_color=bar_colors,
                text=[f"{p*100:.1f}%" for p in probs],
                textposition='auto',
                textfont=dict(color='#333333', size=13, weight='bold')
            ))
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=True, gridcolor='#DDEBE3', tickfont=dict(color='#145A32')),
                yaxis=dict(tickfont=dict(color='#145A32', size=14, weight='bold')),
                margin=dict(l=20, r=20, t=10, b=10), height=250
            )
            
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("<p style='color:#145A32; font-weight:bold;'>Confidence Distribution Array Matrix</p>", unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
