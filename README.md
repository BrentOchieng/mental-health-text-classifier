# Mental Health Text Classifier (BERT-Powered NLP Framework)

An end-to-end Deep Learning and Natural Language Processing (NLP) application engineered to classify textual patterns into four distinct psychological categories: **Anxiety, Depression, Normal, and Suicidal Intent**. This project pairs an advanced transformer model architecture with a live, responsive analytics dashboard for predictive risk assessment.

**Live Interactive App:** https://mental-health-text-classifier-84jlv6lc3mijccrofcx9vb.streamlit.app/  
**Model weights served via:** Hugging Face Hub

---

## Project Overview & Problem Statement

### The Problem
Mental health challenges are rising globally, yet traditional screening methods often rely on manual, reactive evaluations that delay critical intervention. With millions of individuals expressing psychological distress daily through text sequences such as social media posts, support forums, and text lines, there is a critical bottleneck in triaging these inputs swiftly and identifying high-risk individuals before a crisis escalates.

### The Solution
This project bridges the gap between deep learning research and production-grade software. It provides an automated, real-time textual classification framework capable of handling unstructured text, analyzing emotional underlying context, and identifying potential distress signals with probabilistic tracking. 

By deploying this as an active MLOps pipeline, it provides:
* **Immediate Risk Stratification:** Instantly flagging clinical indicators for Anxiety or Depression.
* **Deterministic Crisis Redirection:** Providing an architectural safety rail that bypasses statistical predictions when acute keywords are triggered, immediately redirecting users to emergency resources.

---

## Tech Stack & System Architecture

The application is built using a decoupled, production-optimized data science architecture to minimize memory footprint and keep cloud runtime under free-tier execution limits.

* **Core Deep Learning Framework:** `PyTorch` & `Hugging Face Transformers`
* **Model Engine:** `BERT-base-uncased` fine-tuned framework specialized for psychological text analysis
* **Production UI & Serving Engine:** `Streamlit`
* **Data Visualization:** `Plotly Open Source` (Interactive Analytics Dashboards)

[ Unstructured Input Text ]
│
▼
┌────────────────────────────────────────────────────────┐
│             Streamlit Web Interface / UI               │
└────────────────────────────────────────────────────────┘
│
▼
┌────────────────────────────────────────────────────────┐
│      1. Regex Keyword Safety Bypass Rule               │───[High Risk Triggered]───► [Immediate Emergency UI Redirect]
└────────────────────────────────────────────────────────┘
│
│ [Low/Moderate Risk]
▼
┌────────────────────────────────────────────────────────┐
│  2. Cloud-Linked BERT Transformer Inference Pipeline  │◄─── (Streams weights dynamically from Hugging Face)
└────────────────────────────────────────────────────────┘
│
▼
┌────────────────────────────────────────────────────────┐
│        3. Plotly Reactive Analytics Visuals            │
└────────────────────────────────────────────────────────┘

---

## Key Engineering & MLOps Features

### 1. Cloud-Linked Weight Decoupling
To circumvent GitHub’s 100MB file limit and avoid the infrastructure overhead of local Git LFS configurations, the system uses a decoupled MLOps strategy. The application logic and web environment are served seamlessly via GitHub/Streamlit, while the heavy **440MB transformer weights** are dynamically streamed and cached on-demand from the Hugging Face model ecosystem, cutting cold-start deployment runtimes down significantly.

### 2. Built-in Deterministic Safety Safeguards
Purely statistical models can occasionally fall victim to edge-case misclassifications. To make this tool enterprise-ready and safe, a deterministic keyword override framework is layered over the prediction pipeline. If specific, critical emergency terms are parsed, the system automatically short-circuits model inference and enforces an instant, clean alert redirection to professional help lines.

### 3. Granular Probability Analytics Dashboard
Instead of spitting out a rigid, single text category, the application processes raw classification logits through a softmax layer to yield an interactive multi-class probability distribution. Rendered via Plotly, this gives clinicians or reviewers clear visual insight into the model's confidence scores across all four classes simultaneously.

---

## Repository Structure

* **`app.py`** – The complete production-grade application code containing the Streamlit layout, safety logic, and cached prediction pipeline.
* **`requirements.txt`** – Cloud environment specifications detailing precise package dependencies (`torch`, `transformers`, etc.).
* **`*.ipynb`** – The research and development sandbox notebook detailing exploratory data analysis (EDA), fine-tuning runs, and early model validation.

---

## Local Setup Instructions

To spin up this repository and host the dashboard on your local machine:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/BrentOchieng/mental-health-text-classifier.git](https://github.com/BrentOchieng/mental-health-text-classifier.git)
   cd mental-health-text-classifier

 2. **Install the dependencies:**
  
  pip install -r requirements.txt

  3. **Launch the local Streamlit host:**
     
  streamlit run app.py

---
  Disclaimer: This application is built as a portfolio showcase demonstrating the software implementation of Natural Language Processing and deep learning workflows. It is intended for informational and screening support pipelines and does not substitute for clinical medical advice.
