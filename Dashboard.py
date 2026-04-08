import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import spacy
import json
import os
import pdfplumber
import time
from google import genai
from dotenv import load_dotenv

# ==========================================
# 1. SETUP & CORE CONFIG
# ==========================================
load_dotenv()

# 2. Fetch the key from either .env or Streamlit Secrets
# This replaces: API_KEY = "YOUR_ACTUAL_KEY_HERE"
API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")

if not API_KEY:
    st.error("Missing API Key. Please add it to your .env file or Streamlit Secrets.")
    st.stop()

client = genai.Client(api_key=API_KEY)

@st.cache_resource
def load_nlp():
    try:
        return spacy.load("en_core_web_sm")
    except:
        os.system("python -m spacy download en_core_web_sm")
        return spacy.load("en_core_web_sm")

nlp = load_nlp()

# ==========================================
# 2. UPDATED ANALYSIS ENGINE (v1.5 Pro)
# ==========================================
def get_analysis(text, role, loc):
    # NLP Pre-processing to save tokens
    doc = nlp(text[:3500])
    skills = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT"]][:12]
    
    prompt = f"""
    Analyze this CV for a {role} position in {loc}. 
    Known Skills: {skills}. 
    Provide a professional gap analysis in JSON format:
    {{
      "summary": "2-sentence executive overview",
      "ats_score": 85,
      "gaps": ["Technical Gap 1", "Technical Gap 2", "Technical Gap 3"],
      "jobs": ["Target Job 1", "Target Job 2", "Target Job 3"],
      "orgs": ["Local Company 1", "Local Company 2", "Local Company 3"],
      "spiral": {{"Technical": 8, "Narrative": 7, "Market": 9, "Data Storytelling": 8, "Academic": 9}},
      "roadmap": ["Step 1", "Step 2", "Step 3"],
      "events": ["Local Event | Venue | Date"]
    }}
    """
    
    # Retry Logic for 503 and Model Availability
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-pro", # Updated to stable high-tier model
                contents=prompt,
                config={'response_mime_type': 'application/json', 'temperature': 0.1}
            )
            return json.loads(response.text)
        except Exception as e:
            if ("503" in str(e) or "429" in str(e)) and attempt < 2:
                time.sleep(3) 
                continue
            else:
                st.error(f"AI Service Error: {e}")
                return None

# ==========================================
# 3. UI & STYLING
# ==========================================
st.markdown("""
<style>
    .card { padding: 15px; border-radius: 8px; margin-bottom: 10px; color: #1a1a1a; font-size: 0.85rem; border-left: 5px solid; }
    .gap { background: #fff5f5; border-color: #e53e3e; }
    .job { background: #f0fff4; border-color: #38a169; }
    .org { background: #ebf8ff; border-color: #3182ce; }
    .stButton>button { background-color: #2b6cb0; color: white; width: 100%; border: none; }
</style>
""", unsafe_allow_html=True)

st.title("Career Intelligence Dashboard")
if "data" not in st.session_state: st.session_state.data = None

# Input Section
c1, c2, c3 = st.columns([2, 1, 1])
role = c1.text_input("Target Role", value="Data Business Analyst")
loc = c2.selectbox("Market", ["Adelaide, SA", "Sydney, NSW", "Melbourne, VIC", "Brisbane, QLD", "Perth, WA"])
file = c3.file_uploader("Upload CV (PDF)", type="pdf")

if file and st.button("Generate Audit"):
    with st.spinner("Syncing Market Intelligence..."):
        with pdfplumber.open(file) as pdf:
            txt = "\n".join([p.extract_text() for p in pdf.pages if p.extract_text()])
        st.session_state.data = get_analysis(txt, role, loc)

# Results Section
if st.session_state.data:
    d = st.session_state.data
    st.divider()
    
    col_l, col_r = st.columns([2, 1])
    with col_l:
        st.subheader("Executive Review")
        st.write(d['summary'])
    with col_r:
        fig = go.Figure(go.Indicator(mode="gauge+number", value=d['ats_score'], 
                        title={'text': "ATS Score", 'font': {'size': 16}},
                        gauge={'bar':{'color':"#2b6cb0"}, 'axis':{'range':[0,100]}}))
        fig.update_layout(height=200, margin=dict(t=40, b=0), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    m1, m2, m3 = st.columns(3)
    # Refined loop for cleaner UI
    sections = [('gaps', 'gap', 'Technical Gaps', m1), 
                ('jobs', 'job', 'Career Path', m2), 
                ('orgs', 'org', 'Target Orgs', m3)]
    
    for key, css, label, col in sections:
        col.markdown(f"**{label}**")
        for val in d[key]:
            col.markdown(f"<div class='card {css}'>{val}</div>", unsafe_allow_html=True)

    st.divider()
    v1, v2 = st.columns(2)
    with v1:
        st.subheader("Aptitude Profile")
        rad_df = pd.DataFrame(dict(r=list(d['spiral'].values()), theta=list(d['spiral'].keys())))
        fig_rad = px.line_polar(rad_df, r='r', theta='theta', line_close=True)
        fig_rad.update_traces(fill='toself', fillcolor='rgba(43, 108, 176, 0.2)', line_color='#2b6cb0')
        st.plotly_chart(fig_rad, use_container_width=True)
    with v2:
        st.subheader("Execution Roadmap")
        for step in d['roadmap']: st.write(f"🔹 {step}")
        st.subheader("Regional Networking")
        for ev in d['events']: st.success(ev)