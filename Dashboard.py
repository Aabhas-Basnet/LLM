import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import pdfplumber
import time
from google import genai
from dotenv import load_dotenv

# ==========================================
# 1. SECURE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Career Intel Pro", layout="wide")

# Load local .env (ignored on GitHub)
load_dotenv()

# Secure Key Fetch
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("Missing API Key. Add 'GEMINI_API_KEY' to Streamlit Secrets or your local .env file.")
    st.stop()

client = genai.Client(api_key=API_KEY)

# ==========================================
# 2. ANALYSIS ENGINE (NLP-Lite Version)
# ==========================================
def get_analysis(text, role, loc):
    # We send the raw text directly to Gemini 1.5 Pro. 
    # It is powerful enough to extract its own entities/skills.
    prompt = f"""
    ROLE: Senior Technical Recruiter & Market Analyst.
    LOCATION: {loc}.
    TASK: Perform a Career Gap Analysis for a {role} candidate.
    
    CANDIDATE CV DATA:
    \"\"\"{text[:4000]}\"\"\"

    INSTRUCTIONS:
    1. Identify existing skills and find 3 ADVANCED technical gaps specific to {loc}.
    2. Calculate an ATS match score (0-100).
    3. Generate a 2-sentence executive summary.
    4. List 3 target companies in {loc} and 3 relevant networking events for April-May 2026.

    OUTPUT JSON FORMAT:
    {{
      "summary": "Professional review.",
      "ats_score": 85,
      "gaps": ["Gap 1", "Gap 2", "Gap 3"],
      "jobs": ["Role 1", "Role 2", "Role 3"],
      "orgs": ["Local Org 1", "Local Org 2", "Local Org 3"],
      "spiral": {{"Technical": 8, "Narrative": 7, "Market": 9, "Data Storytelling": 8, "Academic": 9}},
      "roadmap": ["Step 1", "Step 2", "Step 3"],
      "events": ["Event | Venue | Date"]
    }}
    """
    
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=prompt,
                config={'response_mime_type': 'application/json', 'temperature': 0.1}
            )
            return json.loads(response.text)
        except Exception as e:
            if ("503" in str(e) or "429" in str(e)) and attempt < 2:
                time.sleep(3)
                continue
            else:
                st.error(f"Analysis Engine Offline: {e}")
                return None

# ==========================================
# 3. UI ARCHITECTURE
# ==========================================
st.markdown("""
<style>
    .card { padding: 15px; border-radius: 8px; margin-bottom: 10px; color: #1a1a1a; font-size: 0.85rem; border-left: 5px solid; }
    .gap { background: #fff5f5; border-color: #e53e3e; }
    .job { background: #f0fff4; border-color: #38a169; }
    .org { background: #ebf8ff; border-color: #3182ce; }
    .stButton>button { background-color: #2b6cb0; color: white; width: 100%; border: none; font-weight: bold; height: 3em; }
    .stButton>button:hover { background-color: #1a4971; color: white; }
</style>
""", unsafe_allow_html=True)

st.title("Career Intelligence Dashboard")
st.caption("Strategic Market Alignment Analysis | 2026 Edition")

if "data" not in st.session_state: 
    st.session_state.data = None

# Input Layer
c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    role_input = st.text_input("Target Position", value="Data Business Analyst")
with c2:
    loc_input = st.selectbox("Market Focus", ["Adelaide, SA", "Sydney, NSW", "Melbourne, VIC", "Brisbane, QLD", "Perth, WA", "Canberra, ACT"])
with c3:
    file = st.file_uploader("Upload CV (PDF)", type="pdf")

if file and st.button("Generate Professional Audit"):
    with st.spinner("Synthesizing Market Data..."):
        try:
            with pdfplumber.open(file) as pdf:
                txt = "\n".join([p.extract_text() for p in pdf.pages if p.extract_text()])
            st.session_state.data = get_analysis(txt, role_input, loc_input)
        except Exception as e:
            st.error(f"File Processing Error: {e}")

# Results Layer
if st.session_state.data:
    d = st.session_state.data
    st.divider()
    
    col_l, col_r = st.columns([2, 1])
    with col_l:
        st.subheader("Executive Review")
        st.write(d['summary'])
    with col_r:
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = d['ats_score'],
            title = {'text': "ATS Match Score", 'font': {'size': 16}},
            gauge = {'bar': {'color': "#2b6cb0"}, 'axis': {'range': [0, 100]}, 'bgcolor': "white"}
        ))
        fig_gauge.update_layout(height=200, margin=dict(t=40, b=0, l=10, r=10), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.divider()
    m1, m2, m3 = st.columns(3)
    
    sections = [('gaps', 'gap', 'Market Gaps', m1), 
                ('jobs', 'job', 'Role Matches', m2), 
                ('orgs', 'org', 'Local Targets', m3)]
    
    for key, css, label, col in sections:
        col.markdown(f"**{label}**")
        for val in d[key]:
            col.markdown(f"<div class='card {css}'>{val}</div>", unsafe_allow_html=True)

    st.divider()
    v1, v2 = st.columns(2)
    with v1:
        st.subheader("Professional Aptitude")
        rad_df = pd.DataFrame(dict(r=list(d['spiral'].values()), theta=list(d['spiral'].keys())))
        fig_rad = px.line_polar(rad_df, r='r', theta='theta', line_close=True)
        fig_rad.update_traces(fill='toself', fillcolor='rgba(43, 108, 176, 0.2)', line_color='#2b6cb0')
        fig_rad.update_layout(paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_rad, use_container_width=True)
    with v2:
        st.subheader("Strategic Roadmap")
        for step in d['roadmap']: st.write(f"🔹 {step}")
        st.subheader("Upcoming Events")
        for ev in d['events']: st.info(ev)