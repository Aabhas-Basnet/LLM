import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import spacy
import json
import os
import pdfplumber
from google import genai
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

# ==========================================
# 1. CORE CONFIGURATION
# ==========================================
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("API Key not found. Verify .env file.")
    st.stop()

client = genai.Client(api_key=API_KEY)

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# ==========================================
# 2. RESILIENCE ENGINE (Elite Tier)
# ==========================================
def is_retryable(e):
    return "503" in str(e) or "429" in str(e)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=6),
    retry=retry_if_exception(is_retryable)
)
def fetch_ai_intelligence(prompt):
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=prompt,
            config={'response_mime_type': 'application/json', 'temperature': 0.0}
        )
        return json.loads(response.text)
    except Exception:
        response = client.models.generate_content(
            model="gemini-3-flash-preview", 
            contents=prompt,
            config={'response_mime_type': 'application/json', 'temperature': 0.0}
        )
        return json.loads(response.text)

# ==========================================
# 3. ANALYSIS LOGIC
# ==========================================
def process_career_data(cv_text, target_role, location):
    doc = nlp(cv_text)
    existing_skills = list(set([ent.text for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT"]]))
    
    prompt = f"""
    ROLE: Senior Technical Recruiter. LOCATION: {location}. DATE: April 2026.
    TASK: Perform a Gap Analysis for {target_role}.
    
    CONTEXT:
    - User Existing Skills: {existing_skills[:15]}
    - CV Text: {cv_text[:4000]}

    STRICT INSTRUCTIONS:
    1. Identify 3 ADVANCED technical gaps (Cloud, MLOps, Governance) specific to {location}.
    2. Replace Leadership with Data Storytelling in the spiral.
    3. Calculate a consistent ATS Score (0-100) based on professional match.
    4. Provide 3 Networking events in {location} for April-May 2026.

    OUTPUT JSON:
    {{
      "summary": "Professional review.",
      "ats_score": 85,
      "metrics": {{
          "top_skills_gaps": ["Gap 1", "Gap 2", "Gap 3"],
          "top_jobs": ["Job 1", "Job 2", "Job 3"],
          "top_companies": ["Company 1", "Company 2", "Company 3"]
      }},
      "spiral": {{
         "Technical Aptitude": 8,
         "CV Narrative": 7,
         "Market Fit": 9,
         "Data Storytelling": 8,
         "Academic Depth": 9
      }},
      "roadmap_paragraph": "Strategic roadmap.",
      "roadmap_list": ["Action 1", "Action 2", "Action 3"],
      "networking": ["Event | Venue | Date"]
    }}
    """
    return fetch_ai_intelligence(prompt)

# ==========================================
# 4. MAIN UI
# ==========================================
st.set_page_config(page_title="Career Intelligence Pro", layout="wide")

# Persistent Session State
if "audit_data" not in st.session_state:
    st.session_state.audit_data = None

# Professional Styling
st.markdown("""
<style>
    .gap-card { background-color: #fce4ec; border-left: 5px solid #d81b60; padding: 15px; border-radius: 8px; margin-bottom: 10px; color: #1a1a1a; }
    .job-card { background-color: #e8f5e9; border-left: 5px solid #2e7d32; padding: 15px; border-radius: 8px; margin-bottom: 10px; color: #1a1a1a; }
    .org-card { background-color: #e3f2fd; border-left: 5px solid #1565c0; padding: 15px; border-radius: 8px; margin-bottom: 10px; color: #1a1a1a; }
    .header-tag { font-weight: bold; margin-bottom: 5px; text-transform: uppercase; font-size: 0.75rem; color: #444; }
    .stButton>button { background-color: #1565c0; color: white; border-radius: 5px; width: 100%; }
</style>
""", unsafe_allow_html=True)

st.title("Career Intelligence Dashboard")
st.text("Strategic Market Alignment Analysis")

# Inputs
col_in1, col_in2, col_in3 = st.columns([2, 1, 1])
with col_in1:
    target_role = st.text_input("Target Position", value="Data Business Analyst")
with col_in2:
    location = st.selectbox("Australian Market", [
        "Adelaide, SA", "Sydney, NSW", "Melbourne, VIC", 
        "Brisbane, QLD", "Perth, WA", "Hobart, TAS", 
        "Darwin, NT", "Canberra, ACT"
    ])
with col_in3:
    uploaded_file = st.file_uploader("Upload CV (PDF)", type="pdf")

if uploaded_file and target_role:
    if st.button("Execute Professional Audit"):
        with st.spinner("Synthesizing market vectors"):
            try:
                with pdfplumber.open(uploaded_file) as pdf:
                    text = "\n".join([p.extract_text() for p in pdf.pages if p.extract_text()])
                st.session_state.audit_data = process_career_data(text, target_role, location)
            except Exception as e:
                st.error(f"Error: {e}")

# Persistent Result Display
if st.session_state.audit_data:
    data = st.session_state.audit_data
    
    st.divider()
    rev_left, rev_right = st.columns([2, 1])
    
    with rev_left:
        st.subheader("Executive Review")
        st.write(data['summary'])
    
    with rev_right:
        # Theme-aware Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = data['ats_score'],
            title = {'text': "ATS Match Score", 'font': {'size': 20}},
            gauge = {
                'bar': {'color': "#1565c0"},
                'axis': {'range': [0, 100]},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#444"
            }
        ))
        fig_gauge.update_layout(height=250, margin=dict(t=50, b=0, l=10, r=10), paper_bgcolor='rgba(0,0,0,0)', font={'color': "gray"})
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.divider()
    m1, m2, m3 = st.columns(3)
    
    with m1:
        st.markdown("<h4 style='color:#d81b60'>Technical Gaps</h4>", unsafe_allow_html=True)
        for s in data['metrics']['top_skills_gaps']:
            st.markdown(f"<div class='gap-card'><div class='header-tag'>Market Gap</div>{s}</div>", unsafe_allow_html=True)
    
    with m2:
        st.markdown("<h4 style='color:#2e7d32'>Job Matches</h4>", unsafe_allow_html=True)
        for j in data['metrics']['top_jobs']:
            st.markdown(f"<div class='job-card'><div class='header-tag'>Role</div>{j}</div>", unsafe_allow_html=True)
    
    with m3:
        st.markdown("<h4 style='color:#1565c0'>Local Organizations</h4>", unsafe_allow_html=True)
        for c in data['metrics']['top_companies']:
            st.markdown(f"<div class='org-card'><div class='header-tag'>Target</div>{c}</div>", unsafe_allow_html=True)

    st.divider()
    v_left, v_right = st.columns([1, 1])
    
    with v_left:
        st.subheader("Professional Aptitude Spiral")
        df_radar = pd.DataFrame(dict(r=list(data['spiral'].values()), theta=list(data['spiral'].keys())))
        fig_radar = px.line_polar(df_radar, r='r', theta='theta', line_close=True)
        fig_radar.update_traces(fill='toself', line_color='#00e676', fillcolor='rgba(0, 230, 118, 0.3)')
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10], gridcolor="#888")), paper_bgcolor='rgba(0,0,0,0)', font={'color': "gray"})
        st.plotly_chart(fig_radar, use_container_width=True)

    with v_right:
        st.subheader("Development Roadmap")
        st.info(data['roadmap_paragraph'])
        for item in data['roadmap_list']:
            st.write(f"- {item}")
        st.subheader("Regional Networking")
        for event in data['networking']:
            st.success(event)