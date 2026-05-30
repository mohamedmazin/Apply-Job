import ast
import pickle
import re
import os
import numpy as np
import pandas as pd
import streamlit as st
import fitz
import nltk
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import xgboost as xgb

# Download NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# --- 1. Load Pre-trained Models ---
@st.cache_resource
def load_models():
    if os.path.exists('super_stacking_model.pkl'):
        with open('super_stacking_model.pkl', 'rb') as f:
            model = pickle.load(f)
    else:
        st.error("Super model file 'super_stacking_model.pkl' not found.")
        model = None

    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    return model, sbert_model

model, sbert_model = load_models()

if 'applicants_list' not in st.session_state:
    st.session_state['applicants_list'] = []

TRAINED_JOB_CATEGORIES = [
    'AI Engineer', 'Asst. Manager/ Manger (Administrative)', 'Business Development Executive', 
    'Civil Engineer', 'Data Engineer', 'Data Science Engineer', 'Database Administrator (DBA)', 
    'DevOps Engineer', 'Executive - VAT', 'Executive/ Senior Executive- Trade Marketing, Hygiene Products', 
    'Executive/ Sr. Executive -IT', 'Full Stack Developer (Python,React js)', 'HR Officer', 
    'Head of Internal Control & Compliance (ICC) - SEVP/DMD', 'Intern (Generative AI Engineering - 2D/3D Image Generation)', 
    'Machine Learning (ML) Engineer', 'Management Trainee - Mechanical', 'Manager- Human Resource Management (HRM)', 
    'Marketing Officer', 'Mechanical Designer', 'Mechanical Engineer', 'Network Support Engineer', 
    'Project Coordinator (Civil)', 'Senior Software Engineer', 'Senior iOS Engineer', 'Site Engineer', 
    'Sr.Officer / Executive - Internal Audit', 'System Administrator (Operation & Maintenance of Server, Storage & Service Desk System)'
]

TECH_JOB_MAPPING = {
    "Frontend Developer": "Senior Software Engineer",
    "Backend Developer": "Senior Software Engineer",
    "Mobile Developer (Android/iOS)": "Senior iOS Engineer",
    "Flutter Developer": "Senior iOS Engineer",
    "React Native Developer": "Senior iOS Engineer",
    "Software Engineer": "Senior Software Engineer",
    "Web Developer": "Full Stack Developer (Python,React js)",
    "QA Engineer": "Senior Software Engineer",
    "Cybersecurity Specialist": "Network Support Engineer",
    "Cloud Architect": "DevOps Engineer",
    "Embedded Systems Engineer": "Mechanical Engineer"
}

# --- 2. Utility Functions ---
def deep_clean_text(text):
    if pd.isna(text) or text == "": return ""
    text = str(text).replace('[', '').replace(']', '').replace("'", "").replace('"', '')
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    words = text.split()
    clean_words = [w for w in words if w not in stop_words]
    return " ".join(clean_words)

def extract_full_cv_data(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = " ".join([page.get_text() for page in doc])
    current_year = 2026

    all_years = sorted(list(set([int(y) for y in re.findall(r'\b(19[7-9][0-9]|20[0-2][0-9])\b', text)])))
    final_age = 25
    if all_years:
        final_age = (current_year - all_years[0]) if (current_year - all_years[0]) > 18 else (current_year - all_years[0] + 22)
    exp_matches = re.findall(r'(\d+)\s?\+?\s?years?\s?(?:of\s?)?(?:experience|exp)', text, re.I)
    final_exp = int(exp_matches[0]) if exp_matches else (max(all_years) - min(all_years) if len(all_years) > 1 else 0)
    
    addr_match = re.search(r'(?:Address|Location|Lives in)\s?[:\-]?\s*(.*)', text, re.I)
    address = addr_match.group(1).split('\n')[0].strip() if addr_match else "Not explicitly found"

    def get_section(patterns):
        match = re.search(rf"(?:{'|'.join(patterns)})\s?[:\-]?\n?(.*?)(?:\n\n|Education|Skills|Experience|Contact|$)", text, re.I | re.DOTALL)
        return match.group(1).strip() if match else "Information not found"

    return {
        "candidate_name": pdf_file.name.replace(".pdf", ""),
        "age": final_age,
        "experiencere": final_exp,
        "address": address,
        "skills": get_section(["Skills", "Technologies", "Technical Skills"]),
        "education": get_section(["Education", "Academic", "Qualifications"]),
        "positions": get_section(["Experience", "Work History", "Professional Experience"]),
        "full_text": text
    }

# --- 3. Streamlit UI ---
st.set_page_config(page_title="AI Hiring Platform v4.0", layout="wide", page_icon="🚀")

# CSS
st.markdown("""
    <style>
    .main { background-color: #f4f7f9; }
    .score-container { text-align: center; padding: 30px; border-radius: 50%; background: #ffffff; border: 10px solid #4CAF50; width: 180px; height: 180px; display: flex; flex-direction: column; align-items: center; justify-content: center; margin: 20px auto; box-shadow: 0 8px 20px rgba(76,175,80,0.3); }
    .score-container h2 { color: #2e7d32; margin: 0; font-size: 2.5em; }
    .applicant-row { background: white; padding: 15px; border-radius: 8px; border-left: 8px solid #4CAF50; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
    .badge { padding: 4px 10px; border-radius: 12px; font-size: 0.8em; font-weight: bold; margin-left: 5px; }
    .badge-high { background-color: #d4edda; color: #155724; }
    .badge-low { background-color: #f8d7da; color: #721c24; }
    .extracted-box { background: #fffbeb; border: 1px solid #fbbf24; padding: 15px; border-radius: 8px; margin-bottom: 15px; }
    .extracted-label { font-weight: bold; color: #92400e; margin-right: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 AI-Powered Hiring Platform (Backend Integration Ready)")
st.info("⚡ Model Accuracy: 83% | Language Feature Removed | Full control over extracted data")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📝 Job Setup", "🔍 CV Upload & Apply", "🏆 Applicants Ranking"])

with tab1:
    st.header("1. Define Job Requirements")
    col1, col2 = st.columns(2)
    with col1:
        available_roles = sorted(list(set(TRAINED_JOB_CATEGORIES + list(TECH_JOB_MAPPING.keys()))))
        j_title = st.selectbox("Job Title / Position", available_roles, index=available_roles.index("Software Engineer") if "Software Engineer" in available_roles else 0)
        j_skills = st.text_area("Required Skills (comma separated)", "Android, Kotlin, Java, Mobile, Firebase")
        j_loc = st.text_input("Target Location", "Alabama, USA")
    with col2:
        j_exp = st.number_input("Min Experience", 0, 20, 2)
        j_age = st.number_input("Target Age", 18, 60, 25)
    j_desc = st.text_area("Full Description", "Looking for an Android developer...")

    if st.button("Save & Post Job"):
        st.session_state['job_data'] = {"title": j_title, "exp": j_exp, "age": j_age, "desc": j_desc, "skills": j_skills, "loc": j_loc}
        st.session_state['applicants_list'] = [] 
        st.success(f"Job '{j_title}' posted!")

with tab2:
    if 'job_data' not in st.session_state:
        st.warning("Please setup job details first.")
    else:
        st.header("2. Candidate Application")
        uploaded_file = st.file_uploader("Upload Your Resume (PDF)", type="pdf")
        
        if uploaded_file:
            # We only extract once per upload to avoid overwriting edits
            if 'last_uploaded' not in st.session_state or st.session_state['last_uploaded'] != uploaded_file.name:
                cv = extract_full_cv_data(uploaded_file)
                st.session_state['cv_extracted'] = cv
                st.session_state['last_uploaded'] = uploaded_file.name
            
            cv = st.session_state['cv_extracted']
            
            st.markdown("### 📋 Step 1: Review & Edit Extracted Data")
            st.caption("We've extracted this from your CV. You can edit any field before submitting.")
            
            # Display extracted data first
            st.markdown(f"""
                <div class="extracted-box">
                    <p><span class="extracted-label">📍 Address:</span> {cv['address']}</p>
                    <p><span class="extracted-label">📜 Skills Extracted:</span> {cv['skills'][:150]}...</p>
                    <p><span class="extracted-label">🎓 Education:</span> {cv['education'][:100]}...</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### ✏️ Edit Your Data")
            col_edit1, col_edit2 = st.columns(2)
            with col_edit1:
                final_name = st.text_input("Full Name", value=cv['candidate_name'])
                final_addr_in = st.text_input("Current Address", value=cv['address'])
                final_age_in = st.number_input("Your Age", value=int(cv['age']), min_value=18)
            with col_edit2:
                final_exp_in = st.number_input("Years of Experience", value=int(cv['experiencere']), min_value=0)
                final_skills_txt = st.text_area("Skills (Editable - Comma Separated)", value=cv['skills'], height=100)
                final_edu_txt = st.text_area("Education Details (Editable)", value=cv['education'], height=100)

            if st.button("🚀 Apply with AI Evaluation"):
                with st.spinner("Analyzing your edited data..."):
                    # 1. Similarity Calculation
                    job_text = deep_clean_text(st.session_state['job_data']['title'] + " " + st.session_state['job_data']['desc'] + " " + st.session_state['job_data']['skills'])
                    cv_text_combined = deep_clean_text(final_name + " " + final_skills_txt + " " + final_edu_txt + " " + cv['full_text'])
                    
                    cv_emb = sbert_model.encode(cv_text_combined, convert_to_tensor=True)
                    job_emb = sbert_model.encode(job_text, convert_to_tensor=True)
                    semantic_sim = util.cos_sim(cv_emb, job_emb).item()

                    # 2. Strict Title Keyword Match
                    target_role = st.session_state['job_data']['title'].lower()
                    clean_title = re.sub(r'[\(\)/]', ' ', target_role)
                    title_keywords = [w for w in clean_title.split() if len(w) > 2 and w not in ["and", "the", "for"]]
                    
                    title_match_count = sum(1 for kw in title_keywords if kw in cv['full_text'].lower() or kw in final_skills_txt.lower())
                    title_match_ratio = title_match_count / len(title_keywords) if title_keywords else 1.0
                    
                    # 3. Skill Match
                    req_skills = set([s.strip().lower() for s in st.session_state['job_data']['skills'].split(',') if s.strip()])
                    cv_skills_set = set(deep_clean_text(final_skills_txt).split())
                    skill_ratio = len(cv_skills_set.intersection(req_skills)) / len(req_skills) if req_skills else 0.5

                    # 4. Predict with Model
                    selected_title = st.session_state['job_data']['title']
                    mapped_title = TECH_JOB_MAPPING.get(selected_title, selected_title)
                    job_cat = TRAINED_JOB_CATEGORIES.index(mapped_title) if mapped_title in TRAINED_JOB_CATEGORIES else 23

                    features = np.array([[
                        semantic_sim, skill_ratio, title_match_ratio, 
                        float(final_age_in), float(final_exp_in), 
                        float(final_exp_in - st.session_state['job_data']['exp']),
                        float(final_age_in - st.session_state['job_data']['age']),
                        float(job_cat)
                    ]])
                    
                    model_score = model.predict(features)[0]

                    # Domain Logic (NO LANGUAGE BONUS ANYMORE!)
                    domain_bonus = 0.2 if title_match_ratio > 0.6 else 0.0
                    skill_bonus = 0.2 if skill_ratio > 0.7 else 0.0
                    domain_penalty = 0.4 if title_match_ratio < 0.4 else 0.0
                    
                    final_score_raw = (model_score * 0.4) + (skill_ratio * 0.3) + (title_match_ratio * 0.3) + domain_bonus + skill_bonus - domain_penalty
                    final_pct = max(0, min(100.0, final_score_raw * 100))

                    st.session_state['applicants_list'].append({
                        "name": final_name,
                        "score": final_pct,
                        "skill_ratio": skill_ratio,
                        "title_match": title_match_ratio,
                        "semantic_sim": semantic_sim,
                        "location": final_addr_in,
                        "exp": final_exp_in
                    })
                    
                    st.balloons()
                    st.success("Applied!")
                    st.markdown(f"<div class='score-container'><h2>{final_pct:.1f}%</h2></div>", unsafe_allow_html=True)

with tab3:
    st.header("3. Ranked Applicants")
    if not st.session_state['applicants_list']:
        st.info("No applicants yet.")
    else:
        ranked = sorted(st.session_state['applicants_list'], key=lambda x: x['score'], reverse=True)
        for i, app in enumerate(ranked):
            medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "👤"
            skill_class = "badge-high" if app['skill_ratio'] > 0.7 else "badge-low"
            st.markdown(f"""
            <div class="applicant-row">
                <div style="flex: 2;">
                    <b>{medal} {app['name']}</b>
                    <span class="badge {skill_class}">Skills: {app['skill_ratio']*100:.0f}%</span>
                </div>
                <div style="flex: 3; color: #666; font-size: 0.85em;">
                    Semantic Fit: {app['semantic_sim']*100:.1f}% | Title Match: {app['title_match']*100:.0f}% | Experience: {app['exp']} yrs | Location: {app['location']}
                </div>
                <div style="flex: 1; text-align: right; color: #2e7d32; font-weight: bold; font-size: 1.2em;">
                    {app['score']:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
        if st.button("Reset All Applicants"):
            st.session_state['applicants_list'] = []
            st.rerun()
