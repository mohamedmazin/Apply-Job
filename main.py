import ast
import pickle
import re
import os
import numpy as np
import pandas as pd
import fitz
import nltk
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import io

# Download NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

app = FastAPI(title="AI Hiring Platform API", version="1.0")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Variables ---
MODEL = None
SBERT_MODEL = None
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

# --- Pydantic Models for Request/Response ---
class CandidateData(BaseModel):
    candidate_name: str
    age: int
    experiencere: int
    address: str
    skills: str
    education: str
    positions: str
    full_text: str

class JobPostData(BaseModel):
    job_title: str
    required_skills: str
    target_location: str
    min_experience: int
    target_age: int
    job_description: str

class ApplicationRequest(BaseModel):
    candidate_data: CandidateData
    job_post: JobPostData

class MatchScoreResponse(BaseModel):
    final_score: float
    skill_match_ratio: float
    title_match_ratio: float
    semantic_similarity: float

class RankedApplicant(BaseModel):
    rank: int
    name: str
    score: float
    skill_match_ratio: float
    title_match_ratio: float
    semantic_similarity: float
    location: str
    experience: int

class RankingRequest(BaseModel):
    job_post: JobPostData
    applicants: List[CandidateData]

# --- Utility Functions ---
def deep_clean_text(text):
    if pd.isna(text) or text == "":
        return ""
    text = str(text).replace('[', '').replace(']', '').replace("'", "").replace('"', '')
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    words = text.split()
    clean_words = [w for w in words if w not in stop_words]
    return " ".join(clean_words)

def extract_full_cv_data_from_pdf(pdf_bytes: bytes, filename: str):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
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
        "candidate_name": filename.replace(".pdf", ""),
        "age": final_age,
        "experiencere": final_exp,
        "address": address,
        "skills": get_section(["Skills", "Technologies", "Technical Skills"]),
        "education": get_section(["Education", "Academic", "Qualifications"]),
        "positions": get_section(["Experience", "Work History", "Professional Experience"]),
        "full_text": text
    }

def load_models():
    global MODEL, SBERT_MODEL
    
    if MODEL is None:
        if os.path.exists('super_stacking_model.pkl'):
            with open('super_stacking_model.pkl', 'rb') as f:
                MODEL = pickle.load(f)
        else:
            raise FileNotFoundError("super_stacking_model.pkl not found")
    
    if SBERT_MODEL is None:
        SBERT_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    
    return MODEL, SBERT_MODEL

def calculate_match_score(candidate_data: CandidateData, job_post: JobPostData):
    model, sbert_model = load_models()
    
    # 1. Semantic Similarity (SBERT)
    job_text = deep_clean_text(job_post.job_title + " " + job_post.job_description + " " + job_post.required_skills)
    cv_text_combined = deep_clean_text(candidate_data.candidate_name + " " + candidate_data.skills + " " + candidate_data.education + " " + candidate_data.full_text)
    
    cv_emb = sbert_model.encode(cv_text_combined, convert_to_tensor=True)
    job_emb = sbert_model.encode(job_text, convert_to_tensor=True)
    semantic_sim = util.cos_sim(cv_emb, job_emb).item()

    # 2. Strict Title Keyword Match
    target_role = job_post.job_title.lower()
    clean_title = re.sub(r'[\(\)/]', ' ', target_role)
    title_keywords = [w for w in clean_title.split() if len(w) > 2 and w not in ["and", "the", "for"]]
    
    title_match_count = sum(1 for kw in title_keywords if kw in candidate_data.full_text.lower() or kw in candidate_data.skills.lower())
    title_match_ratio = title_match_count / len(title_keywords) if title_keywords else 1.0
    
    # 3. Skill Match
    req_skills = set([s.strip().lower() for s in job_post.required_skills.split(',') if s.strip()])
    cv_skills_set = set(deep_clean_text(candidate_data.skills).split())
    skill_ratio = len(cv_skills_set.intersection(req_skills)) / len(req_skills) if req_skills else 0.5

    # 4. Predict with Model
    selected_title = job_post.job_title
    mapped_title = TECH_JOB_MAPPING.get(selected_title, selected_title)
    job_cat = TRAINED_JOB_CATEGORIES.index(mapped_title) if mapped_title in TRAINED_JOB_CATEGORIES else 23

    features = np.array([[
        semantic_sim, skill_ratio, title_match_ratio, 
        float(candidate_data.age), float(candidate_data.experiencere), 
        float(candidate_data.experiencere - job_post.min_experience),
        float(candidate_data.age - job_post.target_age),
        float(job_cat)
    ]])
    
    model_score = model.predict(features)[0]

    # Domain Logic (No Language Bonus!)
    domain_bonus = 0.2 if title_match_ratio > 0.6 else 0.0
    skill_bonus = 0.2 if skill_ratio > 0.7 else 0.0
    domain_penalty = 0.4 if title_match_ratio < 0.4 else 0.0
    
    final_score_raw = (model_score * 0.4) + (skill_ratio * 0.3) + (title_match_ratio * 0.3) + domain_bonus + skill_bonus - domain_penalty
    final_pct = max(0, min(100.0, final_score_raw * 100))

    return {
        "final_score": final_pct,
        "skill_match_ratio": skill_ratio,
        "title_match_ratio": title_match_ratio,
        "semantic_similarity": semantic_sim
    }

# --- API Endpoints ---
@app.on_event("startup")
async def startup_event():
    print("Loading AI models...")
    load_models()
    print("Models loaded successfully!")

@app.get("/")
async def root():
    return {"message": "AI Hiring Platform API is running!", "version": "1.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/api/extract-cv", response_model=CandidateData)
async def extract_cv(file: UploadFile = File(...)):
    """Extract data from a PDF CV file"""
    try:
        contents = await file.read()
        cv_data = extract_full_cv_data_from_pdf(contents, file.filename)
        return CandidateData(**cv_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting CV: {str(e)}")

@app.post("/api/calculate-score", response_model=MatchScoreResponse)
async def calculate_score(request: ApplicationRequest):
    """Calculate match score between a candidate and a job post"""
    try:
        result = calculate_match_score(request.candidate_data, request.job_post)
        return MatchScoreResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating score: {str(e)}")

@app.post("/api/rank-applicants", response_model=List[RankedApplicant])
async def rank_applicants(request: RankingRequest):
    """Rank multiple applicants for a job post and return sorted results"""
    try:
        ranked_list = []
        
        for idx, applicant in enumerate(request.applicants):
            score_data = calculate_match_score(applicant, request.job_post)
            ranked_list.append({
                "name": applicant.candidate_name,
                "score": score_data["final_score"],
                "skill_match_ratio": score_data["skill_match_ratio"],
                "title_match_ratio": score_data["title_match_ratio"],
                "semantic_similarity": score_data["semantic_similarity"],
                "location": applicant.address,
                "experience": applicant.experiencere,
                "rank": 0
            })
        
        # Sort by score descending
        ranked_list.sort(key=lambda x: x["score"], reverse=True)
        
        # Assign ranks
        for idx, applicant in enumerate(ranked_list):
            applicant["rank"] = idx + 1
        
        return [RankedApplicant(**app) for app in ranked_list]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ranking applicants: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
