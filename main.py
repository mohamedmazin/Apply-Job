import ast
import pickle
import re
import os
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import fitz
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import io

# تحميل NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

app = FastAPI(title="AI Hiring Platform API", version="2.0")

# تفعيل CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- المتغيرات العالمية للنموذج ---
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

# --- نماذج Pydantic للطلبات والاستجابات ---
class CandidateData(BaseModel):
    age: Optional[int] = 25
    years_of_experience: Optional[int] = 0
    address: Optional[str] = ""
    skills_extracted: Optional[str] = ""
    highest_education: Optional[str] = ""
    education_details_extracted: Optional[str] = ""
    full_text: str = ""

class JobPostData(BaseModel):
    title: str
    description: Optional[str] = ""
    location: Optional[str] = ""
    min_age: Optional[int] = 18
    max_age: Optional[int] = 60
    experience_level: Optional[str] = "Entry"  # Entry/Mid/Senior
    required_skills: Optional[str] = ""

class MatchScoreResponse(BaseModel):
    final_score: float
    skill_match_ratio: float
    title_match_ratio: float
    semantic_similarity: float
    age_match_score: float
    experience_match_score: float

class RankedApplicant(BaseModel):
    rank: int
    score: float
    skill_match_ratio: float
    title_match_ratio: float
    semantic_similarity: float
    location: Optional[str] = ""
    years_of_experience: Optional[int] = 0

class RankingRequest(BaseModel):
    job_post: JobPostData
    applicants: List[CandidateData]

# --- تحويل ExperienceLevel إلى رقم ---
def experience_level_to_years(level: str) -> int:
    level_map = {
        "Entry": 0,
        "Mid": 3,
        "Senior": 5
    }
    return level_map.get(level, 0)

# --- دالات الخدمة ---
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
    current_year = 2026

    # --- Extract text WITH LAYOUT (lines + positions) ---
    lines = []
    for page in doc:
        blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
        for block in blocks:
            if block[6] == 0:  # Text block only
                block_text = block[4].strip()
                if block_text:
                    lines.append({
                        "text": block_text,
                        "y0": block[1],  # Top position
                        "y1": block[3],  # Bottom position
                    })
    # Sort lines top to bottom
    lines = sorted(lines, key=lambda x: x["y0"])
    full_text = " ".join([l["text"] for l in lines])

    # --- Helper function to detect headers ---
    def is_header(line_text):
        # Check if line is a section header (ALL CAPS, bold-like, or ends with :)
        t = line_text.strip()
        # Common header keywords
        header_keywords = ['skills', 'technical skills', 'technologies', 'tech stack', 
                           'core competencies', 'technical expertise',
                           'education', 'academic', 'qualifications',
                           'experience', 'work', 'employment',
                           'projects', 'certifications', 'summary', 'objective',
                           'address', 'location', 'contact', 'personal']
        has_header_kw = any(kw in t.lower() for kw in header_keywords)
        is_all_caps = t.isupper() and len(t) > 2
        ends_with_colon = t.endswith(':')
        return has_header_kw or is_all_caps or ends_with_colon

    # --- 1. Extract Age ---
    all_years = sorted(list(set([int(y) for y in re.findall(r'\b(19[7-9][0-9]|20[0-2][0-9])\b', full_text)])))
    final_age = 25
    if all_years:
        final_age = (current_year - all_years[0]) if (current_year - all_years[0]) > 18 else (current_year - all_years[0] + 22)
    
    # --- 2. Extract Experience ---
    exp_matches = re.findall(r'(\d+)\s?\+?\s?years?\s?(?:of\s?)?(?:experience|exp)', full_text, re.I)
    final_exp = 0
    if exp_matches:
        final_exp = int(exp_matches[0])
    else:
        date_ranges = re.findall(r'(\d{4})\s*[-–—]\s*(\d{4}|Present|Current|Now)', full_text, re.I)
        exp_years = 0
        for start, end in date_ranges:
            end_year = current_year if end.lower() in ['present', 'current', 'now'] else int(end)
            exp_years += max(0, end_year - int(start))
        final_exp = exp_years if exp_years > 0 else 0
    
    # --- 3. Extract Address (using line-by-line top section) ---
    address = "Not explicitly found"
    top_lines = lines[:20]  # Check top 20 lines for address
    addr_header_found = False
    for i, line in enumerate(top_lines):
        t = line["text"].lower()
        # Look for Address/Location header
        if any(kw in t for kw in ['address', 'location', 'lives in', 'based in', 'residence']):
            addr_header_found = True
            # Check next 3 lines for address
            for j in range(i+1, min(i+4, len(top_lines))):
                candidate = top_lines[j]["text"].strip()
                if len(candidate) > 3 and len(candidate) < 100 and not is_header(candidate):
                    address = candidate
                    break
            # Also check if address is on same line as header
            same_line = re.search(r'(?:Address|Location|Lives in|Based in|Residence)\s*[:\-]?\s*(.*)', line["text"], re.I)
            if same_line and same_line.group(1):
                address = same_line.group(1).strip()
            break
    # Fallback: look for city, state pattern in top lines
    if address == "Not explicitly found":
        for line in top_lines:
            city_pattern = re.search(r'(\w+,\s*\w+,\s*\w+|\w+,\s*\w+)', line["text"])
            if city_pattern and not any(kw in line["text"].lower() for kw in ['skills', 'education']):
                address = city_pattern.group(1).strip()
                break

    # --- 4. Extract Skills (using section detection) ---
    skills = "Information not found"
    skills_collected = []
    in_skills_section = False
    for i, line in enumerate(lines):
        t = line["text"].lower()
        # Enter skills section
        if any(kw in t for kw in ['skills', 'technical skills', 'technologies', 'tech stack', 'core competencies', 'technical expertise']):
            in_skills_section = True
            # Check if skills are on same line
            same_line = re.search(r'(?:Skills|Technical Skills|Technologies|Tech Stack|Core Competencies|Technical Expertise)\s*[:\-]?\s*(.*)', line["text"], re.I)
            if same_line and same_line.group(1):
                skills_collected.append(same_line.group(1).strip())
            continue
        
        # Exit skills section when next header found
        if in_skills_section and is_header(line["text"]):
            break
        
        # Collect skills
        if in_skills_section:
            skills_collected.append(line["text"].strip())
    
    if skills_collected:
        skills = "\n".join([s for s in skills_collected if s])

    # --- 5. Extract Education (using section detection) ---
    education = "Information not found"
    edu_collected = []
    in_edu_section = False
    for i, line in enumerate(lines):
        t = line["text"].lower()
        if any(kw in t for kw in ['education', 'academic background', 'qualifications', 'academic']):
            in_edu_section = True
            same_line = re.search(r'(?:Education|Academic Background|Qualifications|Academic)\s*[:\-]?\s*(.*)', line["text"], re.I)
            if same_line and same_line.group(1):
                edu_collected.append(same_line.group(1).strip())
            continue
        if in_edu_section and is_header(line["text"]):
            break
        if in_edu_section:
            edu_collected.append(line["text"].strip())
    if edu_collected:
        education = "\n".join([s for s in edu_collected if s])

    # --- 6. Extract Highest Education ---
    highest_education = "Information not found"
    edu_keywords = [
        (r'(PhD|Doctor(?:ate)?|Doctor of)', "PhD"),
        (r'(Master|MS|MSc|MA|MPhil|MBA)', "Master's Degree"),
        (r'(Bachelor|BS|BSc|BA|BBA|BE|BEng)', "Bachelor's Degree"),
        (r'(Associate|Diploma|High School|Secondary)', "Associate/Diploma"),
    ]
    for pattern, degree in edu_keywords:
        if re.search(pattern, education, re.I) or re.search(pattern, full_text, re.I):
            highest_education = degree
            break
    if highest_education == "Information not found" and education:
        highest_education = education.split('\n')[0][:50] if '\n' in education else education[:50]

    return {
        "age": final_age,
        "years_of_experience": final_exp,
        "address": address,
        "skills_extracted": skills,
        "highest_education": highest_education,
        "education_details_extracted": education,
        "full_text": full_text
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

def calculate_match_score(candidate: CandidateData, job_post: JobPostData):
    model, sbert_model = load_models()
    
    # 1. التشابه الدلالي (SBERT)
    job_text = deep_clean_text(job_post.title + " " + (job_post.description or "") + " " + (job_post.required_skills or ""))
    cv_text_combined = deep_clean_text((candidate.skills_extracted or "") + " " + (candidate.education_details_extracted or "") + " " + (candidate.full_text or ""))
    
    cv_emb = sbert_model.encode(cv_text_combined, convert_to_tensor=True)
    job_emb = sbert_model.encode(job_text, convert_to_tensor=True)
    semantic_sim = util.cos_sim(cv_emb, job_emb).item()

    # 2. مطابقة الكلمات المفتاحية للعنوان
    target_role = job_post.title.lower()
    clean_title = re.sub(r'[\(\)/]', ' ', target_role)
    title_keywords = [w for w in clean_title.split() if len(w) > 2 and w not in ["and", "the", "for"]]
    
    title_match_count = sum(1 for kw in title_keywords if kw in (candidate.full_text or "").lower() or kw in (candidate.skills_extracted or "").lower())
    title_match_ratio = title_match_count / len(title_keywords) if title_keywords else 1.0
    
    # 3. مطابقة المهارات
    req_skills = set([s.strip().lower() for s in (job_post.required_skills or "").split(',') if s.strip()])
    cv_skills_set = set(deep_clean_text(candidate.skills_extracted or "").split())
    skill_ratio = len(cv_skills_set.intersection(req_skills)) / len(req_skills) if req_skills else 0.5

    # 4. مطابقة العمر (Range)
    age = candidate.age or 25
    min_age = job_post.min_age or 18
    max_age = job_post.max_age or 60
    if min_age <= age <= max_age:
        age_match_score = 1.0
    elif age < min_age:
        age_match_score = max(0, 1 - (min_age - age) / 10)
    else:
        age_match_score = max(0, 1 - (age - max_age) / 10)

    # 5. مطابقة الخبرة
    exp_candidate = candidate.years_of_experience or 0
    exp_required = experience_level_to_years(job_post.experience_level or "Entry")
    exp_diff = exp_candidate - exp_required
    experience_match_score = max(0, min(1.0, 0.5 + exp_diff / 5))

    # 6. التنبؤ باستخدام النموذج
    selected_title = job_post.title
    mapped_title = TECH_JOB_MAPPING.get(selected_title, selected_title)
    job_cat = TRAINED_JOB_CATEGORIES.index(mapped_title) if mapped_title in TRAINED_JOB_CATEGORIES else 23

    features = np.array([[
        semantic_sim, skill_ratio, title_match_ratio, 
        float(age), float(exp_candidate), 
        float(exp_diff),
        float(age - (min_age + max_age)/2),
        float(job_cat)
    ]])
    
    model_score = model.predict(features)[0]

    # منطق المجال
    domain_bonus = 0.2 if title_match_ratio > 0.6 else 0.0
    skill_bonus = 0.2 if skill_ratio > 0.7 else 0.0
    domain_penalty = 0.4 if title_match_ratio < 0.4 else 0.0
    
    final_score_raw = (model_score * 0.4) + (skill_ratio * 0.3) + (title_match_ratio * 0.3) + domain_bonus + skill_bonus - domain_penalty
    final_score_raw *= (0.7 * age_match_score + 0.3 * experience_match_score)
    final_pct = max(0, min(100.0, final_score_raw * 100))

    return {
        "final_score": final_pct,
        "skill_match_ratio": skill_ratio,
        "title_match_ratio": title_match_ratio,
        "semantic_similarity": semantic_sim,
        "age_match_score": age_match_score,
        "experience_match_score": experience_match_score
    }

# --- نقاط النهاية (Endpoints) ---
@app.on_event("startup")
async def startup_event():
    print("Loading AI models...")
    load_models()
    print("Models loaded successfully!")

@app.get("/")
async def root():
    return {"message": "AI Hiring Platform API is running!", "version": "2.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# --- CV Endpoints ---
@app.post("/api/extract-cv", response_model=CandidateData)
async def extract_cv(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        cv_data = extract_full_cv_data_from_pdf(contents, file.filename)
        return CandidateData(**cv_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting CV: {str(e)}")

# --- Match & Ranking Endpoints ---
@app.post("/api/calculate-score", response_model=MatchScoreResponse)
async def calculate_score_endpoint(job_post: JobPostData, candidate: CandidateData):
    try:
        result = calculate_match_score(candidate, job_post)
        return MatchScoreResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating score: {str(e)}")

@app.post("/api/rank-applicants", response_model=List[RankedApplicant])
async def rank_applicants_endpoint(request: RankingRequest):
    try:
        ranked_list = []
        
        for idx, applicant in enumerate(request.applicants):
            score_data = calculate_match_score(applicant, request.job_post)
            ranked_list.append({
                "score": score_data["final_score"],
                "skill_match_ratio": score_data["skill_match_ratio"],
                "title_match_ratio": score_data["title_match_ratio"],
                "semantic_similarity": score_data["semantic_similarity"],
                "location": applicant.address,
                "years_of_experience": applicant.years_of_experience,
                "rank": 0
            })
        
        # ترتيب حسب النتيجة تنازلياً
        ranked_list.sort(key=lambda x: x["score"], reverse=True)
        
        # إعطاء الترتيب
        for idx, applicant in enumerate(ranked_list):
            applicant["rank"] = idx + 1
        
        return [RankedApplicant(**app) for app in ranked_list]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ranking applicants: {str(e)}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
