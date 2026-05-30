import ast
import pickle
import re
import os
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import fitz
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Text, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import uvicorn
import io

# تحميل المتغيرات من .env
load_dotenv()

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

# --- إعدادات قاعدة البيانات (SQL Server) ---
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    DATABASE_URL = "mssql+pyodbc:///?odbc_connect=DRIVER={ODBC Driver 17 for SQL Server};SERVER=localhost;DATABASE=hiring;UID=sa;PWD=your_password"  # مثال

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- نماذج قاعدة البيانات (JobPosts و JobApplications الحالية) ---
class JobPostDB(Base):
    __tablename__ = "JobPosts"  # اسم الجدول الحالي
    
    # أضف الأعمدة اللي عندك في جدول JobPosts
    # مثال بناءً على الصور اللي شفتها قبل كدا
    id = Column(Integer, primary_key=True, index=True)
    Title = Column(String(255), nullable=False)
    JobType = Column(String(100))
    CompanyName = Column(String(255))
    Description = Column(Text)
    Location = Column(String(255))
    MinSalary = Column(Float)
    MaxSalary = Column(Float)
    MinAge = Column(Integer)
    MaxAge = Column(Integer)
    ExperienceLevel = Column(String(100))  # "Entry", "Mid", "Senior"
    # أضف باقي الأعمدة اللي عندك هنا...

class JobApplicationDB(Base):
    __tablename__ = "JobApplications"  # اسم الجدول الحالي
    
    # أضف الأعمدة اللي عندك في جدول JobApplications
    id = Column(Integer, primary_key=True, index=True)
    Age = Column(Integer)
    YearsOfExperience = Column(Integer)
    AvailableStartDate = Column(String(255))
    MinExpectedSalary = Column(Float)
    MaxExpectedSalary = Column(Float)
    CoverLetter = Column(Text)
    SeekerTitle = Column(String(255))
    Address = Column(String(255))
    SkillsExtracted = Column(Text)
    HighestEducation = Column(Text)
    EducationDetailsExtracted = Column(Text)
    # أضف باقي الأعمدة اللي عندك هنا...

# --- لم نعد نحتاج نعمل create tables لانه الجداول موجودة فعلاً ---

# Dependency للجلسة
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

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
    Age: Optional[int] = 25
    YearsOfExperience: Optional[int] = 0
    Address: Optional[str] = ""
    SkillsExtracted: Optional[str] = ""
    HighestEducation: Optional[str] = ""
    EducationDetailsExtracted: Optional[str] = ""
    FullText: str = ""  # نستخدم ده للنص الكامل للـ CV

class JobPostData(BaseModel):
    id: Optional[int] = None
    Title: str
    Description: Optional[str] = ""
    Location: Optional[str] = ""
    MinAge: Optional[int] = 18
    MaxAge: Optional[int] = 60
    ExperienceLevel: Optional[str] = "Entry"
    # أضف باقي الحقول اللي عندك هنا...

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
    job_post_id: Optional[int] = None
    job_post: Optional[JobPostData] = None
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

    skills = get_section(["Skills", "Technologies", "Technical Skills"])
    education = get_section(["Education", "Academic", "Qualifications"])

    return {
        "Age": final_age,
        "YearsOfExperience": final_exp,
        "Address": address,
        "SkillsExtracted": skills,
        "HighestEducation": education.split('\n')[0] if education else "",
        "EducationDetailsExtracted": education,
        "FullText": text
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
    job_text = deep_clean_text(job_post.Title + " " + (job_post.Description or "") + " ")  # أضف مهارات لو عندك
    cv_text_combined = deep_clean_text((candidate.SkillsExtracted or "") + " " + (candidate.EducationDetailsExtracted or "") + " " + (candidate.FullText or ""))
    
    cv_emb = sbert_model.encode(cv_text_combined, convert_to_tensor=True)
    job_emb = sbert_model.encode(job_text, convert_to_tensor=True)
    semantic_sim = util.cos_sim(cv_emb, job_emb).item()

    # 2. مطابقة الكلمات المفتاحية للعنوان
    target_role = job_post.Title.lower()
    clean_title = re.sub(r'[\(\)/]', ' ', target_role)
    title_keywords = [w for w in clean_title.split() if len(w) > 2 and w not in ["and", "the", "for"]]
    
    title_match_count = sum(1 for kw in title_keywords if kw in (candidate.FullText or "").lower() or kw in (candidate.SkillsExtracted or "").lower())
    title_match_ratio = title_match_count / len(title_keywords) if title_keywords else 1.0
    
    # 3. مطابقة المهارات (لو عندك عمود RequiredSkills في JobPosts)
    skill_ratio = 0.5  # قيمة افتراضية، أضف منطقك هنا لو عندك المهارات في JobPosts

    # 4. مطابقة العمر (Range)
    age = candidate.Age or 25
    min_age = job_post.MinAge or 18
    max_age = job_post.MaxAge or 60
    if min_age <= age <= max_age:
        age_match_score = 1.0
    elif age < min_age:
        age_match_score = max(0, 1 - (min_age - age) / 10)
    else:
        age_match_score = max(0, 1 - (age - max_age) / 10)

    # 5. مطابقة الخبرة
    exp_candidate = candidate.YearsOfExperience or 0
    exp_required = experience_level_to_years(job_post.ExperienceLevel or "Entry")
    exp_diff = exp_candidate - exp_required
    experience_match_score = max(0, min(1.0, 0.5 + exp_diff / 5))

    # 6. التنبؤ باستخدام النموذج
    selected_title = job_post.Title
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

# --- Job Post Endpoints (قراءة فقط) ---
@app.get("/api/job-posts", response_model=List[JobPostData])
async def get_job_posts(db: Session = Depends(get_db)):
    jobs = db.query(JobPostDB).all()
    return [JobPostData(id=j.id, **{c.name: getattr(j, c.name) for c in j.__table__.columns}) for j in jobs]

@app.get("/api/job-posts/{job_id}", response_model=JobPostData)
async def get_job_post(job_id: int, db: Session = Depends(get_db)):
    job = db.query(JobPostDB).filter(JobPostDB.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job post not found")
    return JobPostData(id=job.id, **{c.name: getattr(job, c.name) for c in job.__table__.columns})

# --- Job Applications Endpoints ---
@app.get("/api/job-applications", response_model=List[CandidateData])
async def get_job_applications(db: Session = Depends(get_db)):
    apps = db.query(JobApplicationDB).all()
    return [CandidateData(**{c.name: getattr(app, c.name) for c in app.__table__.columns}) for app in apps]

@app.get("/api/job-applications/{app_id}", response_model=CandidateData)
async def get_job_application(app_id: int, db: Session = Depends(get_db)):
    app = db.query(JobApplicationDB).filter(JobApplicationDB.id == app_id).first()
    if not app:
        raise HTTPException(status_code=404, detail="Job application not found")
    return CandidateData(**{c.name: getattr(app, c.name) for c in app.__table__.columns})

# --- Match & Ranking Endpoints ---
@app.post("/api/calculate-score", response_model=MatchScoreResponse)
async def calculate_score_endpoint(job_post: JobPostData, candidate: CandidateData):
    try:
        result = calculate_match_score(candidate, job_post)
        return MatchScoreResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating score: {str(e)}")

@app.post("/api/rank-applicants", response_model=List[RankedApplicant])
async def rank_applicants_endpoint(request: RankingRequest, db: Session = Depends(get_db)):
    try:
        # الحصول على بيانات الوظيفة إما من ID أو من الطلب مباشرة
        job_post_data = request.job_post
        if request.job_post_id and not job_post_data:
            job_db = db.query(JobPostDB).filter(JobPostDB.id == request.job_post_id).first()
            if not job_db:
                raise HTTPException(status_code=404, detail="Job post not found")
            job_post_data = JobPostData(id=job_db.id, **{c.name: getattr(job_db, c.name) for c in job_db.__table__.columns})
        
        if not job_post_data:
            raise HTTPException(status_code=400, detail="Either job_post or job_post_id is required")

        ranked_list = []
        
        for idx, applicant in enumerate(request.applicants):
            score_data = calculate_match_score(applicant, job_post_data)
            ranked_list.append({
                "score": score_data["final_score"],
                "skill_match_ratio": score_data["skill_match_ratio"],
                "title_match_ratio": score_data["title_match_ratio"],
                "semantic_similarity": score_data["semantic_similarity"],
                "location": applicant.Address,
                "years_of_experience": applicant.YearsOfExperience,
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
