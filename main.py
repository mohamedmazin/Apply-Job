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

# --- Common known skills list (you can add more!) ---
COMMON_SKILLS = [
    # Programming Languages
    "Python", "Java", "C#", "C++", "JavaScript", "TypeScript", "PHP", "Ruby", "Go", "Rust",
    "Swift", "Kotlin", "C", "R", "SQL", "HTML", "CSS", "Scala", "Perl", "Dart", "Lua",
    "Shell", "Bash", "PowerShell", "MATLAB", "Groovy", "Objective-C",
    # Frameworks & Libraries
    "ASP.NET", "ASP.NET Core", "ASP.NET Core MVC", "Razor Pages", "Django", "Flask", "Express", "React",
    "React.js", "React Native", "Vue.js", "Vue", "Angular", "Angular.js", "Spring", "Spring Boot",
    "Entity Framework", "Entity Framework Core", "Node.js", "Next.js", "Nuxt.js", "jQuery",
    "Bootstrap", "Tailwind CSS", "Tailwind", "Material UI", "Material-UI", "MUI", "TensorFlow",
    "PyTorch", "Keras", "Pandas", "NumPy", "Scikit-learn", "Matplotlib", "Seaborn", "FastAPI",
    "Flask", "Laravel", "Symfony", "CodeIgniter", "Yii", "CakePHP", "Zend",
    # Databases
    "SQL Server", "MySQL", "PostgreSQL", "MongoDB", "Oracle", "Redis", "SQLite", "Firebase",
    "MariaDB", "DynamoDB", "Cassandra", "CouchDB", "Elasticsearch", "Solr",
    # Tools & Technologies
    "Git", "GitHub", "GitLab", "Bitbucket", "Docker", "Kubernetes", "AWS", "Amazon Web Services",
    "Azure", "Microsoft Azure", "GCP", "Google Cloud Platform", "Linux", "Windows", "MacOS",
    "REST API", "RESTful APIs", "RESTful API", "GraphQL", "CI/CD", "Jenkins", "GitLab CI",
    "GitHub Actions", "CircleCI", "Travis CI", "Jira", "Confluence", "Slack", "Trello",
    "Figma", "Adobe XD", "Photoshop", "Illustrator", "Sketch", "Postman", "Swagger",
    "RabbitMQ", "Kafka", "Redis", "Memcached", "Nginx", "Apache", "IIS",
    # Concepts
    "Data Structures", "Algorithms", "OOP", "Object-Oriented Programming", "SOLID Principles", "Design Patterns",
    "Machine Learning", "Deep Learning", "Artificial Intelligence", "Data Science", "Big Data",
    "Web Development", "Backend Development", "Frontend Development", "Full Stack Development",
    "Mobile Development", "Responsive Design", "Agile", "Scrum", "Kanban", "DevOps",
    "Microservices", "Monolithic", "REST", "SOAP", "API", "Web Services",
    "Authentication", "Authorization", "OAuth", "JWT", "JSON Web Tokens",
    "Password Hashing", "Security", "Cybersecurity", "Penetration Testing", "Ethical Hacking",
    "Database Management", "SQL Injection", "XSS", "Cross-Site Scripting",
    "Performance Optimization", "Scalability", "High Availability", "Load Balancing",
    # Soft Skills
    "Communication", "Teamwork", "Leadership", "Problem Solving", "Time Management",
    "Stress Management", "Critical Thinking", "Creativity", "Adaptability", "Collaboration",
    "Presentation Skills", "Negotiation", "Decision Making", "Emotional Intelligence",
    "Customer Service", "Multitasking", "Attention to Detail", "Flexibility"
]

def extract_known_skills(text: str):
    if not text or text == "Information not found":
        return "Information not found"
    
    found_skills = []
    text_lower = text.lower()
    
    for skill in COMMON_SKILLS:
        # Use word boundaries to match whole skill only
        # Escape special characters in skill name (like C# becomes C\#)
        skill_lower = skill.lower()
        # Check if skill exists as whole word in text
        pattern = r'\b' + re.escape(skill_lower) + r'\b'
        if re.search(pattern, text_lower):
            found_skills.append(skill)
    
    if found_skills:
        return ", ".join(found_skills)
    else:
        return "Information not found"

def extract_full_cv_data_from_pdf(pdf_bytes: bytes, filename: str):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    current_year = 2026

    # --- Step 1: Extract all text first (simple and safe) ---
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    # Clean up the full text a bit
    full_text = re.sub(r'\n+', '\n', full_text).strip()
    # Also get lines separated by newlines for section detection
    raw_lines = [line.strip() for line in full_text.split('\n') if line.strip()]

    # --- Helper: Find a section and get text until next section ---
    def get_section_text(start_keywords: list, end_keywords: list):
        start_idx = None
        end_idx = None
        
        for i, line in enumerate(raw_lines):
            line_lower = line.lower()
            # Find start of section
            if any(kw.lower() in line_lower for kw in start_keywords):
                if start_idx is None:
                    start_idx = i
            # Find end of section (only if we found start)
            elif start_idx is not None and any(kw.lower() in line_lower for kw in end_keywords):
                end_idx = i
                break
        
        # If start found, collect text
        if start_idx is not None:
            if end_idx is None:
                return '\n'.join(raw_lines[start_idx+1:])
            else:
                return '\n'.join(raw_lines[start_idx+1:end_idx])
        return None

    # --- All possible section keywords (for any format) ---
    skill_section_headers = [
        'Skills', 'Technical Skills', 'Technologies', 'Tech Stack', 
        'Core Competencies', 'Technical Expertise', 'Technical Skills:'
    ]
    education_section_headers = [
        'Education', 'Academic Background', 'Qualifications', 'Academic', 'Education:', 'Academic Background:'
    ]
    experience_section_headers = [
        'Experience', 'Work Experience', 'Employment History', 'Work History', 'Professional Experience', 'Experience:'
    ]
    all_section_headers = skill_section_headers + education_section_headers + experience_section_headers + [
        'Projects', 'Certifications', 'Certificates', 'Summary', 'Objective', 
        'Extracurricular', 'Languages', 'Contact', 'Personal', 'Address', 'Location'
    ]

    # --- 2. Extract Age (from birth year or default) ---
    all_years = sorted(list(set([int(y) for y in re.findall(r'\b(19[7-9][0-9]|20[0-2][0-9])\b', full_text)])))
    final_age = 25
    if all_years:
        final_age = (current_year - all_years[0]) if (current_year - all_years[0]) > 18 else (current_year - all_years[0] + 22)

    # --- 3. Extract Years of Experience ---
    final_exp = 0
    # Try simple "X years of experience" first
    exp_matches = re.findall(r'(\d+)\s?\+?\s?years?\s?(?:of\s?)?(?:experience|exp)', full_text, re.I)
    if exp_matches:
        final_exp = int(exp_matches[0])
    else:
        # Try date ranges (2020 – Present)
        date_ranges = re.findall(r'(\d{4})\s*[-–—]\s*(\d{4}|Present|Current|Now)', full_text, re.I)
        exp_years = 0
        for start, end in date_ranges:
            end_year = current_year if end.lower() in ['present', 'current', 'now'] else int(end)
            exp_years += max(0, end_year - int(start))
        final_exp = exp_years if exp_years > 0 else 0

    # --- 4. Extract Address (from top section or patterns) ---
    address = "Not explicitly found"
    # Look in the first few lines for location patterns
    top_section_text = ' '.join(raw_lines[:5]) if len(raw_lines) > 0 else full_text[:300]
    # Look for patterns like "Mit Akaba, Giza" or "Giza, Egypt"
    location_match = re.search(r'([A-Za-z\s]+,\s*[A-Za-z\s]+)', top_section_text)
    if location_match:
        candidate = location_match.group(1).strip()
        # Make sure it's not something like a name or email
        if len(candidate) > 3 and len(candidate) < 100 and '@' not in candidate and not any(kw in candidate.lower() for kw in ['skills', 'education', 'experience', 'github', 'linkedin', 'gmail', 'email']):
            address = candidate
    # Fallback: look for "Location" or "Address" in top lines
    if address == "Not explicitly found":
        for i in range(min(20, len(raw_lines))):
            line = raw_lines[i]
            line_lower = line.lower()
            if any(kw in line_lower for kw in ['address', 'location', 'lives in']):
                # Check next few lines
                for j in range(i+1, min(i+4, len(raw_lines))):
                    candidate = raw_lines[j].strip()
                    if len(candidate) > 3 and len(candidate) < 100 and not any(kw in candidate.lower() for kw in all_section_headers):
                        address = candidate
                        break

    # --- 5. Extract Skills (raw + filtered) ---
    raw_skills = "Information not found"
    # For skills, end keywords exclude other skill headers (so it doesn't stop at "Technical Skills")
    skill_end_keywords = [kw for kw in all_section_headers if kw not in skill_section_headers]
    skills_text = get_section_text(skill_section_headers, skill_end_keywords)
    if skills_text and len(skills_text) > 10:
        raw_skills = skills_text
    # Fallback: look for lines with many technical keywords (simple heuristic)
    if raw_skills == "Information not found":
        tech_keywords = ['c#', 'python', 'java', 'c++', 'sql', 'asp.net', 'mvc', 'git', 'github', 'api', 'rest', 'entity framework', 'data structures', 'algorithms']
        skill_candidates = []
        for line in raw_lines:
            line_lower = line.lower()
            if len(line) < 200 and any(kw in line_lower for kw in tech_keywords):
                skill_candidates.append(line)
        if len(skill_candidates) > 0:
            raw_skills = '\n'.join(skill_candidates)
    # Filter skills to keep only known ones
    filtered_skills = extract_known_skills(raw_skills)

    # --- 6. Extract Education Details ---
    education = "Information not found"
    edu_text = get_section_text(education_section_headers, all_section_headers)
    if edu_text and len(edu_text) > 10:
        education = edu_text

    # --- 7. Extract Highest Education ---
    highest_education = "Information not found"
    edu_keywords = [
        (r'(PhD|Doctor(?:ate)?|Doctor of)', "PhD"),
        (r'(Master|MS|MSc|MA|MPhil|MBA)', "Master's Degree"),
        (r'(Bachelor|BS|BSc|BA|BBA|BE|BEng|Faculty of)', "Bachelor's Degree"),
        (r'(Associate|Diploma|High School|Secondary)', "Associate/Diploma"),
    ]
    for pattern, degree in edu_keywords:
        if re.search(pattern, education, re.I) or re.search(pattern, full_text, re.I):
            highest_education = degree
            break
    if highest_education == "Information not found" and education:
        highest_education = education.split('\n')[0][:50] if '\n' in education else education[:50]

    # --- Return all data ---
    # If filtered skills are empty, use the raw extracted skills instead
    final_skills = filtered_skills if filtered_skills != "Information not found" else raw_skills
    return {
        "age": final_age,
        "years_of_experience": final_exp,
        "address": address,
        "skills_extracted": final_skills,
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
