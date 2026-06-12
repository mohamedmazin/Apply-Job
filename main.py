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
    "Back-End": "Senior Software Engineer",
    ".NET Developer": "Senior Software Engineer",
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
    model_score: float
    scaled_model_score: float
    skill_match_ratio: float
    title_match_ratio: float
    semantic_similarity: float
    age_match_score: float
    experience_match_score: float
    title_bonus: float
    location_bonus: float

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

# --- تحويل ExperienceLevel إلى رقم (الاصلي) ---
def experience_level_to_years(level: str) -> int:
    level_map = {
        "Entry": 0,
        "Mid": 3,
        "Senior": 5
    }
    return level_map.get(level, 0)

# --- تحويل ExperienceLevel أو Range إلى أرقام أو نطاق (حصرًا 3 زي الصورة) ---
def parse_experience_requirement(exp_req: str) -> tuple[int, int]:
  
    exp_req = str(exp_req).strip().lower()
    
    # أولاً: نحدد بالضبط زي الصورة
    if "entry" in exp_req or "0-2" in exp_req:
        return (0, 2)
    elif "mid" in exp_req or "3-5" in exp_req:
        return (3, 5)
    elif "senior" in exp_req or "5+" in exp_req:
        return (5, 99)
    
    # الافتراضي لو مش لاقى
    return (0, 99)

def calculate_experience_match_score(candidate_exp: int, exp_req: str) -> float:
    """
    حساب درجة مطابقة الخبرة (حسب الـ 3 Ranges):
    - 1.0 لو الخبرة في أو أعلى النطاق المطلوب
    - 0.7 لو أقل بسنة
    - 0.4 لو أقل بسنتين
    - أقل من كده → 0.0
    """
    min_exp, max_exp = parse_experience_requirement(exp_req)
    candidate_exp = max(0, int(candidate_exp))
    
    if candidate_exp >= min_exp:
        return 1.0
    elif candidate_exp >= min_exp - 1:
        return 0.7
    elif candidate_exp >= min_exp - 2:
        return 0.4
    else:
        return 0.0

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

def clamp_number(value: Any, low: float, high: float, default: float) -> float:
    try:
        v = float(value)
    except Exception:
        return float(default)
    if v < low:
        return float(low)
    if v > high:
        return float(high)
    return float(v)

def normalize_location(value: str) -> str:
    if not value:
        return ""
    s = str(value).lower()
    s = s.replace("–", "-").replace("—", "-").replace("|", ",").replace("⋄", ",")
    s = re.sub(r"[^0-9\w\s,/-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def location_match_score(candidate_address: str, job_location: str) -> float:
    cand = normalize_location(candidate_address)
    job = normalize_location(job_location)
    if not cand or not job:
        return 0.0
    if "remote" in job or "hybrid" in job:
        return 0.0

    stop = {
        "in", "at", "from", "to",
        "area", "city", "governorate", "province", "state", "country",
        "street", "st", "road", "rd", "ave", "avenue",
        "egypt", "eg", "ksa", "uae", "saudi", "arabia",
        "onsite", "on", "site", "on-site",
    }

    def to_tokens(s: str) -> set:
        parts = []
        for chunk in re.split(r"[,/;-]+", s):
            chunk = chunk.strip()
            if not chunk:
                continue
            parts.extend([p for p in chunk.split() if p])
        tokens = set()
        for p in parts:
            if p in stop:
                continue
            if len(p) < 2:
                continue
            tokens.add(p)
        return tokens

    cand_tokens = to_tokens(cand)
    job_tokens = to_tokens(job)
    if not cand_tokens or not job_tokens:
        return 0.0

    if cand_tokens.intersection(job_tokens):
        return 1.0

    cand_str = " " + cand + " "
    job_str = " " + job + " "
    for t in cand_tokens:
        if (" " + t + " ") in job_str:
            return 0.5
    for t in job_tokens:
        if (" " + t + " ") in cand_str:
            return 0.5
    return 0.0

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
        skill_lower = skill.lower()
        pattern = r'(?<![a-z0-9])' + re.escape(skill_lower) + r'(?![a-z0-9])'
        if re.search(pattern, text_lower, flags=re.IGNORECASE):
            found_skills.append(skill)
    
    if found_skills:
        return ", ".join(found_skills)
    else:
        return "Information not found"

def extract_full_cv_data_from_pdf(pdf_bytes: bytes, filename: str):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    current_year = 2026

    # --- Step 1: Extract all text and line info ---
    raw_lines = []
    full_text_parts = []
    
    for page in doc:
        # Get text as dictionary with word-level info
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block.get("type") == 0:  # text block
                for line in block["lines"]:
                    line_text = " ".join([span["text"] for span in line["spans"]]).strip()
                    if line_text:
                        raw_lines.append(line_text)
                        full_text_parts.append(line_text)
    
    full_text = ' '.join(full_text_parts)

    # --- Helper: Find a section and get text until next section ---
    def get_section_text(start_keywords: list, end_keywords: list):
        start_idx = None
        end_idx = None
        start_inline_text = ""

        def match_start(line_text: str, keyword: str) -> Optional[str]:
            m = re.match(r'^\s*' + re.escape(keyword) + r'\s*[:\-–—]?\s*(.*)$', line_text, flags=re.IGNORECASE)
            if not m:
                return None
            return (m.group(1) or "").strip()

        def is_end_header(line_text: str, keyword: str) -> bool:
            return re.match(r'^\s*' + re.escape(keyword) + r'\s*[:\-–—]?\s*$', line_text, flags=re.IGNORECASE) is not None

        for i, line in enumerate(raw_lines):
            if start_idx is None:
                for kw in start_keywords:
                    remainder = match_start(line, kw)
                    if remainder is not None:
                        start_idx = i
                        start_inline_text = remainder
                        break
                continue

            for kw in end_keywords:
                if is_end_header(line, kw):
                    end_idx = i
                    break
            if end_idx is not None:
                break

        if start_idx is None:
            return None

        body_lines = []
        if start_inline_text:
            body_lines.append(start_inline_text)

        body_slice = raw_lines[start_idx + 1:] if end_idx is None else raw_lines[start_idx + 1:end_idx]
        body_lines.extend(body_slice)

        return "\n".join([l for l in body_lines if l.strip()]) or None

    # --- All possible section keywords (for any format) ---
    skill_section_headers = [
        'Skills', 'Technical Skills', 'Technologies', 'Tech Stack',
        'Core Competencies', 'Technical Expertise'
    ]
    education_section_headers = [
        'Education', 'Academic Background', 'Qualifications', 'Academic'
    ]
    experience_section_headers = [
        'Experience', 'Work Experience', 'Employment History', 'Work History', 'Professional Experience'
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
    skill_end_keywords = [
        'Education', 'Experience', 'Work Experience', 'Employment History', 'Work History', 'Professional Experience',
        'Projects', 'Certifications', 'Certificates', 'Summary', 'Objective', 'Extracurricular'
    ]
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
    semantic_sim = clamp_number(util.cos_sim(cv_emb, job_emb).item(), -1.0, 1.0, 0.0)

    # 2. مطابقة الكلمات المفتاحية للعنوان (Binary 0 or 1 زي التدريب)
    target_role = job_post.title.lower()
    clean_title = re.sub(r'[\(\)/]', ' ', target_role)
    title_keywords = [w for w in clean_title.split() if len(w) > 2 and w not in ["and", "the", "for"]]
    
    title_match_count = sum(1 for kw in title_keywords if kw in (candidate.full_text or "").lower() or kw in (candidate.skills_extracted or "").lower())
    title_match_ratio = title_match_count / len(title_keywords) if title_keywords else 1.0
    title_match_ratio = clamp_number(title_match_ratio, 0.0, 1.0, 0.0)
    title_match = 1.0 if title_match_ratio >= 0.5 else 0.0
    
    # 3. مطابقة المهارات (زي ما هو)
    req_skill_items = [deep_clean_text(s.strip()) for s in (job_post.required_skills or "").split(",") if s.strip()]
    req_skill_items = [s for s in req_skill_items if s]
    candidate_skills_blob = deep_clean_text(candidate.skills_extracted or "")
    if req_skill_items:
        matched_count = sum(1 for s in req_skill_items if s in candidate_skills_blob)
        skill_ratio = matched_count / len(req_skill_items)
    else:
        skill_ratio = 0.5
    skill_ratio = clamp_number(skill_ratio, 0.0, 1.0, 0.5)

    # 4. مطابقة العمر (Calculate single age_requirement زي التدريب)
    age = clamp_number(candidate.age if candidate.age is not None else 25, 16.0, 80.0, 25.0)
    min_age = clamp_number(job_post.min_age if job_post.min_age is not None else 18, 16.0, 80.0, 18.0)
    max_age = clamp_number(job_post.max_age if job_post.max_age is not None else 60, 16.0, 80.0, 60.0)
    if max_age < min_age:
        min_age, max_age = max_age, min_age
    
    age_requirement = (min_age + max_age) / 2.0

    # --- If age is inside the acceptable range, set age_diff to 0!
    if min_age <= age <= max_age:
        age_diff = 0.0
    else:
        age_diff = age - age_requirement

    age_match_score = 1.0 if min_age <= age <= max_age else (max(0, 1 - (min_age - age)/10) if age < min_age else max(0, 1 - (age - max_age)/10))

    # 5. مطابقة الخبرة (Range Logic + Original Model Compatibility)
    exp_candidate = clamp_number(candidate.years_of_experience if candidate.years_of_experience is not None else 0, 0.0, 60.0, 0.0)

    experience_match_score = calculate_experience_match_score(int(exp_candidate), job_post.experience_level or "Entry")

    min_exp, max_exp = parse_experience_requirement(job_post.experience_level or "Entry")
    exp_required_legacy = experience_level_to_years(job_post.experience_level or "Entry")

    # --- If experience >= min_exp, set exp_diff to 0!
    if exp_candidate >= min_exp:
        exp_diff = 0.0
    else:
        exp_diff = clamp_number(exp_candidate - exp_required_legacy, -60.0, 60.0, 0.0)

    loc_score = location_match_score(candidate.address or "", job_post.location or "")

    # 6. التنبؤ باستخدام النموذج (Features زي التدريب بالضبط!)
    selected_title = job_post.title
    mapped_title = TECH_JOB_MAPPING.get(selected_title, selected_title)
    job_cat = TRAINED_JOB_CATEGORIES.index(mapped_title) if mapped_title in TRAINED_JOB_CATEGORIES else 23

    features = np.array([[
        semantic_sim,
        skill_ratio,
        title_match,  # Binary, زي التدريب!
        float(age),
        float(exp_candidate),
        float(exp_diff),
        float(age_diff),  # Now using our fixed age_diff!
        float(job_cat)
    ]])
    
    model_score = model.predict(features)[0]

    # --- Slightly scale model score to give a boost (make it more "optimistic")
    # Scale from [0, 1] to [0.05, 0.98] then add 8 points, then clamp!
    scaled_model_score = (model_score * 0.93) + 0.07

    # --- Add reasonable manual bonuses/penalties
    title_bonus = 0.15 if title_match_ratio == 1.0 else 0.0
    loc_bonus = 0.05 if loc_score > 0.5 else 0.0

    final_score_raw = scaled_model_score + title_bonus + loc_bonus

    final_pct = max(0, min(100.0, final_score_raw * 100))

    return {
        "final_score": final_pct,
        "model_score": model_score * 100,
        "scaled_model_score": scaled_model_score * 100,
        "skill_match_ratio": skill_ratio,
        "title_match_ratio": title_match_ratio,
        "semantic_similarity": semantic_sim,
        "age_match_score": age_match_score,
        "experience_match_score": experience_match_score,
        "title_bonus": title_bonus * 100,
        "location_bonus": loc_bonus * 100
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
