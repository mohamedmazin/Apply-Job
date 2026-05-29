# AI Hiring Platform API - FastAPI Backend

## Overview
This is a FastAPI service that provides AI-powered resume matching and applicant ranking for a job application platform. It's ready to deploy on Render.

## Key Features
- 📄 **PDF CV Extraction**: Extracts structured data from PDF resumes (skills, education, experience, age, address)
- 🎯 **Match Score Calculation**: Calculates a match score (0-100) between a candidate and a job posting
- 🏆 **Applicant Ranking**: Ranks multiple applicants based on their match scores
- 🧠 **Advanced AI**: Uses SBERT (Semantic BERT) for understanding context + Stacking Ensemble (XGBoost + LightGBM + CatBoost) for final scoring
- ❌ **Language Feature Removed**: Language bonus/penalty feature has been removed as requested

## API Endpoints

### 1. Root & Health Check
- `GET /` - Welcome message
- `GET /health` - Health check endpoint

### 2. Extract Data from PDF CV
- `POST /api/extract-cv`
  - Accepts: PDF file upload
  - Returns: Structured candidate data (skills, education, age, experience, etc.)

### 3. Calculate Match Score
- `POST /api/calculate-score`
  - Accepts: Candidate data + Job post data
  - Returns: Final match score + breakdown (skill match, title match, semantic similarity)

### 4. Rank Multiple Applicants
- `POST /api/rank-applicants`
  - Accepts: Job post data + List of applicants
  - Returns: Sorted list of applicants by score (ranked 1, 2, 3...)

## Request/Response Examples

### Extract CV Request
Send a PDF file via Form Data:
```
curl -X POST "http://your-domain/api/extract-cv" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@resume.pdf;type=application/pdf"
```

### Calculate Score Request Body
```json
{
  "candidate_data": {
    "candidate_name": "John Doe",
    "age": 28,
    "experiencere": 5,
    "address": "New York, USA",
    "skills": "Python, JavaScript, React",
    "education": "Bachelors in Computer Science",
    "positions": "Senior Developer",
    "full_text": "Full resume text..."
  },
  "job_post": {
    "job_title": "Software Engineer",
    "required_skills": "Python, JavaScript, React",
    "target_location": "New York, USA",
    "min_experience": 3,
    "target_age": 30,
    "job_description": "We are looking for a senior software engineer..."
  }
}
```

### Calculate Score Response
```json
{
  "final_score": 87.5,
  "skill_match_ratio": 0.9,
  "title_match_ratio": 1.0,
  "semantic_similarity": 0.85
}
```

### Rank Applicants Request
```json
{
  "job_post": {...},
  "applicants": [
    {"candidate_name": "John Doe", ...},
    {"candidate_name": "Jane Smith", ...}
  ]
}
```

### Ranked Applicants Response
```json
[
  {
    "rank": 1,
    "name": "John Doe",
    "score": 87.5,
    "skill_match_ratio": 0.9,
    "title_match_ratio": 1.0,
    "semantic_similarity": 0.85,
    "location": "New York, USA",
    "experience": 5
  },
  {
    "rank": 2,
    "name": "Jane Smith",
    "score": 75.2,
    ...
  }
]
```

## How to Deploy on Render

### Step 1: Push to GitHub
1. Create a GitHub repository
2. Upload these files to the repo:
   - `main.py` (the FastAPI app)
   - `requirements.txt`
   - `render.yaml`
   - `super_stacking_model.pkl` (IMPORTANT: This is your trained model file)

### Step 2: Deploy on Render
1. Sign in to [Render.com](https://render.com)
2. Click "New" → "Web Service"
3. Connect your GitHub repo
4. Configure the service:
   - **Name**: ai-hiring-api
   - **Region**: Choose your region
   - **Branch**: main (or your default branch)
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port 10000`
5. Click "Create Web Service"

### Step 3: Test Your API
Once deployed, go to `https://your-service.onrender.com/docs` to see the Swagger UI and test the endpoints!

## How the AI Works
1. **Semantic Similarity (SBERT)**: Understands the meaning of the CV and job description
2. **Skill Match**: Checks overlap between candidate skills and required skills
3. **Title Match**: Ensures the candidate's role matches the job title
4. **Ensemble Model**: Combines XGBoost + LightGBM + CatBoost to calculate the final score
5. **Domain Logic**: Adds bonuses for good matches and penalties for mismatches

## How to Run Locally
1. Install dependencies: `pip install -r requirements.txt`
2. Make sure `super_stacking_model.pkl` is in the same folder
3. Run: `python main.py`
4. Visit: `http://localhost:8000/docs` to test via Swagger UI

## Integration with Your Backend
Your backend (Node.js/Laravel/...) should:
1. Let users upload PDF → send to `/api/extract-cv` → store the extracted data in DB
2. Let users edit the extracted data (age, skills, address) before submitting
3. When user applies, send candidate data + job data to `/api/calculate-score`
4. Store the final score in the database
5. When company views job post, fetch all applicants, send to `/api/rank-applicants`, and display ranked list

## Technical Stack
- **FastAPI**: Web framework
- **SBERT**: Sentence embeddings for semantic similarity
- **XGBoost + LightGBM + CatBoost**: ML models for scoring
- **PyMuPDF**: PDF extraction
- **Uvicorn**: ASGI server
