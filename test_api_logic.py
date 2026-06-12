import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import calculate_match_score, load_models, CandidateData, JobPostData

# Load models first
model, sbert_model = load_models()

print("Testing with two different candidate/job pairs:\n")
print("=" * 100)

# --- Test 1: PERFECT MATCH ---
print("\nTest 1: PERFECT MATCH (high similarity, good age, good exp)")
job1 = JobPostData(
    title="Senior Software Engineer",
    description="We are looking for an experienced Python developer with React skills",
    location="Cairo",
    min_age=25,
    max_age=40,
    experience_level="Senior",
    required_skills="Python, React, JavaScript"
)
candidate1 = CandidateData(
    age=30,
    years_of_experience=7,
    address="Cairo, Egypt",
    skills_extracted="Python, React, JavaScript, FastAPI",
    highest_education="Bachelor's Degree",
    education_details_extracted="Bachelor of Computer Science",
    full_text="Senior Software Engineer with 7+ years of experience in Python, React, JavaScript, FastAPI, working in Cairo"
)
result1 = calculate_match_score(candidate1, job1)
print(f"   Model Score: {result1['model_score']:.2f}%")
print(f"   Final Score: {result1['final_score']:.2f}%")

# --- Test 2: TOTALLY BAD MATCH ---
print("\nTest 2: BAD MATCH (no skills, too old, no exp)")
job2 = JobPostData(
    title="Junior Data Scientist",
    description="Entry level data scientist with Python, Machine Learning",
    location="Remote",
    min_age=20,
    max_age=30,
    experience_level="Entry",
    required_skills="Python, Machine Learning, Pandas, NumPy"
)
candidate2 = CandidateData(
    age=70,
    years_of_experience=0,
    address="Alexandria",
    skills_extracted="Word, Excel, PowerPoint",
    highest_education="High School",
    education_details_extracted="High School Diploma",
    full_text="Recent high school graduate, no programming experience"
)
result2 = calculate_match_score(candidate2, job2)
print(f"   Model Score: {result2['model_score']:.2f}%")
print(f"   Final Score: {result2['final_score']:.2f}%")

print("\n" + "=" * 100)
print("\n✅ Done! Now you can see the raw model_score vs. final_score!")