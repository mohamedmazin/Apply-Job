import pandas as pd
import numpy as np
import re
import ast
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
from sentence_transformers import SentenceTransformer, util
import torch
import os

# 1. تحميل البيانات
print("Loading Data...")
df = pd.read_csv("resume_data.csv")

# 2. تنظيف البيانات والـ Preprocessing
def extract_first_year(passing_year):
    if pd.isna(passing_year) or passing_year == "": return np.nan
    try:
        if isinstance(passing_year, str):
            passing_year = passing_year.replace('[', '').replace(']', '').replace("'", "").strip()
            years = passing_year.split(',')
            return float(years[0].strip())
        return float(passing_year)
    except: return np.nan

def get_all_years(date_input, default_current=2026):
    years = []
    if pd.isna(date_input) or date_input == "": return []
    if isinstance(date_input, str) and date_input.startswith('['):
        try: date_list = ast.literal_eval(date_input)
        except: date_list = [date_input]
    else: date_list = [date_input]
    for item in date_list:
        if pd.isna(item): continue
        item_str = str(item).strip()
        if any(kw in item_str for kw in ["Ongoing", "Till Date", "Current", "Present"]):
            years.append(default_current)
        else:
            try:
                dt = pd.to_datetime(item_str, errors='coerce')
                if pd.notnull(dt): years.append(dt.year)
            except: continue
    return years

def calculate_total_experience(row):
    start_years = get_all_years(row['start_dates'])
    end_years = get_all_years(row['end_dates'])
    if not start_years: return 0
    min_start = min(start_years)
    max_end = max(end_years) if end_years else 2026
    return max(0, max_end - min_start)

def extract_average_value(text):
    if pd.isna(text) or text == "" or str(text).lower() == 'nan': return np.nan
    text = str(text).replace('[', '').replace(']', '').replace("'", "")
    numbers = re.findall(r'\d+', text)
    numbers = [int(n) for n in numbers]
    if len(numbers) >= 2: return sum(numbers[:2]) / 2
    elif len(numbers) == 1: return float(numbers[0])
    return np.nan

# تطبيق الـ Cleaning
df['age'] = df['passing_years'].apply(extract_first_year).apply(lambda y: (2026 - y) + 22 if pd.notnull(y) else np.nan)
df['experiencere'] = df.apply(calculate_total_experience, axis=1)
df['age_requirement'] = df['age_requirement'].apply(extract_average_value)
df['experiencere_requirement'] = df['experiencere_requirement'].apply(extract_average_value)

df['age'] = df['age'].fillna(df['age'].mean())
df['age_requirement'] = df['age_requirement'].fillna(df['age_requirement'].mean())
df['experiencere'] = df['experiencere'].fillna(0)
df['experiencere_requirement'] = df['experiencere_requirement'].fillna(0)

# 3. Feature Engineering
def parse_list(x):
    if not x: return []
    try:
        if isinstance(x, str) and x.startswith('['): return ast.literal_eval(x)
        return [x]
    except: return [x]

def count_matching_skills(row):
    resume_skills = set([str(s).lower() for s in parse_list(row['skills'])])
    job_skills = set([str(s).lower() for s in parse_list(row['skills_required'])])
    if not job_skills: return 0
    return len(resume_skills.intersection(job_skills)) / len(job_skills)

def get_title_match(row):
    title = str(row['﻿job_position_name']).lower() if '﻿job_position_name' in row else ""
    resume_pos = str(row['positions']).lower()
    return 1.0 if title in resume_pos or resume_pos in title else 0.0

# Education Level Mapping (New Feature from Paper)
def get_education_level(degree_list_str):
    degrees = str(degree_list_str).lower()
    if 'phd' in degrees or 'doctor' in degrees: return 5
    if 'master' in degrees or 'm.tech' in degrees or 'm.s' in degrees: return 4
    if 'bachelor' in degrees or 'b.tech' in degrees or 'b.e' in degrees or 'b.s' in degrees: return 3
    if 'diploma' in degrees: return 2
    return 1

df['education_level'] = df['degree_names'].apply(get_education_level)
df['skill_match_ratio'] = df.apply(count_matching_skills, axis=1)
df['title_match_feature'] = df.apply(get_title_match, axis=1)

# 4. Text Similarity (TF-IDF + SBERT)
print("Computing TF-IDF and SBERT...")
df['all_resume_text'] = (df['positions'].astype(str) + " " + df['skills'].astype(str) + " " + df['career_objective'].astype(str))
df['all_job_text'] = (df['educationaL_requirements'].astype(str) + " " + df['skills_required'].astype(str) + " " + df['responsibilities.1'].astype(str))

# TF-IDF
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, stop_words='english')
all_text = pd.concat([df['all_resume_text'], df['all_job_text']])
tfidf_matrix = tfidf.fit_transform(all_text)
mid = len(df)
resume_vectors = tfidf_matrix[:mid]
job_vectors = tfidf_matrix[mid:]
df['tfidf_sim'] = [cosine_similarity(resume_vectors[i], job_vectors[i])[0][0] for i in range(mid)]

# SBERT (Using a small model to avoid long wait)
model_sbert = SentenceTransformer('all-MiniLM-L6-v2')
resume_embeddings = model_sbert.encode(df['all_resume_text'].tolist(), show_progress_bar=True)
job_embeddings = model_sbert.encode(df['all_job_text'].tolist(), show_progress_bar=True)
cos_sims = util.cos_sim(resume_embeddings, job_embeddings)
df['semantic_sim'] = cos_sims.diag().numpy()

# Differences
df['exp_diff'] = df['experiencere'] - df['experiencere_requirement']
df['age_diff'] = df['age'] - df['age_requirement']

# 5. Final Model (XGBoost)
# Adding job position as categorical
df['job_cat'] = df['﻿job_position_name'].astype('category').cat.codes

features = ['tfidf_sim', 'semantic_sim', 'skill_match_ratio', 'title_match_feature', 'education_level', 'age', 'experiencere', 'exp_diff', 'age_diff', 'job_cat']
X = df[features].astype(float)
y = df['matched_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training XGBoost...")
# Using better hyperparameters and early stopping for better accuracy
model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=10,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method='hist', # Faster
    random_state=42
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"\nFINAL ULTRA RESULTS:")
print(f"R2 Score: {r2:.4f}")
print(f"MAE: {mae:.4f}")

# Save the model and vectors for app11.py
import pickle
with open('ultra_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
df.to_csv('final_processed_data_with_scores.csv', index=False)

print("\nModel saved to ultra_model.pkl")
