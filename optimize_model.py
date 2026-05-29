import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import ast

# Load data
df = pd.read_csv("path_to_file.csv")

# Preprocessing
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
categorical_cols = df.select_dtypes(exclude=[np.number]).columns
for col in categorical_cols:
    df[col] = df[col].fillna("")

# 1. Feature Engineering: Skill Matching
def parse_list(x):
    if not x: return []
    try:
        return ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else [x]
    except:
        return [x]

def count_matching_skills(row):
    resume_skills = set([str(s).lower() for s in parse_list(row['skills'])])
    job_skills = set([str(s).lower() for s in parse_list(row['skills_required'])])
    if not job_skills: return 0
    return len(resume_skills.intersection(job_skills)) / len(job_skills)

df['skill_match_ratio'] = df.apply(count_matching_skills, axis=1)

# 2. Feature Engineering: Title Match
def title_match(row):
    title = str(row['﻿job_position_name']).lower()
    resume_pos = str(row['positions']).lower()
    return 1 if title in resume_pos or resume_pos in title else 0

df['title_match'] = df.apply(title_match, axis=1)

# Text Similarity
df['all_resume_text'] = (df['positions'].astype(str) + " " + df['skills'].astype(str))
df['all_job_text'] = (df['educationaL_requirements'].astype(str) + " " + df['skills_required'].astype(str))

tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, stop_words='english')
all_text = pd.concat([df['all_resume_text'], df['all_job_text']])
tfidf_matrix = tfidf.fit_transform(all_text)
mid = len(df)
resume_vectors = tfidf_matrix[:mid]
job_vectors = tfidf_matrix[mid:]
df['text_sim'] = [cosine_similarity(resume_vectors[i], job_vectors[i])[0][0] for i in range(mid)]

# Prepare X, y
df['exp_diff'] = df['experiencere'] - df['experiencere_requirement']
df['age_diff'] = df['age'] - df['age_requirement']

features = ['text_sim', 'skill_match_ratio', 'title_match', 'age', 'experiencere', 'exp_diff', 'age_diff']
X = df[features].astype(float)
y = df['matched_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost with Hyperparameter Tuning (Simplified for speed)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0]
}

grid = GridSearchCV(xgb.XGBRegressor(random_state=42), param_grid, cv=3, scoring='r2')
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print(f"Best Params: {grid.best_params_}")
print(f"Optimized R2: {r2_score(y_test, y_pred):.4f}")
print(f"Optimized MAE: {mean_absolute_error(y_test, y_pred):.4f}")
