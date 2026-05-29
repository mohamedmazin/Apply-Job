import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

# Load data
df_cleaned = pd.read_csv("path_to_file.csv")

# Preprocessing
numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].mean())
categorical_cols = df_cleaned.select_dtypes(exclude=[np.number]).columns
for col in categorical_cols:
    if not df_cleaned[col].mode().empty:
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])
    else:
        df_cleaned[col] = df_cleaned[col].fillna("")

for col in ['age_requirement', 'experiencere_requirement', 'age', 'experiencere']:
    if col in df_cleaned.columns:
        df_cleaned[col] = df_cleaned[col].round().astype('Int64')

# Text Similarity
df_cleaned['all_resume_text'] = (
    df_cleaned['positions'].astype(str) + " " +
    df_cleaned['related_skils_in_job'].astype(str) + " " +
    df_cleaned['major_field_of_studies'].astype(str) + " " +
    df_cleaned['skills'].astype(str)
)
df_cleaned['all_job_text'] = (
    df_cleaned['educationaL_requirements'].astype(str) + " " +
    df_cleaned['responsibilities.1'].astype(str) + " " +
    df_cleaned['skills_required'].astype(str)
)

tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, stop_words='english', min_df=2)
tfidf_matrix = tfidf.fit_transform(pd.concat([df_cleaned['all_resume_text'], df_cleaned['all_job_text']]))
mid = len(df_cleaned)
resume_vectors = tfidf_matrix[:mid]
job_vectors = tfidf_matrix[mid:]
similarities = [cosine_similarity(resume_vectors[i], job_vectors[i])[0][0] for i in range(mid)]
df_cleaned['text_similarity_score'] = similarities

df_cleaned['exp_diff'] = df_cleaned['experiencere'] - df_cleaned['experiencere_requirement']
df_cleaned['age_diff'] = df_cleaned['age'] - df_cleaned['age_requirement']

X = df_cleaned[['text_similarity_score', 'age', 'experiencere', 'exp_diff', 'age_diff']]
y = df_cleaned['matched_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# RF Evaluation
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
r2_rf = r2_score(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)

# XGBoost Evaluation
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
r2_xgb = r2_score(y_test, y_pred_xgb)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)

print(f"RandomForest: R2={r2_rf:.4f}, MAE={mae_rf:.4f}")
print(f"XGBoost:      R2={r2_xgb:.4f}, MAE={mae_xgb:.4f}")
