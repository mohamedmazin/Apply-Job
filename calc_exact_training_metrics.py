import pandas as pd
import numpy as np
import re
import ast
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sentence_transformers import SentenceTransformer, util
import pickle

# 1. EXACTLY REPLICATE TRAINING PREPROCESSING (including augmentation)
print("Replicating training data prep...")
df = pd.read_csv("resume_data.csv")

# Augmentation (same as training)
def augment_data(df):
    df_aug = df.copy()
    df_aug['matched_score'] = df_aug['matched_score'] + np.random.normal(0, 0.01, len(df_aug))
    df_aug['matched_score'] = df_aug['matched_score'].clip(0, 1)
    return pd.concat([df, df_aug]).reset_index(drop=True)

df = augment_data(df)
print(f"Data size after augmentation: {len(df)}")

# Same preprocessing
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

df['age'] = df['passing_years'].apply(extract_first_year).apply(lambda y: (2026 - y) + 22 if pd.notnull(y) else np.nan)
df['experiencere'] = df.apply(calculate_total_experience, axis=1)

def extract_average_value(text):
    if pd.isna(text) or text == "" or str(text).lower() == 'nan': return np.nan
    text = str(text).replace('[', '').replace(']', '').replace("'", "")
    numbers = re.findall(r'\d+', text)
    numbers = [int(n) for n in numbers]
    if len(numbers) >= 2: return sum(numbers[:2]) / 2
    elif len(numbers) == 1: return float(numbers[0])
    return np.nan

df['age_requirement'] = df['age_requirement'].apply(extract_average_value)
df['experiencere_requirement'] = df['experiencere_requirement'].apply(extract_average_value)
df['age'] = df['age'].fillna(df['age'].mean())
df['age_requirement'] = df['age_requirement'].fillna(df['age_requirement'].mean())
df['experiencere'] = df['experiencere'].fillna(0)
df['experiencere_requirement'] = df['experiencere_requirement'].fillna(0)

def parse_list(x):
    if not x: return []
    try:
        if isinstance(x, str) and x.startswith('['): return ast.literal_eval(x)
        return [x]
    except: return [x]

df['skill_match_ratio'] = df.apply(lambda r: len(set(parse_list(r['skills'])).intersection(set(parse_list(r['skills_required'])))) / len(set(parse_list(r['skills_required']))) if parse_list(r['skills_required']) else 0.5, axis=1)
df['title_match'] = df.apply(lambda r: 1.0 if str(r['\ufeffjob_position_name']).lower() in str(r['positions']).lower() else 0.0, axis=1)

# SBERT embeddings (we don't need to recompute, we'll use the split)
print("Encoding texts (quick, small)...")
model_sbert = SentenceTransformer('all-MiniLM-L6-v2')
df['all_resume_text'] = (df['positions'].astype(str) + " " + df['skills'].astype(str))
df['all_job_text'] = (df['educationaL_requirements'].astype(str) + " " + df['skills_required'].astype(str))
resume_embeddings = model_sbert.encode(df['all_resume_text'].tolist(), batch_size=64)
job_embeddings = model_sbert.encode(df['all_job_text'].tolist(), batch_size=64)
cos_sims = util.cos_sim(resume_embeddings, job_embeddings)
df['semantic_sim'] = cos_sims.diag().numpy()

df['exp_diff'] = df['experiencere'] - df['experiencere_requirement']
df['age_diff'] = df['age'] - df['age_requirement']
df['job_cat'] = df['\ufeffjob_position_name'].astype('category').cat.codes

features = ['semantic_sim', 'skill_match_ratio', 'title_match', 'age', 'experiencere', 'exp_diff', 'age_diff', 'job_cat']
X = df[features].astype(float)
y = df['matched_score']

# Calculate sample weights (same as training)
y_class = (y >= 0.5).astype(int)
class_counts = y_class.value_counts()
total_samples = len(y_class)
weight_0 = total_samples / (2 * class_counts.get(0, 1))
weight_1 = total_samples / (2 * class_counts.get(1, 1))
sample_weights = np.where(y_class == 0, weight_0, weight_1)

# EXACT SAME TRAIN/TEST SPLIT (random_state=42)
X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(X, y, sample_weights, test_size=0.2, random_state=42)

# 2. Load the trained model and evaluate on EXACT TEST SET
print("\nLoading model and evaluating on test set...")
with open('super_stacking_model.pkl', 'rb') as f:
    model = pickle.load(f)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\n" + "="*50)
print("EXACT TRAINING TEST SET METRICS")
print("="*50)
print(f"R2 Score: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print("="*50)
