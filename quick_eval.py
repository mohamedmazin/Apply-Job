import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

print("Loading data...")
df = pd.read_csv('final_processed_data_with_scores.csv')

# Rename column to match model's expected feature name
df = df.rename(columns={'title_match_feature': 'title_match'})

# Define features EXACTLY as the model expects
features = ['semantic_sim', 'skill_match_ratio', 'title_match', 'age', 'experiencere', 'exp_diff', 'age_diff', 'job_cat']
X = df[features].astype(float)
y = df['matched_score']

# EXACT same split as training (random_state=42, test_size=0.2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Loading model...")
with open('super_stacking_model.pkl', 'rb') as f:
    model = pickle.load(f)

print("Predicting...")
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\n" + "="*50)
print("TEST SET EVALUATION (ORIGINAL DATA, NO AUGMENTATION)")
print("="*50)
print(f"R2 Score: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print("="*50)
