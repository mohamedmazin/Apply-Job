import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import r2_score, mean_absolute_error

# Load model and data
print("Loading model and data...")
with open('super_stacking_model.pkl', 'rb') as f:
    model = pickle.load(f)

df = pd.read_csv('final_processed_data_with_scores.csv')
df = df.rename(columns={'title_match_feature': 'title_match'})

features = ['semantic_sim', 'skill_match_ratio', 'title_match', 'age', 'experiencere', 'exp_diff', 'age_diff', 'job_cat']
X = df[features].astype(float)
y = df['matched_score']

# Get predictions
print("Calculating predictions...")
y_pred = model.predict(X)

# Calculate regression metrics
r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)

print("\n" + "="*50)
print("REGRESSION METRICS")
print("="*50)
print(f"R2 Score: {r2:.4f} ({r2*100:.2f}%)")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print("="*50)
