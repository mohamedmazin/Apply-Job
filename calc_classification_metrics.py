import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, f1_score

# Load model and data
with open('super_stacking_model.pkl', 'rb') as f:
    model = pickle.load(f)

df = pd.read_csv('final_processed_data_with_scores.csv')
df = df.rename(columns={'title_match_feature': 'title_match'})

features = ['semantic_sim', 'skill_match_ratio', 'title_match', 'age', 'experiencere', 'exp_diff', 'age_diff', 'job_cat']
X = df[features].astype(float)
y = df['matched_score']

# Get predictions
preds = model.predict(X)

# --- Convert to Binary Classification ---
# Threshold: >= 0.5 = "Match", < 0.5 = "No Match"
threshold = 0.5

y_true_binary = (y >= threshold).astype(int)
y_pred_binary = (preds >= threshold).astype(int)

# Calculate metrics
print("=" * 70)
print(f"MODEL AS BINARY CLASSIFIER (Threshold: {threshold:.1f})")
print("=" * 70)
print(f"F1 Score: {f1_score(y_true_binary, y_pred_binary):.4f}")
print("\nConfusion Matrix:")
cm = confusion_matrix(y_true_binary, y_pred_binary)
print(cm)
print(f"\n  TN: {cm[0][0]}  FP: {cm[0][1]}")
print(f"  FN: {cm[1][0]}  TP: {cm[1][1]}")
print("\nClassification Report:")
print(classification_report(y_true_binary, y_pred_binary))
print("=" * 70)