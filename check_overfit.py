import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# Load model and data
with open('super_stacking_model.pkl', 'rb') as f:
    model = pickle.load(f)

df = pd.read_csv('final_processed_data_with_scores.csv')
df = df.rename(columns={'title_match_feature': 'title_match'})

features = ['semantic_sim', 'skill_match_ratio', 'title_match', 'age', 'experiencere', 'exp_diff', 'age_diff', 'job_cat']
X = df[features].astype(float)
y = df['matched_score']

# --- 1. Check Train vs Test Performance ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

train_r2 = r2_score(y_train, train_preds)
test_r2 = r2_score(y_test, test_preds)
train_mae = mean_absolute_error(y_train, train_preds)
test_mae = mean_absolute_error(y_test, test_preds)

# --- 2. Cross-Validation ---
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2 = cross_val_score(model, X, y, cv=kf, scoring='r2')

print("=" * 70)
print("OVERFITTING CHECK")
print("=" * 70)

print("\n1. Train vs Test Performance:")
print(f"   Train R2: {train_r2:.4f}")
print(f"   Test  R2: {test_r2:.4f}")
print(f"   Train MAE: {train_mae:.4f}")
print(f"   Test  MAE: {test_mae:.4f}")

diff_r2 = train_r2 - test_r2
if diff_r2 > 0.1:
    print(f"\n   ⚠️  WARNING: Possible OVERFITTING (Train R2 - Test R2 = {diff_r2:.4f})")
else:
    print(f"\n   ✅ OKAY: Model is generalizing well (Train R2 - Test R2 = {diff_r2:.4f})")

print("\n2. Cross-Validation R2 Scores (5-Fold):")
for i, score in enumerate(cv_r2, 1):
    print(f"   Fold {i}: {score:.4f}")
print(f"\n   Mean CV R2: {cv_r2.mean():.4f} (+/- {cv_r2.std() * 2:.4f})")

if cv_r2.std() > 0.1:
    print(f"\n   ⚠️  WARNING: Model is unstable (high variance in folds)")
else:
    print(f"\n   ✅ OKAY: Model is stable")

print("=" * 70)