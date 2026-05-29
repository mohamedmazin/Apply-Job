import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, KFold

def run_diagnostics():
    print("Starting Model Diagnostics...")
    
    # 1. Load Model and Data
    if not os.path.exists('super_stacking_model.pkl'):
        print("Error: super_stacking_model.pkl not found!")
        return

    with open('super_stacking_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Note: We need the processed features to test accurately
    # Let's recreate a small subset of features for testing if final_processed_data doesn't exist
    if os.path.exists('final_processed_data_with_scores.csv'):
        df = pd.read_csv('final_processed_data_with_scores.csv')
    else:
        print("Error: final_processed_data_with_scores.csv not found!")
        return

    # Map CSV columns to match expected feature names
    df = df.rename(columns={'title_match_feature': 'title_match'})
    
    features = ['semantic_sim', 'skill_match_ratio', 'title_match', 'age', 'experiencere', 'exp_diff', 'age_diff', 'job_cat']
    X = df[features].astype(float)
    y = df['matched_score']

    # 2. Check for Overfitting
    print("\n--- Overfitting Check ---")
    # In a real scenario, we'd use the original split, but let's check overall consistency
    y_pred = model.predict(X)
    overall_r2 = r2_score(y, y_pred)
    overall_mae = mean_absolute_error(y, y_pred)
    
    print(f"Overall R2 Score (on entire dataset): {overall_r2:.4f}")
    print(f"Overall MAE: {overall_mae:.4f}")
    
    # K-Fold Cross Validation (The real test for overfitting)
    print("\nRunning 5-Fold Cross Validation...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    # Note: StackingRegressor cross_val_score might be slow
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
    
    print(f"CV R2 Scores: {cv_scores}")
    print(f"Mean CV R2: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    if cv_scores.std() < 0.05:
        print("Result: Model is STABLE (Low variance between folds).")
    else:
        print("Warning: Model shows some variance between folds.")

    # 3. Check for Outliers
    print("\n--- Outlier Analysis ---")
    for col in ['age', 'experiencere']:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        outliers = df[(df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))]
        print(f"Outliers in '{col}': {len(outliers)} ({(len(outliers)/len(df))*100:.2f}%)")

    # 4. Stress Test (Edge Cases)
    print("\n--- Stress Testing ---")
    # Test with extreme values
    stress_data = pd.DataFrame([{
        'semantic_sim': 0.0, 'skill_match_ratio': 0.0, 'title_match': 0.0, 
        'age': 100.0, 'experiencere': 80.0, 'exp_diff': 50.0, 'age_diff': 50.0, 'job_cat': 0
    }])
    stress_pred = model.predict(stress_data)[0]
    print(f"Extreme Case Score (0% match, 80yr exp): {stress_pred*100:.2f}%")
    
    perfect_data = pd.DataFrame([{
        'semantic_sim': 1.0, 'skill_match_ratio': 1.0, 'title_match': 1.0, 
        'age': 25.0, 'experiencere': 5.0, 'exp_diff': 0.0, 'age_diff': 0.0, 'job_cat': 0
    }])
    perfect_pred = model.predict(perfect_data)[0]
    print(f"Perfect Case Score (100% match, perfect fit): {perfect_pred*100:.2f}%")

    print("\nDiagnostics Complete!")

if __name__ == "__main__":
    run_diagnostics()
