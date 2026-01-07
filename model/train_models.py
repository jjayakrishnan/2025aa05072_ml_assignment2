"""
Train all 6 classification models and save them
"""
import os
import sys
import joblib
import json
import numpy as np
import pandas as pd
from datetime import datetime

# Scikit-learn models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# XGBoost
from xgboost import XGBClassifier

# Local modules
from preprocessing import DataPreprocessor
from evaluate import (
    calculate_all_metrics,
    print_metrics,
    plot_confusion_matrix,
    get_classification_report,
    compare_models,
    plot_model_comparison
)


def train_all_models(X_train, y_train, random_state=42):
    """
    Train all 6 classification models
    
    Returns:
        Dictionary of trained models
    """
    print("\n" + "="*60)
    print("TRAINING ALL MODELS")
    print("="*60)
    
    models = {}
    
    # 1. Logistic Regression
    print("\n[1/6] Training Logistic Regression...")
    models['Logistic Regression'] = LogisticRegression(
        random_state=random_state,
        max_iter=1000,
        solver='lbfgs'
    )
    models['Logistic Regression'].fit(X_train, y_train)
    print("✓ Logistic Regression trained")
    
    # 2. Decision Tree
    print("\n[2/6] Training Decision Tree...")
    models['Decision Tree'] = DecisionTreeClassifier(
        random_state=random_state,
        max_depth=10,
        min_samples_split=20
    )
    models['Decision Tree'].fit(X_train, y_train)
    print("✓ Decision Tree trained")
    
    # 3. K-Nearest Neighbors
    print("\n[3/6] Training K-Nearest Neighbors...")
    models['K-Nearest Neighbors'] = KNeighborsClassifier(
        n_neighbors=5,
        weights='uniform'
    )
    models['K-Nearest Neighbors'].fit(X_train, y_train)
    print("✓ K-Nearest Neighbors trained")
    
    # 4. Naive Bayes
    print("\n[4/6] Training Naive Bayes...")
    models['Naive Bayes'] = GaussianNB()
    models['Naive Bayes'].fit(X_train, y_train)
    print("✓ Naive Bayes trained")
    
    # 5. Random Forest
    print("\n[5/6] Training Random Forest...")
    models['Random Forest'] = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        max_depth=15,
        min_samples_split=20,
        n_jobs=-1
    )
    models['Random Forest'].fit(X_train, y_train)
    print("✓ Random Forest trained")
    
    # 6. XGBoost
    print("\n[6/6] Training XGBoost...")
    models['XGBoost'] = XGBClassifier(
        n_estimators=100,
        random_state=random_state,
        max_depth=6,
        learning_rate=0.1,
        eval_metric='logloss',
        use_label_encoder=False
    )
    models['XGBoost'].fit(X_train, y_train)
    print("✓ XGBoost trained")
    
    print("\n" + "="*60)
    print("ALL MODELS TRAINED SUCCESSFULLY!")
    print("="*60)
    
    return models


def evaluate_all_models(models, X_test, y_test):
    """
    Evaluate all models and return results
    
    Returns:
        Dictionary with evaluation results for each model
    """
    print("\n" + "="*60)
    print("EVALUATING ALL MODELS")
    print("="*60)
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Get probabilities (if available)
        try:
            y_pred_proba = model.predict_proba(X_test)
        except:
            y_pred_proba = None
        
        # Calculate metrics
        metrics = calculate_all_metrics(y_test, y_pred, y_pred_proba)
        results[model_name] = metrics
        
        # Print metrics
        print_metrics(metrics, model_name)
    
    return results


def save_models(models, save_dir='model/saved_models'):
    """Save all trained models"""
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("SAVING MODELS")
    print("="*60)
    
    model_files = {
        'Logistic Regression': 'logistic_regression.pkl',
        'Decision Tree': 'decision_tree.pkl',
        'K-Nearest Neighbors': 'knn.pkl',
        'Naive Bayes': 'naive_bayes.pkl',
        'Random Forest': 'random_forest.pkl',
        'XGBoost': 'xgboost.pkl'
    }
    
    for model_name, filename in model_files.items():
        filepath = os.path.join(save_dir, filename)
        joblib.dump(models[model_name], filepath)
        print(f"✓ Saved {model_name} to {filepath}")
    
    print("\n" + "="*60)
    print("ALL MODELS SAVED!")
    print("="*60)


def save_results(results, save_path='model/saved_models/evaluation_results.json'):
    """Save evaluation results to JSON"""
    # Convert to serializable format
    results_serializable = {}
    for model_name, metrics in results.items():
        results_serializable[model_name] = {
            k: float(v) if v is not None else None 
            for k, v in metrics.items()
        }
    
    with open(save_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"\n✓ Results saved to {save_path}")


def main():
    """Main training pipeline"""
    print("\n" + "="*60)
    print("BANK MARKETING CLASSIFICATION - MODEL TRAINING")
    print("Student ID: 2025aa05072")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Step 1: Load and preprocess data
    print("\n[STEP 1] Loading and preprocessing data...")
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data()
    
    # Preprocess
    X, y = preprocessor.preprocess(df, fit=True)
    
    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    
    # Save preprocessor
    preprocessor.save_preprocessor()
    
    # Step 2: Train all models
    print("\n[STEP 2] Training all models...")
    models = train_all_models(X_train, y_train)
    
    # Step 3: Evaluate all models
    print("\n[STEP 3] Evaluating all models...")
    results = evaluate_all_models(models, X_test, y_test)
    
    # Step 4: Create comparison
    print("\n[STEP 4] Creating model comparison...")
    results_df = compare_models(results)
    print("\nModel Comparison Table:")
    print(results_df.to_string())
    
    # Save comparison table
    results_df.to_csv('model/saved_models/model_comparison.csv')
    print("\n✓ Comparison table saved to model/saved_models/model_comparison.csv")
    
    # Plot comparison
    plot_model_comparison(results_df, 'model/saved_models/model_comparison.png')
    
    # Step 5: Generate confusion matrices for all models
    print("\n[STEP 5] Generating confusion matrices...")
    os.makedirs('model/saved_models/confusion_matrices', exist_ok=True)
    
    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        filename = model_name.lower().replace(' ', '_').replace('-', '_')
        save_path = f'model/saved_models/confusion_matrices/{filename}_cm.png'
        plot_confusion_matrix(y_test, y_pred, f"{model_name} - Confusion Matrix", save_path)
    
    # Step 6: Save models and results
    print("\n[STEP 6] Saving models and results...")
    save_models(models)
    save_results(results)
    
    # Step 7: Find best model
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    
    best_model = results_df.index[0]
    best_f1 = results_df.loc[best_model, 'F1 Score']
    
    print(f"\n🏆 Best Model: {best_model}")
    print(f"   F1 Score: {best_f1:.4f}")
    print(f"\nAll results saved to: model/saved_models/")
    print(f"View comparison: model/saved_models/model_comparison.csv")
    
    return models, results, results_df


if __name__ == "__main__":
    models, results, results_df = main()
