"""
Streamlit Web Application for Bank Marketing Classification
Student ID: 2025aa05072
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
from io import StringIO

# Add model directory to path
sys.path.append('model')

from model.preprocessing import DataPreprocessor
from model.evaluate import (
    calculate_all_metrics,
    plot_confusion_matrix,
    get_classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Bank Marketing Classifier",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Force light theme and high visibility
st.markdown("""
    <style>
    /* Force light background everywhere */
    .stApp {
        background-color: #ffffff;
    }
    
    .main {
        background-color: #ffffff;
        color: #000000;
    }
    
    /* All text should be dark and visible */
    body, p, span, div, label, h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
    }
    
    /* Metric cards */
    .stMetric {
        background-color: #f0f0f0;
        padding: 20px;
        border-radius: 8px;
        border: 2px solid #cccccc;
    }
    
    .stMetric label {
        font-size: 18px !important;
        font-weight: 700 !important;
        color: #000000 !important;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        font-size: 36px !important;
        font-weight: 900 !important;
        color: #000000 !important;
    }
    
    /* Headers - all black */
    h1 {
        color: #000000 !important;
        font-size: 32px !important;
        font-weight: 800 !important;
    }
    
    h2 {
        color: #000000 !important;
        font-size: 24px !important;
        font-weight: 700 !important;
    }
    
    h3 {
        color: #000000 !important;
        font-size: 20px !important;
        font-weight: 700 !important;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #0066cc;
        color: #ffffff;
        font-size: 16px;
        font-weight: 700;
        padding: 12px 24px;
        border-radius: 6px;
    }
    
    /* Dropdown/Selectbox - light background */
    .stSelectbox > div > div {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    .stSelectbox [data-baseweb="select"] {
        background-color: #ffffff !important;
    }
    
    .stSelectbox [data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 2px solid #cccccc !important;
    }
    
    /* File uploader - light background */
    .stFileUploader > div {
        background-color: #f8f8f8 !important;
        border: 2px dashed #999999 !important;
    }
    
    .stFileUploader section {
        background-color: #f8f8f8 !important;
        color: #000000 !important;
    }
    
    .stFileUploader label, .stFileUploader span, .stFileUploader div {
        color: #000000 !important;
    }
    
    /* Input labels */
    .stSelectbox label, .stFileUploader label {
        color: #000000 !important;
        font-size: 16px !important;
        font-weight: 600 !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #f5f5f5;
    }
    
    [data-testid="stSidebar"] * {
        color: #000000 !important;
    }
    
    /* Success/Info/Warning/Error messages */
    .stSuccess {
        background-color: #d4edda;
        color: #155724 !important;
        font-weight: 600;
    }
    
    .stInfo {
        background-color: #d1ecf1;
        color: #0c5460 !important;
        font-weight: 600;
    }
    
    .stWarning {
        background-color: #fff3cd;
        color: #856404 !important;
        font-weight: 600;
    }
    
    .stError {
        background-color: #f8d7da;
        color: #721c24 !important;
        font-weight: 600;
    }
    
    /* Tables - light background with black text */
    .dataframe {
        background-color: #ffffff !important;
        color: #000000 !important;
        font-size: 15px !important;
    }
    
    .dataframe th {
        background-color: #f0f0f0 !important;
        color: #000000 !important;
        font-weight: 700 !important;
    }
    
    .dataframe td {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    .dataframe tr {
        background-color: #ffffff !important;
    }
    
    .dataframe tbody tr:nth-child(even) {
        background-color: #f8f8f8 !important;
    }
    
    /* Streamlit dataframe container */
    [data-testid="stDataFrame"] {
        background-color: #ffffff !important;
    }
    
    [data-testid="stDataFrame"] * {
        color: #000000 !important;
    }
    
    /* Table headers and cells */
    table {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    table th, table td {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    table thead {
        background-color: #e0e0e0 !important;
    }
    
    /* Radio buttons and other inputs */
    .stRadio label {
        color: #000000 !important;
        font-size: 16px !important;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load all trained models"""
    models = {}
    model_files = {
        'Logistic Regression': 'logistic_regression.pkl',
        'Decision Tree': 'decision_tree.pkl',
        'K-Nearest Neighbors': 'knn.pkl',
        'Naive Bayes': 'naive_bayes.pkl',
        'Random Forest': 'random_forest.pkl',
        'XGBoost': 'xgboost.pkl'
    }
    
    for model_name, filename in model_files.items():
        filepath = os.path.join('model', 'saved_models', filename)
        if os.path.exists(filepath):
            models[model_name] = joblib.load(filepath)
        else:
            st.error(f"Model file not found: {filepath}")
    
    return models


@st.cache_resource
def load_preprocessor():
    """Load the preprocessor"""
    preprocessor = DataPreprocessor()
    try:
        preprocessor.load_preprocessor('model/saved_models')
        return preprocessor
    except Exception as e:
        st.error(f"Error loading preprocessor: {e}")
        return None


@st.cache_data
def load_comparison_results():
    """Load model comparison results"""
    try:
        df = pd.read_csv('model/saved_models/model_comparison.csv', index_col=0)
        return df
    except:
        return None


def preprocess_uploaded_data(df, preprocessor):
    """Preprocess uploaded CSV data"""
    try:
        # Check if target column exists
        has_target = 'y' in df.columns
        
        if has_target:
            X, y = preprocessor.preprocess(df, fit=False)
            return X, y, has_target
        else:
            # Add dummy target for preprocessing
            df['y'] = 'no'
            X, _ = preprocessor.preprocess(df, fit=False)
            return X, None, has_target
    except Exception as e:
        st.error(f"Error preprocessing data: {e}")
        return None, None, False


def display_metrics(metrics):
    """Display metrics in a nice layout"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
        st.metric("Precision", f"{metrics['Precision']:.4f}")
    
    with col2:
        st.metric("Recall", f"{metrics['Recall']:.4f}")
        st.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
    
    with col3:
        if metrics['AUC Score'] is not None:
            st.metric("AUC Score", f"{metrics['AUC Score']:.4f}")
        else:
            st.metric("AUC Score", "N/A")
        st.metric("MCC", f"{metrics['MCC']:.4f}")


def main():
    # Header
    st.markdown("<h1>Bank Marketing Classification System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>Student ID: 2025aa05072 | ML Assignment 2</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        page = st.radio("Select Page", ["Model Prediction", "Model Comparison", "About"])
        
        st.markdown("---")
        st.header("Information")
        st.info("""
        Dataset: Bank Marketing  
        Task: Binary Classification  
        Target: Term Deposit Subscription  
        Models: 6 Classification Algorithms
        """)
        
        st.markdown("---")
        st.header("Links")
        st.markdown("[GitHub Repository](https://github.com/jjayakrishnan/2025aa05072_ml_assignment2)")
    
    # Main content based on page selection
    if page == "Model Prediction":
        show_prediction_page()
    elif page == "Model Comparison":
        show_comparison_page()
    else:
        show_about_page()


def show_prediction_page():
    """Main prediction page"""
    st.header("Model Prediction and Evaluation")
    
    # Load models and preprocessor
    with st.spinner("Loading models..."):
        models = load_models()
        preprocessor = load_preprocessor()
    
    if not models or not preprocessor:
        st.error("Failed to load models or preprocessor. Please ensure models are trained.")
        return
    
    st.success(f"Loaded {len(models)} models successfully")
    
    # Model selection
    st.subheader("Step 1: Select Model")
    selected_model = st.selectbox(
        "Choose a classification model:",
        list(models.keys()),
        help="Select the model you want to use for prediction"
    )
    
    # File upload
    st.subheader("Step 2: Upload Test Data")
    uploaded_file = st.file_uploader(
        "Upload CSV file with test data",
        type=['csv'],
        help="Upload a CSV file with the same format as the training data"
    )
    
    # Sample data option
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Use Sample Data"):
            uploaded_file = "sample"
    
    if uploaded_file is not None:
        # Load data
        if uploaded_file == "sample":
            if os.path.exists('data/sample_test.csv'):
                df = pd.read_csv('data/sample_test.csv')
                st.info("Using sample test data from data/sample_test.csv")
            else:
                st.error("Sample data not found. Please upload your own CSV file.")
                return
        else:
            df = pd.read_csv(uploaded_file)
        
        # Display data preview
        st.subheader("Step 3: Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        st.caption(f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
        
        # Preprocess data
        with st.spinner("Preprocessing data..."):
            X, y_true, has_target = preprocess_uploaded_data(df, preprocessor)
        
        if X is None:
            st.error("Failed to preprocess data. Please check the data format.")
            return
        
        # Make predictions
        st.subheader("Step 4: Predictions and Evaluation")
        
        with st.spinner(f"Running {selected_model}..."):
            model = models[selected_model]
            y_pred = model.predict(X)
            
            # Get probabilities
            try:
                y_pred_proba = model.predict_proba(X)
            except:
                y_pred_proba = None
        
        # Show results
        if has_target:
            # Calculate metrics
            metrics = calculate_all_metrics(y_true, y_pred, y_pred_proba)
            
            st.success("Evaluation Complete")
            
            # Display metrics
            st.subheader("Evaluation Metrics")
            display_metrics(metrics)
            
            # Confusion Matrix
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(8, 6))
            cm = plt.cm.Blues
            sns.heatmap(
                pd.crosstab(y_true, y_pred, rownames=['Actual'], colnames=['Predicted']),
                annot=True,
                fmt='d',
                cmap=cm,
                xticklabels=['No', 'Yes'],
                yticklabels=['No', 'Yes'],
                ax=ax
            )
            ax.set_title(f'{selected_model} - Confusion Matrix', fontsize=14, fontweight='bold')
            st.pyplot(fig)
            
            # Classification Report
            st.subheader("Classification Report")
            report = get_classification_report(y_true, y_pred)
            report_df = pd.DataFrame(report).T
            st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)
            
        else:
            st.warning("No target column found. Showing predictions only.")
            
            # Show predictions
            st.subheader("Predictions")
            predictions_df = df.copy()
            predictions_df['Prediction'] = ['Yes' if p == 1 else 'No' for p in y_pred]
            
            if y_pred_proba is not None:
                predictions_df['Confidence'] = y_pred_proba.max(axis=1)
            
            st.dataframe(predictions_df, use_container_width=True)
            
            # Download predictions
            csv = predictions_df.to_csv(index=False)
            st.download_button(
                label="Download Predictions",
                data=csv,
                file_name=f"{selected_model.lower().replace(' ', '_')}_predictions.csv",
                mime="text/csv"
            )


def show_comparison_page():
    """Model comparison page"""
    st.header("Model Comparison")
    
    # Load comparison results
    results_df = load_comparison_results()
    
    if results_df is not None:
        st.subheader("Performance Metrics Comparison")
        
        # Display table
        st.dataframe(
            results_df.style.format("{:.4f}").background_gradient(cmap='RdYlGn', axis=0),
            use_container_width=True
        )
        
        # Best model
        best_model = results_df.index[0]
        best_f1 = results_df.loc[best_model, 'F1 Score']
        
        st.success(f"Best Model: {best_model} (F1 Score: {best_f1:.4f})")
        
        # Visualization
        st.subheader("Visual Comparison")
        
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'MCC']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        results_df[metrics_to_plot].plot(kind='bar', ax=ax, width=0.8, colormap='viridis')
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xlabel('Models', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_xticklabels(results_df.index, rotation=45, ha='right')
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        st.pyplot(fig)
        
    else:
        st.warning("Model comparison results not found. Please train the models first.")


def show_about_page():
    """About page"""
    st.header("About This Project")
    
    st.markdown("""
    ### ML Assignment 2
    Student ID: 2025aa05072  
    Course: M.Tech (AIML)  
    Institution: BITS Pilani
    
    ---
    
    ### Dataset
    Name: Bank Marketing Dataset  
    Source: UCI Machine Learning Repository  
    Task: Binary Classification  
    Target: Predict if a client will subscribe to a term deposit (yes/no)  
    Features: 20 features (10 numerical, 10 categorical)  
    Instances: 41,188 samples
    
    ---
    
    ### Models Implemented
    
    1. Logistic Regression - Linear classifier with regularization
    2. Decision Tree - Tree-based rule learning
    3. K-Nearest Neighbors - Instance-based learning
    4. Naive Bayes - Probabilistic classifier
    5. Random Forest - Ensemble of decision trees
    6. XGBoost - Gradient boosting ensemble
    
    ---
    
    ### Evaluation Metrics
    
    - Accuracy - Overall correctness
    - AUC Score - Area under ROC curve
    - Precision - Positive predictive value
    - Recall - Sensitivity/True positive rate
    - F1 Score - Harmonic mean of precision and recall
    - MCC - Matthews Correlation Coefficient
    
    ---
    
    ### How to Use
    
    1. Navigate to Model Prediction page
    2. Select a classification model from the dropdown
    3. Upload your test data CSV file (or use sample data)
    4. View predictions and evaluation metrics
    5. Analyze confusion matrix and classification report
    
    ---
    
    ### Resources
    
    - GitHub: [github.com/jjayakrishnan/2025aa05072_ml_assignment2](https://github.com/jjayakrishnan/2025aa05072_ml_assignment2)
    - Dataset: [UCI Bank Marketing](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
    
    ---
    
    ### Notes
    
    - All models are trained on 80% of the data
    - Test set contains 20% of the data
    - Random state is set to 42 for reproducibility
    - Models use default hyperparameters (can be tuned)
    
    """)


if __name__ == "__main__":
    main()
