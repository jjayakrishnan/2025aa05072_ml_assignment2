# Bank Marketing Classification - ML Assignment 2

**Student ID:** 2025aa05072  
**Course:** M.Tech (AIML)  
**Institution:** BITS Pilani, Work Integrated Learning Programmes Division

---

##  Table of Contents

- [Project Overview](#project-overview)
- [Dataset Information](#dataset-information)
- [Models Implemented](#models-implemented)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Evaluation Results](#evaluation-results)
- [Streamlit Application](#streamlit-application)
- [Deployment](#deployment)
- [Observations & Insights](#observations--insights)
- [Screenshots](#screenshots)
- [References](#references)

---

##  Project Overview

This project implements an end-to-end machine learning classification pipeline for predicting bank term deposit subscriptions. The system includes:

- **6 Classification Models** trained and evaluated
- **6 Evaluation Metrics** for comprehensive performance analysis
- **Interactive Streamlit Web Application** for model deployment
- **Complete MLOps Pipeline** from data preprocessing to deployment

### Objective

Predict whether a client will subscribe to a term deposit (yes/no) based on demographic and campaign-related features.

---

##  Dataset Information

### Bank Marketing Dataset

- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
- **Type:** Binary Classification
- **Instances:** 41,188 samples
- **Features:** 20 features (10 numerical, 10 categorical)
- **Target Variable:** `y` (yes/no - will the client subscribe to a term deposit?)
- **Class Distribution:** Imbalanced (majority class: no)

### Features

**Numerical Features (10):**
- `age` - Age of the client
- `duration` - Last contact duration in seconds
- `campaign` - Number of contacts during this campaign
- `pdays` - Days since last contact from previous campaign
- `previous` - Number of contacts before this campaign
- `emp.var.rate` - Employment variation rate
- `cons.price.idx` - Consumer price index
- `cons.conf.idx` - Consumer confidence index
- `euribor3m` - Euribor 3 month rate
- `nr.employed` - Number of employees

**Categorical Features (10):**
- `job` - Type of job
- `marital` - Marital status
- `education` - Education level
- `default` - Has credit in default?
- `housing` - Has housing loan?
- `loan` - Has personal loan?
- `contact` - Contact communication type
- `month` - Last contact month
- `day_of_week` - Last contact day of the week
- `poutcome` - Outcome of previous marketing campaign

---

##  Models Implemented

### 1. Logistic Regression
- **Type:** Linear classifier
- **Library:** `sklearn.linear_model.LogisticRegression`
- **Hyperparameters:** max_iter=1000, solver='lbfgs'
- **Use Case:** Baseline model, interpretable coefficients

### 2. Decision Tree Classifier
- **Type:** Tree-based classifier
- **Library:** `sklearn.tree.DecisionTreeClassifier`
- **Hyperparameters:** max_depth=10, min_samples_split=20
- **Use Case:** Non-linear decision boundaries, feature importance

### 3. K-Nearest Neighbors (KNN)
- **Type:** Instance-based learning
- **Library:** `sklearn.neighbors.KNeighborsClassifier`
- **Hyperparameters:** n_neighbors=5, weights='uniform'
- **Use Case:** Non-parametric, local decision boundaries

### 4. Naive Bayes (Gaussian)
- **Type:** Probabilistic classifier
- **Library:** `sklearn.naive_bayes.GaussianNB`
- **Hyperparameters:** Default
- **Use Case:** Fast training, probabilistic predictions

### 5. Random Forest
- **Type:** Ensemble (Bagging)
- **Library:** `sklearn.ensemble.RandomForestClassifier`
- **Hyperparameters:** n_estimators=100, max_depth=15, min_samples_split=20
- **Use Case:** Robust to overfitting, feature importance

### 6. XGBoost
- **Type:** Ensemble (Boosting)
- **Library:** `xgboost.XGBClassifier`
- **Hyperparameters:** n_estimators=100, max_depth=6, learning_rate=0.1
- **Use Case:** State-of-the-art performance, handles imbalanced data

---

##  Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/jjayakrishnan/2025aa05072_ml_assignment2.git
   cd 2025aa05072_ml_assignment2
   ```

2. **Create virtual environment (recommended)**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

##  Usage

### 1. Train Models

Run the training script to train all 6 models:

```bash
cd model
python train_models.py
```

This will:
- Load and preprocess the Bank Marketing dataset
- Train all 6 classification models
- Evaluate models on test set
- Save trained models to `model/saved_models/`
- Generate evaluation metrics and visualizations

**Expected Output:**
- Trained model files (`.pkl`)
- Model comparison table (`model_comparison.csv`)
- Confusion matrices for each model
- Evaluation results (`evaluation_results.json`)

### 2. Run Streamlit Application

Launch the interactive web application:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### 3. Use the Web Application

1. **Select Model:** Choose from 6 trained models
2. **Upload Data:** Upload CSV file with test data
3. **View Results:** See predictions, metrics, and confusion matrix
4. **Compare Models:** Navigate to comparison page to see all model performances

---

##  Project Structure

```
2025aa05072_ml_assignment2/
├── app.py                              # Streamlit web application
├── requirements.txt                    # Python dependencies
├── README.md                          # Project documentation
├── .gitignore                         # Git ignore file
│
├── data/
│   ├── bank-additional-full.csv       # Training dataset (41,188 samples)
│   └── sample_test.csv                # Sample test data for demo
│
├── model/
│   ├── preprocessing.py               # Data preprocessing utilities
│   ├── train_models.py                # Model training script
│   ├── evaluate.py                    # Evaluation metrics utilities
│   │
│   └── saved_models/                  # Trained models and results
│       ├── logistic_regression.pkl    # Logistic Regression model
│       ├── decision_tree.pkl          # Decision Tree model
│       ├── knn.pkl                    # K-Nearest Neighbors model
│       ├── naive_bayes.pkl            # Naive Bayes model
│       ├── random_forest.pkl          # Random Forest model
│       ├── xgboost.pkl                # XGBoost model
│       ├── scaler.pkl                 # Feature scaler
│       ├── label_encoders.pkl         # Categorical encoders
│       ├── feature_names.pkl          # Feature names
│       ├── evaluation_results.json    # Evaluation metrics
│       ├── model_comparison.csv       # Model comparison table
│       ├── model_comparison.png       # Comparison visualization
│       │
│       └── confusion_matrices/        # Confusion matrix plots
│           ├── logistic_regression_cm.png
│           ├── decision_tree_cm.png
│           ├── knn_cm.png
│           ├── naive_bayes_cm.png
│           ├── random_forest_cm.png
│           └── xgboost_cm.png
│
└── notebooks/
    └── exploratory_analysis.ipynb     # (Optional) EDA notebook
```

---

##  Evaluation Results

### Evaluation Metrics

All models are evaluated using 6 comprehensive metrics:

1. **Accuracy** - Overall correctness of predictions
2. **AUC Score** - Area Under the ROC Curve (discrimination ability)
3. **Precision** - Positive Predictive Value (TP / (TP + FP))
4. **Recall** - Sensitivity/True Positive Rate (TP / (TP + FN))
5. **F1 Score** - Harmonic mean of Precision and Recall
6. **MCC** - Matthews Correlation Coefficient (balanced measure)

### Model Performance Comparison

| Model | Accuracy | AUC Score | Precision | Recall | F1 Score | MCC |
|-------|----------|-----------|-----------|--------|----------|-----|
| **XGBoost** | **0.9219** | **0.9547** | **0.6829** | **0.5733** | **0.6233** | **0.5829** |
| Decision Tree | 0.9161 | 0.9213 | 0.6357 | 0.5981 | 0.6163 | 0.5696 |
| Random Forest | 0.9211 | 0.9534 | 0.7050 | 0.5151 | 0.5953 | 0.5611 |
| Logistic Regression | 0.9138 | 0.9370 | 0.7004 | 0.4106 | 0.5177 | 0.4943 |
| Naive Bayes | 0.8536 | 0.8606 | 0.4024 | 0.6175 | 0.4872 | 0.4189 |
| K-Nearest Neighbors | 0.9026 | 0.8596 | 0.6098 | 0.3772 | 0.4660 | 0.4303 |

**Best Model:** XGBoost (F1 Score: 0.6233)

---

##  Streamlit Application

### Features

1. **Model Selection**
   - Dropdown menu to select from 6 trained models
   - Dynamic model loading

2. **CSV Upload**
   - Upload custom test data
   - Sample data option for quick testing
   - Data preview with shape information

3. **Predictions & Evaluation**
   - Real-time predictions
   - All 6 evaluation metrics displayed
   - Interactive confusion matrix heatmap
   - Detailed classification report

4. **Model Comparison**
   - Side-by-side performance comparison
   - Visual bar charts
   - Best model highlighting

5. **About Page**
   - Project information
   - Dataset details
   - Model descriptions
   - Usage instructions

### User Interface

- **Clean & Professional Design** with custom CSS styling
- **Responsive Layout** with Streamlit columns
- **Interactive Visualizations** using Matplotlib and Seaborn
- **Color-coded Metrics** for easy interpretation
- **Download Options** for predictions

---

##  Deployment

### GitHub Repository

**URL:** [https://github.com/jjayakrishnan/2025aa05072_ml_assignment2](https://github.com/jjayakrishnan/2025aa05072_ml_assignment2)

### Streamlit Community Cloud

**Live App URL:** TBD (will be updated after deployment)

### Deployment Steps

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: ML Assignment 2"
   git branch -M main
   git remote add origin https://github.com/jjayakrishnan/2025aa05072_ml_assignment2.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect GitHub repository
   - Select `app.py` as main file
   - Click "Deploy"

3. **BITS Virtual Lab Execution**
   - Clone repository in Virtual Lab
   - Install dependencies
   - Run application
   - Capture screenshot for submission

---

##  Observations & Insights

### Key Findings

1. **Best Performing Model: XGBoost**
   - Achieved highest F1 Score of **0.6233**
   - Best balance between Precision (0.6829) and Recall (0.5733)
   - Highest AUC Score (0.9547) indicating excellent discrimination ability
   - Superior performance due to gradient boosting and handling of imbalanced data

2. **Model Comparison Analysis:**
   - **Ensemble Models Outperform:** Both XGBoost and Random Forest achieved top performance
   - **Decision Tree:** Good performance (F1: 0.6163) but prone to overfitting without ensemble
   - **Logistic Regression:** Solid baseline (F1: 0.5177) with high precision (0.7004)
   - **Naive Bayes:** Highest recall (0.6175) but lower precision, suitable when false negatives are costly
   - **KNN:** Moderate performance (F1: 0.4660), sensitive to feature scaling

3. **Dataset Characteristics:**
   - **Class Imbalance:** Dataset is imbalanced (majority class: "no")
   - **Feature Impact:** Duration and campaign-related features are strong predictors
   - **Categorical Encoding:** Label encoding worked well for tree-based models
   - **Scaling Importance:** Critical for distance-based models (KNN, Logistic Regression)

4. **Evaluation Insights:**
   - **F1 Score:** Most important metric for imbalanced classification
   - **MCC:** XGBoost achieved highest MCC (0.5829), confirming balanced performance
   - **AUC Score:** All models achieved >0.85 AUC, indicating good discrimination
   - **Precision vs Recall Trade-off:** Random Forest has highest precision (0.7050) but lower recall

### Recommendations

1. **For Production Use:**
   - Use Random Forest or XGBoost for best performance
   - Consider ensemble of top models
   - Implement threshold tuning for business requirements

2. **Future Improvements:**
   - Hyperparameter tuning (GridSearchCV, RandomizedSearchCV)
   - Feature engineering (interaction terms, polynomial features)
   - Handle class imbalance (SMOTE, class weights)
   - Cross-validation for robust evaluation

---

##  Screenshots

*(Screenshots will be added after deployment)*

### 1. Streamlit Application Home Page
![Home Page](screenshots/home_page.png)

### 2. Model Prediction Interface
![Prediction Page](screenshots/prediction_page.png)

### 3. Evaluation Metrics Display
![Metrics](screenshots/metrics_display.png)

### 4. Confusion Matrix
![Confusion Matrix](screenshots/confusion_matrix.png)

### 5. Model Comparison
![Model Comparison](screenshots/model_comparison.png)

### 6. BITS Virtual Lab Execution
![Virtual Lab](screenshots/virtual_lab_execution.png)

---

##  References

1. **Dataset:**
   - [UCI Machine Learning Repository - Bank Marketing](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
   - Moro et al., 2014. "A Data-Driven Approach to Predict the Success of Bank Telemarketing"

2. **Libraries:**
   - [Scikit-learn Documentation](https://scikit-learn.org/)
   - [XGBoost Documentation](https://xgboost.readthedocs.io/)
   - [Streamlit Documentation](https://docs.streamlit.io/)

3. **Metrics:**
   - [Classification Metrics - Scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)
   - Matthews Correlation Coefficient: [Wikipedia](https://en.wikipedia.org/wiki/Matthews_correlation_coefficient)

---

##  Author

**Student ID:** 2025aa05072  
**Program:** M.Tech (AIML)  
**Institution:** BITS Pilani

---

##  License

This project is created for academic purposes as part of ML Assignment 2.

---

##  Acknowledgments

- BITS Pilani for providing the assignment framework
- UCI Machine Learning Repository for the dataset
- Open-source community for excellent ML libraries

---

**Last Updated:** January 2026
