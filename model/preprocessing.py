"""
Data preprocessing utilities for Bank Marketing dataset
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os


class DataPreprocessor:
    """Handle data loading, cleaning, and preprocessing"""
    
    def __init__(self, data_path='../data/bank-additional-full.csv'):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
    def load_data(self):
        """Load the Bank Marketing dataset"""
        print(f"Loading data from {self.data_path}...")
        df = pd.read_csv(self.data_path, sep=';')
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        return df
    
    def preprocess(self, df, fit=True):
        """
        Preprocess the dataset
        
        Args:
            df: Input dataframe
            fit: If True, fit encoders and scaler. If False, use existing ones.
        
        Returns:
            X: Feature matrix
            y: Target vector
        """
        # Make a copy
        data = df.copy()
        
        # Separate features and target
        X = data.drop('y', axis=1)
        y = data['y'].map({'yes': 1, 'no': 0})
        
        # Identify categorical and numerical columns
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        print(f"\nCategorical columns ({len(categorical_cols)}): {categorical_cols}")
        print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols}")
        
        # Encode categorical variables
        if fit:
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
        else:
            for col in categorical_cols:
                if col in self.label_encoders:
                    X[col] = self.label_encoders[col].transform(X[col].astype(str))
        
        # Scale numerical features
        if fit:
            X[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
        else:
            X[numerical_cols] = self.scaler.transform(X[numerical_cols])
        
        # Store feature names
        if fit:
            self.feature_names = X.columns.tolist()
        
        return X.values, y.values
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"\nTrain set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        print(f"Train class distribution: {np.bincount(y_train)}")
        print(f"Test class distribution: {np.bincount(y_test)}")
        return X_train, X_test, y_train, y_test
    
    def save_preprocessor(self, save_dir='model/saved_models'):
        """Save scaler and encoders"""
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(self.scaler, os.path.join(save_dir, 'scaler.pkl'))
        joblib.dump(self.label_encoders, os.path.join(save_dir, 'label_encoders.pkl'))
        joblib.dump(self.feature_names, os.path.join(save_dir, 'feature_names.pkl'))
        print(f"\nPreprocessor saved to {save_dir}")
    
    def load_preprocessor(self, save_dir='model/saved_models'):
        """Load scaler and encoders"""
        self.scaler = joblib.load(os.path.join(save_dir, 'scaler.pkl'))
        self.label_encoders = joblib.load(os.path.join(save_dir, 'label_encoders.pkl'))
        self.feature_names = joblib.load(os.path.join(save_dir, 'feature_names.pkl'))
        print(f"\nPreprocessor loaded from {save_dir}")


def create_sample_test_data(output_path='../data/sample_test.csv', n_samples=100):
    """Create a sample test CSV for demonstration"""
    # Load original data
    df = pd.read_csv('../data/bank-additional-full.csv', sep=';')
    
    # Sample random rows
    sample = df.sample(n=n_samples, random_state=42)
    
    # Save as CSV
    sample.to_csv(output_path, index=False)
    print(f"Sample test data saved to {output_path}")


if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data()
    
    print("\nDataset Info:")
    print(df.info())
    print("\nFirst few rows:")
    print(df.head())
    print("\nTarget distribution:")
    print(df['y'].value_counts())
    
    # Preprocess
    X, y = preprocessor.preprocess(df, fit=True)
    print(f"\nPreprocessed data shape: {X.shape}")
    
    # Split
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    
    # Save preprocessor
    preprocessor.save_preprocessor()
    
    # Create sample test data
    create_sample_test_data()
    
    print("\n✅ Preprocessing complete!")
