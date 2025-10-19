"""
Data normalization script for the flotation process dataset.
Normalizes training and testing features using StandardScaler.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def normalize_data():
    """
    Normalize the training and testing feature sets.
    Fits scaler on training data and applies to both train and test sets.
    """
    
    # Define paths
    input_dir = Path("data/processed_data")
    output_dir = Path("data/processed_data")
    models_dir = Path("models")
    
    # Create directories if they don't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the split data
    print("Loading split datasets...")
    X_train = pd.read_csv(input_dir / "X_train.csv")
    X_test = pd.read_csv(input_dir / "X_test.csv")
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    # Initialize and fit the scaler on training data
    print("Fitting scaler on training data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Apply the same scaler to test data
    print("Applying scaler to test data...")
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrames with original column names
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # Save the normalized datasets
    print("Saving normalized datasets...")
    X_train_scaled_df.to_csv(output_dir / "X_train_scaled.csv", index=False)
    X_test_scaled_df.to_csv(output_dir / "X_test_scaled.csv", index=False)
    
    # Save the scaler for future use
    print("Saving scaler...")
    with open(models_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    # Print summary statistics
    print("\nNormalization completed:")
    print(f"  Training set mean: {X_train_scaled_df.mean().mean():.6f}")
    print(f"  Training set std: {X_train_scaled_df.std().mean():.6f}")
    print(f"  Testing set mean: {X_test_scaled_df.mean().mean():.6f}")
    print(f"  Testing set std: {X_test_scaled_df.std().mean():.6f}")
    
    return X_train_scaled_df, X_test_scaled_df, scaler

if __name__ == "__main__":
    normalize_data()