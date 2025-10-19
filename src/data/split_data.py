"""
Data splitting script for the flotation process dataset.
Splits data into training and testing sets with silica_concentrate as target variable.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import os

def split_data(test_size=0.2, random_state=42):
    """
    Split the raw data into training and testing sets.
    
    Args:
        test_size (float): Proportion of dataset for testing
        random_state (int): Random state for reproducibility
    """
    
    # Define paths
    data_path = Path("data/raw_data/raw.csv")
    output_dir = Path("data/processed_data")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    # Drop date column as it's not needed for modeling
    df = df.drop('date', axis=1)
    
    # Separate features and target
    X = df.drop('silica_concentrate', axis=1)
    y = df['silica_concentrate']
    
    # Split the data
    print(f"Splitting data with test_size={test_size}...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        shuffle=True
    )
    
    # Save the splits
    print("Saving split datasets...")
    X_train.to_csv(output_dir / "X_train.csv", index=False)
    X_test.to_csv(output_dir / "X_test.csv", index=False)
    y_train.to_csv(output_dir / "y_train.csv", index=False)
    y_test.to_csv(output_dir / "y_test.csv", index=False)
    
    # Print summary
    print(f"Data split completed:")
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Testing set: {len(X_test)} samples")
    print(f"  Features: {len(X.columns)}")
    print(f"  Target variable: silica_concentrate")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    split_data()