"""
Model training script for the flotation process dataset.
Trains the final model using the best parameters found by GridSearch.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR


def train_final_model():
    """
    Train the final model using the best parameters from GridSearch.
    """
    
    # Define paths
    input_dir = Path("data/processed_data")
    models_dir = Path("models")
    
    # Load the best model parameters
    print("Loading best model parameters...")
    with open(models_dir / "best_model_params.pkl", "rb") as f:
        best_config = pickle.load(f)
    
    model_name = best_config['model_name']
    best_params = best_config['params']
    
    print(f"Training final model: {model_name}")
    print(f"Parameters: {best_params}")
    
    # Load training data
    print("Loading training data...")
    X_train_scaled = pd.read_csv(input_dir / "X_train_scaled.csv")
    y_train = pd.read_csv(input_dir / "y_train.csv")["silica_concentrate"]
    
    print(f"Training data shape: {X_train_scaled.shape}")
    print(f"Target data shape: {y_train.shape}")
    
    # Initialize the model with best parameters
    model_classes = {
        'RandomForest': RandomForestRegressor,
        'Ridge': Ridge,
        'ElasticNet': ElasticNet,
        'SVR': SVR
    }
    
    if model_name not in model_classes:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Create model instance with best parameters
    if model_name in ['RandomForest', 'Ridge', 'ElasticNet']:
        final_model = model_classes[model_name](random_state=42, **best_params)
    else:
        final_model = model_classes[model_name](**best_params)
    
    # Train the final model on all training data
    print("Training final model on full training set...")
    final_model.fit(X_train_scaled, y_train)
    
    # Evaluate on training data
    y_train_pred = final_model.predict(X_train_scaled)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_r2 = r2_score(y_train, y_train_pred)
    
    print(f"Training performance:")
    print(f"  MSE: {train_mse:.4f}")
    print(f"  RMSE: {train_rmse:.4f}")
    print(f"  R² Score: {train_r2:.4f}")
    
    # Save the trained model
    print("Saving trained model...")
    with open(models_dir / "trained_model.pkl", "wb") as f:
        pickle.dump(final_model, f)
    
    # Save training metrics
    training_metrics = {
        'model_name': model_name,
        'best_parameters': best_params,
        'training_metrics': {
            'mse': float(train_mse),
            'rmse': float(train_rmse),
            'r2_score': float(train_r2)
        },
        'training_samples': len(X_train_scaled),
        'features': list(X_train_scaled.columns)
    }
    
    with open(models_dir / "training_info.json", "w") as f:
        json.dump(training_metrics, f, indent=2)
    
    print(f"Model training completed successfully!")
    print(f"Final model saved as: trained_model.pkl")
    
    return final_model, training_metrics

if __name__ == "__main__":
    train_final_model()