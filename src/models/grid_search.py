"""
GridSearch script for hyperparameter optimization.
Finds the best parameters for regression models on the flotation process dataset.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR


def grid_search_best_params():
    """
    Perform grid search to find the best hyperparameters for different regression models.
    """
    
    # Define paths
    input_dir = Path("data/processed_data")
    models_dir = Path("models")
    
    # Create models directory if it doesn't exist
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the normalized training data
    print("Loading normalized training data...")
    X_train_scaled = pd.read_csv(input_dir / "X_train_scaled.csv")
    y_train = pd.read_csv(input_dir / "y_train.csv")["silica_concentrate"]
    
    print(f"Training data shape: {X_train_scaled.shape}")
    print(f"Target data shape: {y_train.shape}")
    
    # Define models and their parameter grids
    models_params = {
        'RandomForest': {
            'model': RandomForestRegressor(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        'Ridge': {
            'model': Ridge(random_state=42),
            'params': {
                'alpha': [0.1, 1.0, 10.0, 100.0],
                'solver': ['auto', 'svd', 'cholesky']
            }
        },
        'ElasticNet': {
            'model': ElasticNet(random_state=42, max_iter=2000),
            'params': {
                'alpha': [0.1, 1.0, 10.0],
                'l1_ratio': [0.1, 0.5, 0.7, 0.9]
            }
        },
        'SVR': {
            'model': SVR(),
            'params': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto'],
                'kernel': ['rbf', 'linear']
            }
        }
    }
    
    best_models = {}
    results = {}
    
    print("\nStarting grid search for each model...")
    
    for model_name, config in models_params.items():
        print(f"\n--- Grid Search for {model_name} ---")
        
        # Perform grid search with cross-validation
        grid_search = GridSearchCV(
            estimator=config['model'],
            param_grid=config['params'],
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit the grid search
        grid_search.fit(X_train_scaled, y_train)
        
        # Store results
        best_models[model_name] = grid_search.best_estimator_
        results[model_name] = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': {
                'mean_test_score': float(grid_search.best_score_),
                'std_test_score': float(grid_search.cv_results_['std_test_score'][grid_search.best_index_])
            }
        }
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score (neg MSE): {grid_search.best_score_:.4f}")
    
    # Find the overall best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['best_score'])
    best_model = best_models[best_model_name]
    
    print(f"\n=== BEST MODEL: {best_model_name} ===")
    print(f"Best parameters: {results[best_model_name]['best_params']}")
    print(f"Best CV score: {results[best_model_name]['best_score']:.4f}")
    
    # Save the best model parameters
    print("\nSaving best model parameters...")
    with open(models_dir / "best_model_params.pkl", "wb") as f:
        pickle.dump({
            'model_name': best_model_name,
            'model': best_model,
            'params': results[best_model_name]['best_params']
        }, f)
    
    # Save detailed results
    with open(models_dir / "grid_search_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Grid search completed. Best model saved: {best_model_name}")
    
    return best_model, results

if __name__ == "__main__":
    grid_search_best_params()