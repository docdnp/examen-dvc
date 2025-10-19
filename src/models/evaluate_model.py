"""
Model evaluation script for the flotation process dataset.
Evaluates the trained model and generates predictions with performance metrics.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model():
    """
    Evaluate the trained model on test data and generate predictions.
    """
    
    # Define paths
    input_dir = Path("data/processed_data")
    models_dir = Path("models")
    metrics_dir = Path("metrics")
    data_dir = Path("data")
    
    # Create directories if they don't exist
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the trained model
    print("Loading trained model...")
    with open(models_dir / "trained_model.pkl", "rb") as f:
        trained_model = pickle.load(f)
    
    # Load test data
    print("Loading test data...")
    X_test_scaled = pd.read_csv(input_dir / "X_test_scaled.csv")
    y_test = pd.read_csv(input_dir / "y_test.csv")["silica_concentrate"]
    
    print(f"Test data shape: {X_test_scaled.shape}")
    print(f"Test target shape: {y_test.shape}")
    
    # Make predictions
    print("Making predictions on test data...")
    y_pred = trained_model.predict(X_test_scaled)
    
    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate additional metrics
    y_mean = y_test.mean()
    y_std = y_test.std()
    rmse_normalized = rmse / y_std
    mae_normalized = mae / y_std
    
    # Print evaluation results
    print(f"\n=== MODEL EVALUATION RESULTS ===")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE as % of target std: {rmse_normalized*100:.2f}%")
    print(f"MAE as % of target std: {mae_normalized*100:.2f}%")
    
    # Create predictions dataset
    predictions_df = pd.DataFrame({
        'y_true': y_test.values,
        'y_pred': y_pred,
        'residuals': y_test.values - y_pred,
        'abs_residuals': np.abs(y_test.values - y_pred)
    })
    
    # Add prediction intervals (simple approach using residual std)
    residual_std = np.std(predictions_df['residuals'])
    predictions_df['pred_lower_95'] = y_pred - 1.96 * residual_std
    predictions_df['pred_upper_95'] = y_pred + 1.96 * residual_std
    
    # Save predictions
    print("Saving predictions...")
    predictions_df.to_csv(data_dir / "predictions.csv", index=False)
    
    # Prepare scores for JSON export
    scores = {
        "model_performance": {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2_score": float(r2)
        },
        "normalized_metrics": {
            "rmse_normalized": float(rmse_normalized),
            "mae_normalized": float(mae_normalized)
        },
        "target_statistics": {
            "mean": float(y_mean),
            "std": float(y_std),
            "min": float(y_test.min()),
            "max": float(y_test.max())
        },
        "prediction_statistics": {
            "pred_mean": float(np.mean(y_pred)),
            "pred_std": float(np.std(y_pred)),
            "pred_min": float(np.min(y_pred)),
            "pred_max": float(np.max(y_pred))
        },
        "residual_analysis": {
            "residual_mean": float(np.mean(predictions_df['residuals'])),
            "residual_std": float(residual_std),
            "max_absolute_error": float(np.max(predictions_df['abs_residuals']))
        },
        "test_samples": len(y_test)
    }
    
    # Save scores
    print("Saving evaluation metrics...")
    with open(metrics_dir / "scores.json", "w") as f:
        json.dump(scores, f, indent=2)
    
    # Performance summary
    print(f"\n=== SUMMARY ===")
    print(f"Model explains {r2*100:.2f}% of the variance in silica concentration")
    print(f"Average prediction error: ±{mae:.4f} units")
    print(f"95% of predictions are within ±{1.96*residual_std:.4f} units")
    
    print(f"\nEvaluation completed successfully!")
    print(f"Predictions saved to: data/predictions.csv")
    print(f"Metrics saved to: metrics/scores.json")
    
    return scores, predictions_df

if __name__ == "__main__":
    evaluate_model()