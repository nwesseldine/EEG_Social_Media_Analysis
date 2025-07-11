import os
import pandas as pd
import joblib
from matplotlib import pyplot as plt
import glob

# Load the most recent trained models
def load_latest_models():
    # Find the most recent model files
    reg_models = glob.glob('models/concentration_rf_reg_model_*.pkl')
    clf_models = glob.glob('models/concentration_rf_clf_model_*.pkl')
    
    if not reg_models or not clf_models:
        raise FileNotFoundError("No trained models found in the models directory")
    
    # Get the most recent models (sorted by filename which includes timestamp)
    latest_reg_model = sorted(reg_models)[-1]
    latest_clf_model = sorted(clf_models)[-1]
    
    print(f"Loading regression model: {latest_reg_model}")
    print(f"Loading classification model: {latest_clf_model}")
    
    reg_loaded = joblib.load(latest_reg_model)
    clf_loaded = joblib.load(latest_clf_model)
    
    return reg_loaded, clf_loaded

# Load real data for prediction
def load_real_data():
    real_data = pd.read_csv("featuresets/local datasets_2025-07-10_16-45.csv")
    # Remove the label column for prediction
    real_data_features = real_data.drop(axis=1, labels=["Label"])
    return real_data_features

# Make predictions and visualize
def predict_and_visualize():
    # Load the trained models
    reg_model, clf_model = load_latest_models()
    
    # Load real data
    real_data = load_real_data()
    
    # Make predictions
    real_clf_predictions = clf_model.predict(real_data)
    real_reg_predictions = reg_model.predict(real_data)
    
    print(f"Classification predictions shape: {real_clf_predictions.shape}")
    print(f"Regression predictions shape: {real_reg_predictions.shape}")
    
    # Visualize regression predictions
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(real_reg_predictions)
    plt.title("Regression Predictions on Real Data")
    plt.xlabel("Sample Index")
    plt.ylabel("Predicted Label")
    
    plt.subplot(1, 2, 2)
    plt.plot(real_clf_predictions)
    plt.title("Classification Predictions on Real Data")
    plt.xlabel("Sample Index")
    plt.ylabel("Predicted Class")
    
    plt.tight_layout()
    plt.show()
    
    return real_clf_predictions, real_reg_predictions

if __name__ == "__main__":
    clf_predictions, reg_predictions = predict_and_visualize()
