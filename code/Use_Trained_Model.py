import os
import pandas as pd
import joblib
from matplotlib import pyplot as plt
import glob

# Load all trained models
def load_all_models():
    # Find all model files
    reg_models = glob.glob('models/concentration_rf_reg_model_*.pkl')
    clf_models = glob.glob('models/concentration_rf_clf_model_*.pkl')
    
    if not reg_models or not clf_models:
        raise FileNotFoundError("No trained models found in the models directory")
    
    # Sort models by filename (which includes timestamp)
    reg_models = sorted(reg_models)
    clf_models = sorted(clf_models)
    
    print(f"Found {len(reg_models)} regression models:")
    for model in reg_models:
        print(f"  - {model}")
    
    print(f"Found {len(clf_models)} classification models:")
    for model in clf_models:
        print(f"  - {model}")
    
    # Load all regression models
    reg_loaded_models = []
    for model_path in reg_models:
        print(f"Loading regression model: {model_path}")
        reg_loaded_models.append(joblib.load(model_path))
    
    # Load all classification models
    clf_loaded_models = []
    for model_path in clf_models:
        print(f"Loading classification model: {model_path}")
        clf_loaded_models.append(joblib.load(model_path))
    
    return reg_loaded_models, clf_loaded_models, reg_models, clf_models

# Load real data for prediction
def load_real_data():
    real_data = pd.read_csv("featuresets/local datasets_2025-07-10_16-45.csv")
    # Remove the label column for prediction
    real_data_features = real_data.drop(axis=1, labels=["Label"])
    return real_data_features

# Make predictions and visualize
def predict_and_visualize():
    # Load all trained models
    reg_models, clf_models, reg_model_names, clf_model_names = load_all_models()
    
    # Load real data
    real_data = load_real_data()
    
    # Make predictions with all models
    all_reg_predictions = []
    all_clf_predictions = []
    
    print("\nMaking predictions with regression models:")
    for i, (model, model_name) in enumerate(zip(reg_models, reg_model_names)):
        predictions = model.predict(real_data)
        all_reg_predictions.append(predictions)
        print(f"  Model {i+1} ({model_name}): predictions shape {predictions.shape}")
    
    print("\nMaking predictions with classification models:")
    for i, (model, model_name) in enumerate(zip(clf_models, clf_model_names)):
        predictions = model.predict(real_data)
        all_clf_predictions.append(predictions)
        print(f"  Model {i+1} ({model_name}): predictions shape {predictions.shape}")
    
    # Visualize predictions from all models
    num_reg_models = len(all_reg_predictions)
    num_clf_models = len(all_clf_predictions)
    
    # Create subplots for regression models
    if num_reg_models > 0:
        plt.figure(figsize=(15, 5 * num_reg_models))
        for i, (predictions, model_name) in enumerate(zip(all_reg_predictions, reg_model_names)):
            plt.subplot(num_reg_models, 1, i + 1)
            plt.plot(predictions)
            plt.title(f"Regression Predictions - {os.path.basename(model_name)}")
            plt.xlabel("Sample Index")
            plt.ylabel("Predicted Label")
        plt.tight_layout()
        plt.show()
    
    # Create subplots for classification models
    if num_clf_models > 0:
        plt.figure(figsize=(15, 5 * num_clf_models))
        for i, (predictions, model_name) in enumerate(zip(all_clf_predictions, clf_model_names)):
            plt.subplot(num_clf_models, 1, i + 1)
            plt.plot(predictions)
            plt.title(f"Classification Predictions - {os.path.basename(model_name)}")
            plt.xlabel("Sample Index")
            plt.ylabel("Predicted Class")
        plt.tight_layout()
        plt.show()
    
    return all_clf_predictions, all_reg_predictions

if __name__ == "__main__":
    clf_predictions, reg_predictions = predict_and_visualize()
