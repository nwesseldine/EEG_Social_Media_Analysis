import os
import pandas as pd
import joblib
from matplotlib import pyplot as plt
import numpy as np
import glob
from datetime import datetime

# Load all trained models
def load_all_models():
    # Find all .pkl model files in the models directory
    all_model_files = glob.glob('models/*.pkl')
    
    if not all_model_files:
        raise FileNotFoundError("No .pkl model files found in the models directory")
    
    # Sort models by filename (which may include timestamp)
    all_model_files = sorted(all_model_files)
    
    # Separate regression and classification models based on filename patterns
    reg_models = []
    clf_models = []
    other_models = []
    
    for model_path in all_model_files:
        filename = os.path.basename(model_path).lower()
        if 'reg' in filename or 'regression' in filename:
            reg_models.append(model_path)
        elif 'clf' in filename or 'classification' in filename or 'classifier' in filename:
            clf_models.append(model_path)
        else:
            # If we can't determine the type from filename, treat as general model
            other_models.append(model_path)
    
    print(f"Found {len(all_model_files)} total model files:")
    print(f"  - {len(reg_models)} regression models")
    print(f"  - {len(clf_models)} classification models")
    print(f"  - {len(other_models)} other/unknown type models")
    
    print("\nRegression models:")
    for model in reg_models:
        print(f"  - {model}")
    
    print("\nClassification models:")
    for model in clf_models:
        print(f"  - {model}")
    
    if other_models:
        print("\nOther models (will be treated as general models):")
        for model in other_models:
            print(f"  - {model}")
    
    # Load all regression models
    reg_loaded_models = []
    for model_path in reg_models:
        try:
            print(f"Loading regression model: {model_path}")
            reg_loaded_models.append(joblib.load(model_path))
        except Exception as e:
            print(f"Error loading {model_path}: {e}")
    
    # Load all classification models
    clf_loaded_models = []
    for model_path in clf_models:
        try:
            print(f"Loading classification model: {model_path}")
            clf_loaded_models.append(joblib.load(model_path))
        except Exception as e:
            print(f"Error loading {model_path}: {e}")
    
    # Load other models
    other_loaded_models = []
    for model_path in other_models:
        try:
            print(f"Loading general model: {model_path}")
            other_loaded_models.append(joblib.load(model_path))
        except Exception as e:
            print(f"Error loading {model_path}: {e}")
    
    return reg_loaded_models, clf_loaded_models, other_loaded_models, reg_models, clf_models, other_models

# Load real data for prediction
def load_real_data():
    real_data = pd.read_csv("featuresets/local datasets_2025-07-10_16-45.csv")
    # Remove the label column for prediction
    real_data_features = real_data.drop(axis=1, labels=["Label"])
    return real_data_features

# Make predictions and visualize
def predict_and_visualize():
    # Load all trained models
    reg_models, clf_models, other_models, reg_model_names, clf_model_names, other_model_names = load_all_models()
    
    # Load real data
    real_data = load_real_data()
    
    # Make predictions with all models
    all_reg_predictions = []
    all_clf_predictions = []
    all_other_predictions = []
    
    # Regression models
    if reg_models:
        print("\nMaking predictions with regression models:")
        for i, (model, model_name) in enumerate(zip(reg_models, reg_model_names)):
            try:
                predictions = model.predict(real_data)
                all_reg_predictions.append(predictions)
                print(f"  Model {i+1} ({model_name}): predictions shape {predictions.shape}")
            except Exception as e:
                print(f"  Error predicting with {model_name}: {e}")
    
    # Classification models
    if clf_models:
        print("\nMaking predictions with classification models:")
        for i, (model, model_name) in enumerate(zip(clf_models, clf_model_names)):
            try:
                predictions = model.predict(real_data)
                all_clf_predictions.append(predictions)
                print(f"  Model {i+1} ({model_name}): predictions shape {predictions.shape}")
            except Exception as e:
                print(f"  Error predicting with {model_name}: {e}")
    
    # Other/general models
    if other_models:
        print("\nMaking predictions with other/general models:")
        for i, (model, model_name) in enumerate(zip(other_models, other_model_names)):
            try:
                predictions = model.predict(real_data)
                all_other_predictions.append(predictions)
                print(f"  Model {i+1} ({model_name}): predictions shape {predictions.shape}")
            except Exception as e:
                print(f"  Error predicting with {model_name}: {e}")
    
    # Create simple, effective visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    create_simple_visualizations(all_reg_predictions, all_clf_predictions, all_other_predictions,
                                reg_model_names, clf_model_names, other_model_names)
    
    # Ask if user wants to save
    save_option = input("\n💾 Would you like to save these visualizations? (y/n): ").lower().strip()
    if save_option in ['y', 'yes']:
        save_simple_visualizations(all_reg_predictions, all_clf_predictions, all_other_predictions,
                                  reg_model_names, clf_model_names, other_model_names)
    
    return all_clf_predictions, all_reg_predictions, all_other_predictions

def create_simple_visualizations(all_reg_predictions, all_clf_predictions, all_other_predictions, 
                               reg_model_names, clf_model_names, other_model_names):
    """Create simple, effective visualizations for model predictions"""
    
    # Set a clean style
    plt.style.use('default')
    
    # 1. Overview plot - all models in one view
    create_overview_plot(all_reg_predictions, all_clf_predictions, all_other_predictions,
                        reg_model_names, clf_model_names, other_model_names)
    
    # 2. Individual model plots (only if there are multiple models)
    if len(all_reg_predictions) > 1:
        create_comparison_plot(all_reg_predictions, reg_model_names, "Regression Models")
    
    if len(all_clf_predictions) > 1:
        create_comparison_plot(all_clf_predictions, clf_model_names, "Classification Models")

def create_overview_plot(all_reg_predictions, all_clf_predictions, all_other_predictions,
                        reg_model_names, clf_model_names, other_model_names):
    """Create a single overview plot with all essential information"""
    
    total_models = len(all_reg_predictions) + len(all_clf_predictions) + len(all_other_predictions)
    if total_models == 0:
        print("No models to visualize")
        return
    
    # Create subplots based on what we have
    fig_height = 6
    if all_reg_predictions and all_clf_predictions:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    elif all_reg_predictions or all_clf_predictions or all_other_predictions:
        fig, ax1 = plt.subplots(1, 1, figsize=(15, 6))
        ax2 = None
    else:
        return
    
    # Plot regression models
    if all_reg_predictions:
        sample_indices = np.arange(len(all_reg_predictions[0]))
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_reg_predictions)))
        
        for i, (predictions, model_name, color) in enumerate(zip(all_reg_predictions, reg_model_names, colors)):
            label = f"Reg Model {i+1}: {os.path.basename(model_name).replace('.pkl', '')}"
            ax1.plot(sample_indices, predictions, label=label, color=color, linewidth=2, alpha=0.8)
        
        ax1.set_title('Regression Model Predictions', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Sample Index')
        ax1.set_ylabel('Predicted Value')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Add summary statistics
        if len(all_reg_predictions) > 1:
            mean_pred = np.mean([np.mean(pred) for pred in all_reg_predictions])
            ax1.text(0.02, 0.98, f'Avg Mean: {mean_pred:.3f}', transform=ax1.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
                    verticalalignment='top')
    
    # Plot classification models
    if all_clf_predictions and ax2 is not None:
        sample_indices = np.arange(len(all_clf_predictions[0]))
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_clf_predictions)))
        
        for i, (predictions, model_name, color) in enumerate(zip(all_clf_predictions, clf_model_names, colors)):
            label = f"Clf Model {i+1}: {os.path.basename(model_name).replace('.pkl', '')}"
            ax2.plot(sample_indices, predictions, label=label, color=color, 
                    linewidth=2, marker='o', markersize=3, alpha=0.8)
        
        ax2.set_title('Classification Model Predictions', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Sample Index')
        ax2.set_ylabel('Predicted Class')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Add class distribution info
        if len(all_clf_predictions) > 0:
            unique_classes = len(np.unique(all_clf_predictions[0]))
            ax2.text(0.02, 0.98, f'Classes: {unique_classes}', transform=ax2.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8),
                    verticalalignment='top')
    elif all_clf_predictions and ax2 is None:
        # If we only have classification models, use ax1
        sample_indices = np.arange(len(all_clf_predictions[0]))
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_clf_predictions)))
        
        for i, (predictions, model_name, color) in enumerate(zip(all_clf_predictions, clf_model_names, colors)):
            label = f"Clf Model {i+1}: {os.path.basename(model_name).replace('.pkl', '')}"
            ax1.plot(sample_indices, predictions, label=label, color=color, 
                    linewidth=2, marker='o', markersize=3, alpha=0.8)
        
        ax1.set_title('Classification Model Predictions', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Sample Index')
        ax1.set_ylabel('Predicted Class')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
    
    plt.suptitle(f'EEG Model Predictions Overview ({total_models} models)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def create_comparison_plot(predictions_list, model_names, title):
    """Create a simple comparison plot for multiple models of the same type"""
    
    if len(predictions_list) < 2:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Time series comparison
    sample_indices = np.arange(len(predictions_list[0]))
    colors = plt.cm.tab10(np.linspace(0, 1, len(predictions_list)))
    
    for i, (predictions, model_name, color) in enumerate(zip(predictions_list, model_names, colors)):
        short_name = os.path.basename(model_name).replace('.pkl', '')
        ax1.plot(sample_indices, predictions, label=f'Model {i+1}', 
                color=color, linewidth=2, alpha=0.8)
    
    ax1.set_title(f'{title} - Time Series', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Predicted Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot for distribution comparison
    ax2.boxplot(predictions_list, tick_labels=[f'Model {i+1}' for i in range(len(predictions_list))])
    ax2.set_title(f'{title} - Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Predicted Value')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'{title} Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def save_simple_visualizations(all_reg_predictions, all_clf_predictions, all_other_predictions,
                              reg_model_names, clf_model_names, other_model_names):
    """Save simple visualizations to files"""
    
    # Create visualizations directory if it doesn't exist
    viz_dir = "visualizations"
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    saved_files = []
    
    print(f"Saving visualizations to {viz_dir}/...")
    
    # Save overview plot
    fig = create_overview_plot_for_saving(all_reg_predictions, all_clf_predictions, all_other_predictions,
                                         reg_model_names, clf_model_names, other_model_names)
    if fig is not None:
        filename = f"{viz_dir}/model_overview_{timestamp}.png"
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        saved_files.append(filename)
    
    # Save comparison plots
    if len(all_reg_predictions) > 1:
        fig = create_comparison_plot_for_saving(all_reg_predictions, reg_model_names, "Regression Models")
        if fig is not None:
            filename = f"{viz_dir}/regression_comparison_{timestamp}.png"
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            saved_files.append(filename)
    
    if len(all_clf_predictions) > 1:
        fig = create_comparison_plot_for_saving(all_clf_predictions, clf_model_names, "Classification Models")
        if fig is not None:
            filename = f"{viz_dir}/classification_comparison_{timestamp}.png"
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            saved_files.append(filename)
    
    print(f"\n✅ Successfully saved {len(saved_files)} visualization(s):")
    for file in saved_files:
        print(f"   - {file}")
    print()

# Legacy function for backwards compatibility (removed complex visualizations)
def create_comprehensive_visualizations(*args, **kwargs):
    """Redirects to simple visualizations"""
    return create_simple_visualizations(*args, **kwargs)

def create_overview_plot_for_saving(all_reg_predictions, all_clf_predictions, all_other_predictions,
                                   reg_model_names, clf_model_names, other_model_names):
    """Create overview plot for saving (returns figure instead of showing)"""
    
    total_models = len(all_reg_predictions) + len(all_clf_predictions) + len(all_other_predictions)
    if total_models == 0:
        return None
    
    # Create subplots based on what we have
    if all_reg_predictions and all_clf_predictions:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    elif all_reg_predictions or all_clf_predictions or all_other_predictions:
        fig, ax1 = plt.subplots(1, 1, figsize=(15, 6))
        ax2 = None
    else:
        return None
    
    # Plot regression models
    if all_reg_predictions:
        sample_indices = np.arange(len(all_reg_predictions[0]))
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_reg_predictions)))
        
        for i, (predictions, model_name, color) in enumerate(zip(all_reg_predictions, reg_model_names, colors)):
            label = f"Reg Model {i+1}: {os.path.basename(model_name).replace('.pkl', '')}"
            ax1.plot(sample_indices, predictions, label=label, color=color, linewidth=2, alpha=0.8)
        
        ax1.set_title('Regression Model Predictions', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Sample Index')
        ax1.set_ylabel('Predicted Value')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Add summary statistics
        if len(all_reg_predictions) > 1:
            mean_pred = np.mean([np.mean(pred) for pred in all_reg_predictions])
            ax1.text(0.02, 0.98, f'Avg Mean: {mean_pred:.3f}', transform=ax1.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
                    verticalalignment='top')
    
    # Plot classification models
    if all_clf_predictions and ax2 is not None:
        sample_indices = np.arange(len(all_clf_predictions[0]))
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_clf_predictions)))
        
        for i, (predictions, model_name, color) in enumerate(zip(all_clf_predictions, clf_model_names, colors)):
            label = f"Clf Model {i+1}: {os.path.basename(model_name).replace('.pkl', '')}"
            ax2.plot(sample_indices, predictions, label=label, color=color, 
                    linewidth=2, marker='o', markersize=3, alpha=0.8)
        
        ax2.set_title('Classification Model Predictions', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Sample Index')
        ax2.set_ylabel('Predicted Class')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Add class distribution info
        if len(all_clf_predictions) > 0:
            unique_classes = len(np.unique(all_clf_predictions[0]))
            ax2.text(0.02, 0.98, f'Classes: {unique_classes}', transform=ax2.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8),
                    verticalalignment='top')
    elif all_clf_predictions and ax2 is None:
        # If we only have classification models, use ax1
        sample_indices = np.arange(len(all_clf_predictions[0]))
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_clf_predictions)))
        
        for i, (predictions, model_name, color) in enumerate(zip(all_clf_predictions, clf_model_names, colors)):
            label = f"Clf Model {i+1}: {os.path.basename(model_name).replace('.pkl', '')}"
            ax1.plot(sample_indices, predictions, label=label, color=color, 
                    linewidth=2, marker='o', markersize=3, alpha=0.8)
        
        ax1.set_title('Classification Model Predictions', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Sample Index')
        ax1.set_ylabel('Predicted Class')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
    
    fig.suptitle(f'EEG Model Predictions Overview ({total_models} models)', fontsize=16, fontweight='bold')
    fig.tight_layout()
    return fig

def create_comparison_plot_for_saving(predictions_list, model_names, title):
    """Create comparison plot for saving (returns figure instead of showing)"""
    
    if len(predictions_list) < 2:
        return None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Time series comparison
    sample_indices = np.arange(len(predictions_list[0]))
    colors = plt.cm.tab10(np.linspace(0, 1, len(predictions_list)))
    
    for i, (predictions, model_name, color) in enumerate(zip(predictions_list, model_names, colors)):
        short_name = os.path.basename(model_name).replace('.pkl', '')
        ax1.plot(sample_indices, predictions, label=f'Model {i+1}', 
                color=color, linewidth=2, alpha=0.8)
    
    ax1.set_title(f'{title} - Time Series', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Predicted Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot for distribution comparison
    ax2.boxplot(predictions_list, tick_labels=[f'Model {i+1}' for i in range(len(predictions_list))])
    ax2.set_title(f'{title} - Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Predicted Value')
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(f'{title} Comparison', fontsize=16, fontweight='bold')
    fig.tight_layout()
    return fig

def check_saved_visualizations():
    """Check the size and status of recently saved visualizations"""
    import glob
    
    viz_files = glob.glob("visualizations/*.png")
    if not viz_files:
        print("No visualization files found in visualizations/ directory")
        return
    
    # Sort by modification time, most recent first
    viz_files.sort(key=os.path.getmtime, reverse=True)
    
    print("\nRecent visualization files:")
    print("=" * 50)
    
    for i, file in enumerate(viz_files[:5]):  # Show last 5 files
        size = os.path.getsize(file)
        mod_time = datetime.fromtimestamp(os.path.getmtime(file)).strftime('%Y-%m-%d %H:%M:%S')
        
        status = "Valid" if size > 1000 else "Blank/Invalid"
        size_mb = size / (1024 * 1024)  # Convert to MB
        
        filename = os.path.basename(file)
        print(f"{i+1}. {filename}")
        print(f"   Size: {size_mb:.2f} MB ({size:,} bytes)")
        print(f"   Modified: {mod_time}")
        print(f"   Status: {status}")
        print()

if __name__ == "__main__":
    clf_predictions, reg_predictions, other_predictions = predict_and_visualize()
