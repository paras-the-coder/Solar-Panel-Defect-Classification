import os
import argparse
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from src.data import prepare_datasets

def run_evaluation(model_path, data_dir, plots_dir="./plots", seed=42):
    """
    Evaluates the trained model on the unseen test split:
    1. Loads dataset splits (retrieves the test dataset).
    2. Loads the Keras model.
    3. Runs predictions and extracts predictions vs true labels.
    4. Computes Scikit-Learn classification report.
    5. Plots and saves confusion matrix.
    """
    print(f"Loading test split of dataset from: {data_dir}")
    try:
        # Load datasets (using standard batch size of 32)
        datasets, class_names, _ = prepare_datasets(
            data_dir=data_dir,
            batch_size=32,
            seed=seed
        )
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return
        
    _, _, test_ds = datasets
    
    print(f"Loading trained model from: {model_path}")
    if not os.path.exists(model_path):
        print(f"Error: Model file does not exist at '{model_path}'. Please run train.py first.")
        return
        
    try:
        # Since compile=False was used when loading, we compile with standard metric configurations
        # We load without custom_objects for preprocess_input since we removed it from the model layer.
        model = tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    print("Running predictions on the test set...")
    y_true = []
    y_pred = []
    
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 1. Output and Save Classification Report
    print("\n" + "="*60)
    print("                CLASSIFICATION REPORT (TEST SET)          ")
    print("="*60)
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)
    print("="*60)
    
    os.makedirs(plots_dir, exist_ok=True)
    report_path = os.path.join(plots_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Classification report text saved successfully to: {report_path}")
    
    # 2. Compute and Save Confusion Matrix Plot
    print("Computing confusion matrix...")
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    plot_path = os.path.join(plots_dir, "confusion_matrix.png")
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        xticklabels=class_names, 
        yticklabels=class_names
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Solar Panel Defect Classification: Confusion Matrix (Test Split)")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Confusion matrix plot saved successfully to: {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solar Panel Defect Classification: Modular Evaluation Script")
    parser.add_argument("--model", type=str, default="./models/best_model.keras", help="Path to trained model")
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to data directory containing class subdirectories")
    parser.add_argument("--plots_dir", type=str, default="./plots", help="Directory where metrics plots will be saved")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (must match training script seed for split alignment)")
    
    args = parser.parse_args()
    
    run_evaluation(
        model_path=args.model,
        data_dir=args.data_dir,
        plots_dir=args.plots_dir,
        seed=args.seed
    )
