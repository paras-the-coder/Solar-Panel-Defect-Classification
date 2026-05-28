import os
import argparse
import tensorflow as tf
import keras
from src.data import prepare_datasets
from src.models import build_resnet_model

def run_training(data_dir, output_model_path, batch_size=32, initial_epochs=10, fine_tune_epochs=15, seed=42):
    """
    Executes the training pipeline:
    1. Loads datasets with stratified splits.
    2. Builds the ResNet50 model.
    3. Trains the classification head with class weights.
    4. Fine-tunes the top 30 layers of the backbone with class weights.
    5. Saves the final best model weights.
    """
    print("TensorFlow Version:", tf.__version__)
    
    # 1. Prepare Datasets and Class Weights
    print(f"Loading data from: {data_dir}")
    try:
        datasets, class_names, class_weight_dict = prepare_datasets(
            data_dir=data_dir,
            batch_size=batch_size,
            seed=seed
        )
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return
        
    train_ds, val_ds, _ = datasets
    num_classes = len(class_names)
    print(f"Detected classes: {class_names}")
    print(f"Calculated class weights: {class_weight_dict}")
    
    # 2. Build the ResNet50 model (without duplicate internal Lambda preprocessor)
    print("Building ResNet50 transfer learning model...")
    model, base_model = build_resnet_model(
        num_classes=num_classes,
        dense_units=512,
        dropout_rate=0.2
    )
    
    # 3. Compile and Train the head (Feature Extraction)
    # Using the best learning rate found by the Keras Tuner (approx 2.5e-4)
    lr_initial = 2.5e-4
    print(f"Step 1: Training the classification head for {initial_epochs} epochs (LR={lr_initial})...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_initial),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=initial_epochs,
        class_weight=class_weight_dict,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        ]
    )
    
    # 4. Fine-Tuning: Unfreeze the top 30 layers of ResNet50
    print("\nStep 2: Unfreezing top 30 layers of ResNet50 backbone for fine-tuning...")
    base_model.trainable = True
    # Freeze all layers except the last 30
    for layer in base_model.layers[:-30]:
        layer.trainable = False
        
    # Recompile with a very low learning rate (1e-5) to prevent destroying pre-trained weights
    lr_finetune = 1e-5
    print(f"Compining model for fine-tuning (LR={lr_finetune})...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_finetune),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Create output directory for saving the model
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        filepath=output_model_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    early_stopping_cb = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    print(f"Fine-tuning the model for up to {fine_tune_epochs} epochs...")
    history_finetune = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=fine_tune_epochs,
        class_weight=class_weight_dict,
        callbacks=[checkpoint_cb, early_stopping_cb]
    )
    
    # Save the absolute final model (just in case model checkpoint didn't trigger, 
    # but restore_best_weights was set by EarlyStopping)
    model.save(output_model_path)
    print(f"\nTraining pipeline completed successfully! Model saved to: {output_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solar Panel Defect Classification: Modular Training Script")
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to data directory containing class subdirectories")
    parser.add_argument("--output_model", type=str, default="./modular_pipeline/models/best_model.keras", help="Path to save the output model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for dataset training")
    parser.add_argument("--initial_epochs", type=int, default=10, help="Epochs to train the classification head")
    parser.add_argument("--fine_tune_epochs", type=int, default=15, help="Max epochs for backbone fine-tuning")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data splitting and initialization")
    
    args = parser.parse_args()
    
    run_training(
        data_dir=args.data_dir,
        output_model_path=args.output_model,
        batch_size=args.batch_size,
        initial_epochs=args.initial_epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        seed=args.seed
    )
