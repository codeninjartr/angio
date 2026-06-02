import os
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from dataset import get_train_val_datasets
from model import DeepLabV3Plus
from losses import bce_dice_loss
from metrics import iou_score

# Ensure target directories exist before running
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

def train_pipeline():
    """
    Main pipeline to load datasets, compile the DeepLabV3+ model, 
    and run training with early stopping and learning rate scheduler.
    """
    # 1. Load training and validation datasets
    X_train, Y_train, X_val, Y_val = get_train_val_datasets()

    # 2. Instantiate or Load DeepLabV3+ model
    model_path_keras = "models/best_model.keras"
    model_path_h5 = "models/best_model.h5"
    
    loaded_model = False
    if os.path.exists(model_path_keras):
        print(f"Loading existing model from {model_path_keras} to resume training...")
        model = tf.keras.models.load_model(
            model_path_keras,
            custom_objects={"bce_dice_loss": bce_dice_loss, "iou_score": iou_score},
            compile=False
        )
        loaded_model = True
    elif os.path.exists(model_path_h5):
        print(f"Loading existing model from {model_path_h5} to resume training...")
        model = tf.keras.models.load_model(
            model_path_h5,
            custom_objects={"bce_dice_loss": bce_dice_loss, "iou_score": iou_score},
            compile=False
        )
        loaded_model = True
    else:
        print("Building DeepLabV3+ model...")
        model = DeepLabV3Plus(input_shape=(256, 256, 3))
        
    # 3. Compile the model
    print("Compiling model...")
    # Use a lower learning rate if resuming to prevent loss spikes
    lr = 2e-5 if loaded_model else 1e-4
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=bce_dice_loss,
        metrics=[
            "accuracy",
            iou_score
        ]
    )
        
    model.summary()

    # 4. Define Callbacks
    # We save best weights using model checkpoint
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath="models/best_model.keras",
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            save_weights_only=False
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
    ]

    # 5. Start Model Training
    print("Starting training loop...")
    history = model.fit(
        x=X_train,
        y=Y_train,
        validation_data=(X_val, Y_val),
        batch_size=4,
        epochs=50,
        callbacks=callbacks
    )

    # 6. Save Training History
    print("Saving training history...")
    history_file = "outputs/training_history.json"
    with open(history_file, "w") as f:
        json.dump(history.history, f, indent=4)
    print(f"Saved training history to {history_file}")

    # 7. Plot and Save Learning Curves
    print("Plotting learning curves...")
    epochs_range = range(1, len(history.history['loss']) + 1)
    
    plt.figure(figsize=(18, 5))
    
    # Plot Loss Curve
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, history.history['loss'], 'b-o', label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(epochs_range, history.history['val_loss'], 'r-o', label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot Accuracy Curve
    plt.subplot(1, 3, 2)
    acc_key = 'accuracy' if 'accuracy' in history.history else 'acc'
    if acc_key in history.history:
        plt.plot(epochs_range, history.history[acc_key], 'b-o', label='Training Accuracy')
        if f'val_{acc_key}' in history.history:
            plt.plot(epochs_range, history.history[f'val_{acc_key}'], 'r-o', label='Validation Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot IoU Curve
    plt.subplot(1, 3, 3)
    iou_key = 'iou_score' if 'iou_score' in history.history else 'iou'
    if iou_key in history.history:
        plt.plot(epochs_range, history.history[iou_key], 'b-o', label='Training IoU')
        if f'val_{iou_key}' in history.history:
            plt.plot(epochs_range, history.history[f'val_{iou_key}'], 'r-o', label='Validation IoU')
    plt.title('IoU Score Curves')
    plt.xlabel('Epochs')
    plt.ylabel('IoU Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    curves_file = "outputs/training_curves.png"
    plt.savefig(curves_file, dpi=150)
    plt.close()
    print(f"Saved learning curves plot to {curves_file}")

    # 8. Final Evaluation
    print("Evaluating model on validation set...")
    val_results = model.evaluate(X_val, Y_val, verbose=1)
    
    print("\n--- Validation Results ---")
    metrics_names = model.metrics_names
    for name, value in zip(metrics_names, val_results):
        print(f"  {name}: {value:.4f}")

if __name__ == "__main__":
    train_pipeline()
