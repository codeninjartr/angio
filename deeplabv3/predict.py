import os
import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from dataset import get_train_val_datasets
from losses import bce_dice_loss
from metrics import iou_score

os.makedirs("outputs", exist_ok=True)

def run_predictions():
    """
    Loads the best trained model, predicts vessel masks on validation images,
    and saves side-by-side visual comparisons (RGB, Ground Truth, Prediction).
    """
    # 1. Load validation data
    # (We load train as well since get_train_val_datasets returns both, but only need val)
    _, _, X_val, Y_val = get_train_val_datasets()

    model_path_keras = "models/best_model.keras"
    model_path_h5 = "models/best_model.h5"
    
    if os.path.exists(model_path_keras):
        model_path = model_path_keras
    elif os.path.exists(model_path_h5):
        model_path = model_path_h5
    else:
        print(f"Error: Model not found. Please run train.py first.")
        return

    # 2. Load model
    print(f"Loading trained model from {model_path}...")
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "bce_dice_loss": bce_dice_loss,
            "iou_score": iou_score
        },
        compile=False
    )

    # 3. Choose a few random samples to visualize
    num_samples = 3
    indices = random.sample(range(len(X_val)), min(num_samples, len(X_val)))
    
    print(f"Generating visual predictions for indices: {indices}")

    for i, idx in enumerate(indices):
        img = X_val[idx]
        gt = Y_val[idx]

        # Model expectation: batch dimension (1, 256, 256, 3)
        input_tensor = np.expand_dims(img, axis=0)
        pred = model.predict(input_tensor)
        
        # Squeeze batch dimension and threshold at 0.5 to get binary mask
        pred_mask = (pred[0] > 0.5).astype(np.float32)

        # Plot side-by-side
        plt.figure(figsize=(12, 4))
        
        # 1. RGB Image
        plt.subplot(1, 3, 1)
        plt.imshow(img)
        plt.title("RGB Embryo Image")
        plt.axis("off")

        # 2. Ground Truth Mask
        plt.subplot(1, 3, 2)
        plt.imshow(gt.squeeze(), cmap="gray")
        plt.title("Ground Truth Mask")
        plt.axis("off")

        # 3. Predicted Mask
        plt.subplot(1, 3, 3)
        plt.imshow(pred_mask.squeeze(), cmap="gray")
        plt.title("Predicted Binary Mask")
        plt.axis("off")

        plt.tight_layout()
        
        # Save output figure
        output_file = f"outputs/prediction_val_idx_{idx}.png"
        plt.savefig(output_file, dpi=150)
        plt.close()
        print(f"Saved prediction comparison to {output_file}")

    print("Prediction run complete!")

if __name__ == "__main__":
    run_predictions()
