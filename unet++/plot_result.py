import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_metrics():
    # Use the unified training log CSV
    log_path = "outputs/training_log.csv"
    if not os.path.exists(log_path):
        print(f"Error: {log_path} not found.")
        return

    df = pd.read_csv(log_path)
    
    # We will plot Loss, Dice Coefficient, and IoU Score
    metrics_to_plot = [
        ("loss", "val_loss", "Loss (BCE + Dice)"),
        ("dice_coeff", "val_dice_coeff", "Dice Coefficient"),
        ("iou_score", "val_iou_score", "IoU Score")
    ]
    
    plt.figure(figsize=(15, 5))
    
    for i, (train_col, val_col, title) in enumerate(metrics_to_plot, 1):
        plt.subplot(1, 3, i)
        if train_col in df.columns:
            plt.plot(df['epoch'], df[train_col], label=f"Train {title}", marker='o', markersize=4)
        if val_col in df.columns:
            plt.plot(df['epoch'], df[val_col], label=f"Val {title}", marker='o', markersize=4)
            
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
    plt.tight_layout()
    
    out_path = "outputs/training_metrics_plot.png"
    plt.savefig(out_path, dpi=150)
    print(f"[OK] Training metrics plot saved to -> {out_path}")
    plt.close()

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    plot_metrics()
