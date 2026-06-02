import re
import matplotlib.pyplot as plt

filepaths = ["outputs/result.txt", "outputs/result_part2.txt"]

losses = []
accuracies = []
ious = []
val_losses = []
val_accuracies = []
val_ious = []

for filepath in filepaths:
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            # We look for lines that contain "val_loss:" to ensure it's the end of an epoch
            if "val_loss:" in line:
                try:
                    loss_match = re.search(r'- loss: ([\d\.]+)', line)
                    if loss_match: losses.append(float(loss_match.group(1)))
                    
                    acc_match = re.search(r'- accuracy: ([\d\.]+)', line)
                    if acc_match: accuracies.append(float(acc_match.group(1)))
                    
                    iou_match = re.search(r'- iou_score: ([\d\.]+)', line)
                    if iou_match: ious.append(float(iou_match.group(1)))
                    
                    val_loss_match = re.search(r'- val_loss: ([\d\.]+)', line)
                    if val_loss_match: val_losses.append(float(val_loss_match.group(1)))
                    
                    val_acc_match = re.search(r'- val_accuracy: ([\d\.]+)', line)
                    if val_acc_match: val_accuracies.append(float(val_acc_match.group(1)))
                    
                    val_iou_match = re.search(r'- val_iou_score: ([\d\.]+)', line)
                    if val_iou_match: val_ious.append(float(val_iou_match.group(1)))
                except Exception as e:
                    print("Error parsing line:", line)

epochs_range = range(1, len(losses) + 1)
print(f"Parsed {len(losses)} total epochs from both logs.")

plt.figure(figsize=(18, 5))

# Plot Loss Curve
plt.subplot(1, 3, 1)
plt.plot(epochs_range, losses, 'b-o', label='Training Loss')
plt.plot(epochs_range, val_losses, 'r-o', label='Validation Loss')
plt.title('Loss Curves (Combined)')
plt.xlabel('Effective Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot Accuracy Curve
plt.subplot(1, 3, 2)
plt.plot(epochs_range, accuracies, 'b-o', label='Training Accuracy')
plt.plot(epochs_range, val_accuracies, 'r-o', label='Validation Accuracy')
plt.title('Accuracy Curves (Combined)')
plt.xlabel('Effective Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Plot IoU Curve
plt.subplot(1, 3, 3)
plt.plot(epochs_range, ious, 'b-o', label='Training IoU')
plt.plot(epochs_range, val_ious, 'r-o', label='Validation IoU')
plt.title('IoU Score Curves (Combined)')
plt.xlabel('Effective Epochs')
plt.ylabel('IoU Score')
plt.legend()
plt.grid(True)

plt.tight_layout()
curves_file = "outputs/parsed_training_curves.png"
plt.savefig(curves_file, dpi=150)
plt.close()
print(f"Saved learning curves plot to {curves_file}")
