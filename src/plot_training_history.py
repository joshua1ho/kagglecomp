
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "models"

history_path = MODEL_DIR / "training_history.csv"
history = pd.read_csv(history_path)

# Check that the required columns exist
required_columns = [
    "epoch",
    "train_loss",
    "train_accuracy",
    "validation_loss",
    "validation_accuracy",
]

missing = [
    column for column in required_columns
    if column not in history.columns
]

if missing:
    raise ValueError(f"Missing columns: {missing}")

# Plot training and validation loss
plt.figure(figsize=(8, 5))
plt.plot(history["epoch"], history["train_loss"], marker="o",
         label="Training loss")
plt.plot(history["epoch"], history["validation_loss"], marker="o",
         label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()

loss_path = MODEL_DIR / "loss_curve.png"
plt.savefig(loss_path, dpi=150)
plt.show()

# Plot training and validation accuracy
plt.figure(figsize=(8, 5))
plt.plot(history["epoch"], history["train_accuracy"], marker="o",
         label="Training accuracy")
plt.plot(history["epoch"], history["validation_accuracy"], marker="o",
         label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()

accuracy_path = MODEL_DIR / "accuracy_curve.png"
plt.savefig(accuracy_path, dpi=150)
plt.show()

print("Learning curves saved:")
print(loss_path)
print(accuracy_path)

print("\nFinal recorded epoch:")
print(history.tail(1).to_string(index=False))

print("\nBest validation accuracy:")
best_row = history.loc[history["validation_accuracy"].idxmax()]
print(f"Epoch: {int(best_row['epoch'])}")
print(f"Accuracy: {best_row['validation_accuracy'] * 100:.2f}%")
