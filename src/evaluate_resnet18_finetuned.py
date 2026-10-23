from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt


# ==========================================
# SETTINGS
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "models"

# Load the fine-tuned checkpoint
CHECKPOINT_PATH = MODEL_DIR / "resnet18_finetuned_v2_best.pth"

BATCH_SIZE = 16

MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


# ==========================================
# LOAD CLASS NAMES
# ==========================================

CLASS_NAMES = np.load(
    DATA_DIR / "class_names.npy"
).tolist()

NUM_CLASSES = len(CLASS_NAMES)

print("Classes:", CLASS_NAMES)


# ==========================================
# VALIDATION TRANSFORM
# Must match fine-tuning preprocessing
# ==========================================

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

validation_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(
        mean=imagenet_mean,
        std=imagenet_std
    )
])


# ==========================================
# DATASET
# ==========================================

class FacialFeatureDataset(Dataset):

    def __init__(self, image_file, label_file, transform=None):
        self.images = np.load(image_file, mmap_mode="r")
        self.labels = np.load(label_file)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = np.array(
            self.images[index],
            dtype=np.uint8,
            copy=True
        )

        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1).float() / 255.0

        if self.transform:
            image = self.transform(image)

        label = int(self.labels[index])

        return image, label


# ==========================================
# LOAD VALIDATION DATA
# ==========================================

validation_dataset = FacialFeatureDataset(
    DATA_DIR / "X_validation.npy",
    DATA_DIR / "y_validation.npy",
    validation_transform
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

print("Validation samples:", len(validation_dataset))


# ==========================================
# LOAD FINE-TUNED RESNET18
# ==========================================

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"Fine-tuned checkpoint not found: {CHECKPOINT_PATH}\n"
        "Run src/train_resnet18_finetune.py first."
    )

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=True
)

if checkpoint["class_names"] != CLASS_NAMES:
    raise ValueError(
        "Checkpoint class names do not match class_names.npy"
    )

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    NUM_CLASSES
)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE)
model.eval()

print("Loaded checkpoint:", CHECKPOINT_PATH)

if "epoch" in checkpoint:
    print("Best checkpoint epoch:", checkpoint["epoch"])


# ==========================================
# EVALUATION
# ==========================================

all_labels = []
all_predictions = []

total_loss = 0.0
total_samples = 0

criterion = nn.CrossEntropyLoss()

with torch.no_grad():
    for images, labels in validation_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        predictions = outputs.argmax(dim=1)
        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())

y_true = np.array(all_labels)
y_pred = np.array(all_predictions)

average_loss = total_loss / total_samples
accuracy = accuracy_score(y_true, y_pred)

print("\n" + "=" * 55)
print("FINE-TUNED RESNET18 VALIDATION RESULTS")
print("=" * 55)

print("Number of samples:", total_samples)
print(f"Validation loss: {average_loss:.6f}")
print(f"Validation accuracy: {accuracy:.4%}")


# ==========================================
# CLASSIFICATION REPORT
# ==========================================

report_text = classification_report(
    y_true,
    y_pred,
    labels=list(range(NUM_CLASSES)),
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)

report = classification_report(
    y_true,
    y_pred,
    labels=list(range(NUM_CLASSES)),
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

print("\nClassification report:")
print(report_text)

report_path = MODEL_DIR / "resnet18_finetuned_classification_report.csv"
report_df.to_csv(report_path, index=True)


# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(NUM_CLASSES))
)

cm_df = pd.DataFrame(
    cm,
    index=CLASS_NAMES,
    columns=CLASS_NAMES
)

cm_path = MODEL_DIR / "resnet18_finetuned_confusion_matrix.csv"
cm_df.to_csv(cm_path, index=True)

print("\nConfusion matrix:")
print(cm_df)


# ==========================================
# PER-CLASS ACCURACY
# ==========================================

class_totals = cm.sum(axis=1)
correct_predictions = np.diag(cm)

class_accuracy = np.divide(
    correct_predictions,
    class_totals,
    out=np.zeros(NUM_CLASSES, dtype=float),
    where=class_totals != 0
)

class_accuracy_df = pd.DataFrame({
    "class": CLASS_NAMES,
    "correct_predictions": correct_predictions,
    "total_samples": class_totals,
    "accuracy": class_accuracy
})

print("\nPer-class accuracy:")
print(
    class_accuracy_df.to_string(
        index=False,
        formatters={"accuracy": lambda value: f"{value:.4%}"}
    )
)

class_accuracy_path = MODEL_DIR / "resnet18_finetuned_class_accuracy.csv"
class_accuracy_df.to_csv(class_accuracy_path, index=False)


# ==========================================
# PLOT CONFUSION MATRIX
# ==========================================

fig, ax = plt.subplots(figsize=(10, 8))

image = ax.imshow(cm, interpolation="nearest")
fig.colorbar(image, ax=ax)

ax.set(
    xticks=np.arange(NUM_CLASSES),
    yticks=np.arange(NUM_CLASSES),
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    xlabel="Predicted class",
    ylabel="True class",
    title="Fine-Tuned ResNet18 Validation Confusion Matrix"
)

plt.setp(
    ax.get_xticklabels(),
    rotation=45,
    ha="right",
    rotation_mode="anchor"
)

threshold = cm.max() / 2 if cm.size else 0

for row in range(NUM_CLASSES):
    for col in range(NUM_CLASSES):
        ax.text(
            col,
            row,
            str(cm[row, col]),
            ha="center",
            va="center",
            color="white" if cm[row, col] > threshold else "black"
        )

fig.tight_layout()

confusion_plot_path = (
    MODEL_DIR / "resnet18_finetuned_confusion_matrix.png"
)

fig.savefig(
    confusion_plot_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)


# ==========================================
# SAVE SUMMARY
# ==========================================

summary_path = MODEL_DIR / "resnet18_finetuned_evaluation_summary.txt"

with open(summary_path, "w", encoding="utf-8") as file:
    file.write("FINE-TUNED RESNET18 VALIDATION EVALUATION\n")
    file.write("=" * 45 + "\n\n")

    file.write(f"Checkpoint: {CHECKPOINT_PATH.name}\n")
    file.write(
        f"Checkpoint epoch: {checkpoint.get('epoch', 'Unknown')}\n"
    )
    file.write(f"Validation samples: {total_samples}\n")
    file.write(f"Validation loss: {average_loss:.6f}\n")
    file.write(f"Validation accuracy: {accuracy:.6f}\n")
    file.write(f"Validation accuracy percentage: {accuracy:.2%}\n\n")

    file.write("CLASSIFICATION REPORT\n")
    file.write("-" * 40 + "\n")
    file.write(report_text)

    file.write("\nPER-CLASS ACCURACY\n")
    file.write("-" * 40 + "\n")
    file.write(class_accuracy_df.to_string(index=False))
    file.write("\n")


# ==========================================
# FINISHED
# ==========================================

print("\nEvaluation finished.")
print("Saved files:")
print("1.", summary_path)
print("2.", report_path)
print("3.", cm_path)
print("4.", class_accuracy_path)
print("5.", confusion_plot_path)