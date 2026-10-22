
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ==================================================
# 1. SETTINGS
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "models"

BATCH_SIZE = 32

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASS_NAMES = np.load(
    DATA_DIR / "class_names.npy"
).tolist()


# ==================================================
# 2. DATASET
# ==================================================

class FacialFeatureDataset(Dataset):

    def __init__(self, image_file, label_file):
        self.images = np.load(
            image_file,
            mmap_mode="r"
        )

        self.labels = np.load(label_file)

        if len(self.images) != len(self.labels):
            raise ValueError(
                "Image and label counts do not match."
            )

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

        return image, int(self.labels[index])


# ==================================================
# 3. CNN ARCHITECTURE
# Must match train_model.py exactly
# ==================================================

class FacialFeatureCNN(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ==================================================
# 4. LOAD CHECKPOINT
# ==================================================

checkpoint_path = MODEL_DIR / "best_model.pth"

if not checkpoint_path.exists():
    raise FileNotFoundError(
        f"Model checkpoint not found: {checkpoint_path}"
    )

# The checkpoint is your own locally generated model file.
checkpoint = torch.load(
    checkpoint_path,
    map_location=DEVICE,
    weights_only=True
)

model = FacialFeatureCNN(
    num_classes=len(CLASS_NAMES)
).to(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Device:", DEVICE)
print("Checkpoint epoch:", checkpoint["epoch"])
print("Checkpoint validation loss:", checkpoint["validation_loss"])


# ==================================================
# 5. LOAD VALIDATION DATA
# ==================================================

dataset = FacialFeatureDataset(
    DATA_DIR / "X_validation.npy",
    DATA_DIR / "y_validation.npy"
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

criterion = nn.CrossEntropyLoss()

all_labels = []
all_predictions = []
total_loss = 0.0
total_samples = 0


# ==================================================
# 6. EVALUATE
# ==================================================

with torch.no_grad():

    for images, labels in loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        outputs = model(images)

        loss = criterion(outputs, labels)

        predictions = outputs.argmax(dim=1)

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())


# ==================================================
# 7. PRINT METRICS
# ==================================================

average_loss = total_loss / total_samples

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

print("\n" + "=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)

print(f"Validation loss: {average_loss:.4f}")
print(f"Validation accuracy: {accuracy:.4f}")
print(f"Validation accuracy (%): {accuracy * 100:.2f}%")

print("\nClassification report:")

report = classification_report(
    all_labels,
    all_predictions,
    labels=list(range(len(CLASS_NAMES))),
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)

print(report)

print("Confusion matrix:")

matrix = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(len(CLASS_NAMES)))
)

print(
    pd.DataFrame(
        matrix,
        index=CLASS_NAMES,
        columns=CLASS_NAMES
    )
)


# ==================================================
# 8. SAVE RESULTS
# ==================================================

results_path = MODEL_DIR / "validation_results.txt"

with open(results_path, "w", encoding="utf-8") as file:

    file.write("VALIDATION RESULTS\n")
    file.write("==================\n\n")

    file.write(f"Validation loss: {average_loss:.4f}\n")
    file.write(f"Validation accuracy: {accuracy:.4f}\n")
    file.write(f"Validation accuracy (%): {accuracy * 100:.2f}%\n\n")

    file.write("CLASSIFICATION REPORT\n")
    file.write(report)
    file.write("\n\nCONFUSION MATRIX\n")

    file.write(
        pd.DataFrame(
            matrix,
            index=CLASS_NAMES,
            columns=CLASS_NAMES
        ).to_string()
    )

print(f"\nResults saved to: {results_path}")
