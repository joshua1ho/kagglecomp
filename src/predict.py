
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "models"

BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestDataset(Dataset):
    def __init__(self, image_path):
        self.images = np.load(image_path, mmap_mode="r")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = np.array(self.images[index], dtype=np.uint8, copy=True)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return image


class FacialFeatureCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# Load the trained model
checkpoint = torch.load(
    MODEL_DIR / "best_model.pth",
    map_location=DEVICE,
    weights_only=True
)

class_names = np.load(DATA_DIR / "class_names.npy").tolist()

model = FacialFeatureCNN(len(class_names)).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Load test images and their original IDs
dataset = TestDataset(DATA_DIR / "X_test.npy")
test_ids = np.load(DATA_DIR / "test_ids.npy")

if len(dataset) != len(test_ids):
    raise ValueError("Test image count and test ID count do not match.")

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

# Predict class indices
predictions = []

with torch.no_grad():
    for images in loader:
        images = images.to(DEVICE)
        outputs = model(images)
        predictions.extend(outputs.argmax(dim=1).cpu().numpy())

predicted_classes = [class_names[i] for i in predictions]

# Build predictions keyed by original image ID
predictions_df = pd.DataFrame({
    "image_id": test_ids,
    "class": predicted_classes
})

# Match the provided submission template
template_path = PROJECT_DIR / "data" / "sample_submission.csv"
template = pd.read_csv(template_path)

if not {"image_id", "class"}.issubset(template.columns):
    raise ValueError(
        "The sample submission columns differ from the expected "
        "'image_id' and 'class'. Check sample_submission.csv and "
        "adjust the column names in this script."
    )

if predictions_df["image_id"].duplicated().any():
    raise ValueError("Duplicate test image IDs found.")

submission = template[["image_id"]].merge(
    predictions_df,
    on="image_id",
    how="left",
    validate="one_to_one"
)

if submission["class"].isna().any():
    raise ValueError("Some test IDs have no predictions.")

output_path = PROJECT_DIR / "submission.csv"
submission.to_csv(output_path, index=False)

print("Predictions complete.")
print("Device:", DEVICE)
print("Number of predictions:", len(submission))
print("Submission columns:", submission.columns.tolist())
print("Missing predictions:", submission["class"].isna().sum())
print("\nFirst five rows:")
print(submission.head())
print("\nSaved to:", output_path)