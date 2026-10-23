
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image, ImageDraw


# ==========================================
# SETTINGS
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "models"

CHECKPOINT_PATH = MODEL_DIR / "resnet18_best.pth"
OUTPUT_DIR = MODEL_DIR / "misclassified_images"

BATCH_SIZE = 16
MAX_IMAGES_TO_SAVE = 100

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


# ==========================================
# CLASS NAMES
# ==========================================

CLASS_NAMES = np.load(
    DATA_DIR / "class_names.npy"
).tolist()

NUM_CLASSES = len(CLASS_NAMES)

print("Classes:", CLASS_NAMES)


# ==========================================
# VALIDATION TRANSFORM
# Must match ResNet18 evaluation
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
        self.images = np.load(
            image_file,
            mmap_mode="r"
        )
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

        # Keep an unmodified copy for saving the image.
        original_image = image.copy()

        image_tensor = torch.from_numpy(image)
        image_tensor = image_tensor.permute(2, 0, 1).float() / 255.0

        if self.transform:
            image_tensor = self.transform(image_tensor)

        label = int(self.labels[index])

        return image_tensor, label, original_image, index


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
# LOAD RESNET18 CHECKPOINT
# ==========================================

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT_PATH}"
    )

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=True
)

if checkpoint["class_names"] != CLASS_NAMES:
    raise ValueError(
        "Checkpoint class names do not match dataset."
    )

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    NUM_CLASSES
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()

print("Loaded checkpoint:", CHECKPOINT_PATH)


# ==========================================
# FIND INCORRECT PREDICTIONS
# ==========================================

errors = []

with torch.no_grad():

    for images, labels, original_images, indices in validation_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)

        confidences, predictions = probabilities.max(dim=1)

        for i in range(len(labels)):

            true_label = int(labels[i].item())
            predicted_label = int(predictions[i].item())
            confidence = float(confidences[i].item())

            if true_label != predicted_label:

                errors.append({
                    "index": int(indices[i]),
                    "true_label": true_label,
                    "true_class": CLASS_NAMES[true_label],
                    "predicted_label": predicted_label,
                    "predicted_class": CLASS_NAMES[predicted_label],
                    "confidence": confidence,
                    "image": original_images[i].numpy()
                })


# ==========================================
# SUMMARY
# ==========================================

print("\n" + "=" * 55)
print("MISCLASSIFICATION SUMMARY")
print("=" * 55)

print("Total validation images:", len(validation_dataset))
print("Incorrect predictions:", len(errors))
print(
    "Validation accuracy:",
    f"{(1 - len(errors) / len(validation_dataset)):.4%}"
)

if not errors:
    print("No incorrect predictions found.")
    raise SystemExit(0)


# ==========================================
# SAVE ERROR DETAILS TO CSV
# ==========================================

error_records = []

for error in errors:

    error_records.append({
        "validation_index": error["index"],
        "true_class": error["true_class"],
        "predicted_class": error["predicted_class"],
        "confidence": error["confidence"]
    })

errors_df = pd.DataFrame(error_records)

csv_path = MODEL_DIR / "resnet18_misclassified.csv"
errors_df.to_csv(csv_path, index=False)

print("\nIncorrect predictions by class pair:")
print(
    errors_df.groupby(
        ["true_class", "predicted_class"]
    ).size().sort_values(ascending=False)
)

print("\nSaved error list:", csv_path)


# ==========================================
# SAVE INCORRECT IMAGES WITH LABELS
# ==========================================

saved_count = 0

for error in errors[:MAX_IMAGES_TO_SAVE]:

    image_array = error["image"]

    image = Image.fromarray(image_array).convert("RGB")
    image = image.resize((336, 336))

    # Add a white information area below the image.
    canvas = Image.new("RGB", (336, 405), "white")
    canvas.paste(image, (0, 0))

    draw = ImageDraw.Draw(canvas)

    draw.text(
        (8, 342),
        f"True: {error['true_class']}",
        fill="black"
    )

    draw.text(
        (8, 361),
        f"Predicted: {error['predicted_class']}",
        fill="red"
    )

    draw.text(
        (8, 380),
        f"Confidence: {error['confidence']:.2%}",
        fill="black"
    )

    filename = (
        f"index_{error['index']:05d}_"
        f"true_{error['true_class']}_"
        f"pred_{error['predicted_class']}.png"
    )

    canvas.save(OUTPUT_DIR / filename)
    saved_count += 1


# ==========================================
# FINISHED
# ==========================================

print("\nError analysis finished.")
print("Misclassified images saved:", saved_count)
print("Image folder:", OUTPUT_DIR)
print("CSV file:", csv_path)
