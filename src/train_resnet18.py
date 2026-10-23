
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


# ==========================================
# SETTINGS
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 16
EPOCHS = 8
LEARNING_RATE = 0.00003
PATIENCE = 3
SEED = 42

CHECKPOINT_PATH = MODEL_DIR / "resnet18_best.pth"
FINETUNED_PATH = MODEL_DIR / "resnet18_finetuned_best.pth"
HISTORY_PATH = MODEL_DIR / "resnet18_finetune_history.csv"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASS_NAMES = np.load(
    DATA_DIR / "class_names.npy"
).tolist()

print("Device:", DEVICE)
print("Classes:", CLASS_NAMES)


# ==========================================
# TRANSFORMS
# ==========================================

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=8),
    transforms.Normalize(imagenet_mean, imagenet_std)
])

validation_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(imagenet_mean, imagenet_std)
])


# ==========================================
# DATASET
# ==========================================

class FacialFeatureDataset(Dataset):

    def __init__(
        self,
        image_file,
        label_file=None,
        transform=None
    ):
        self.images = np.load(
            image_file,
            mmap_mode="r"
        )

        self.labels = (
            np.load(label_file)
            if label_file is not None
            else None
        )

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

        if self.labels is None:
            return image

        return image, int(self.labels[index])


# ==========================================
# LOAD DATA
# ==========================================

train_dataset = FacialFeatureDataset(
    DATA_DIR / "X_train.npy",
    DATA_DIR / "y_train.npy",
    train_transform
)

validation_dataset = FacialFeatureDataset(
    DATA_DIR / "X_validation.npy",
    DATA_DIR / "y_validation.npy",
    validation_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

print("Training samples:", len(train_dataset))
print("Validation samples:", len(validation_dataset))


# ==========================================
# LOAD EXISTING RESNET18 CHECKPOINT
# ==========================================

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT_PATH}\n"
        "Run src/train_resnet18.py first."
    )

# Recreate the same architecture.
model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    len(CLASS_NAMES)
)

# Load your previously trained model.
checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=True
)

if checkpoint["class_names"] != CLASS_NAMES:
    raise ValueError(
        "Checkpoint class names do not match the dataset."
    )

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)

print("Loaded checkpoint:", CHECKPOINT_PATH)
print("Previous best epoch:", checkpoint["epoch"])
print(
    "Previous validation accuracy:",
    f'{checkpoint["validation_accuracy"]:.4f}'
)
print(
    "Previous validation loss:",
    f'{checkpoint["validation_loss"]:.4f}'
)


# ==========================================
# LOSS, OPTIMIZER AND SCHEDULER
# ==========================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=1
)


# ==========================================
# FINE-TUNING
# ==========================================

best_val_loss = float("inf")
best_val_accuracy = 0.0
epochs_without_improvement = 0
history = []

for epoch in range(1, EPOCHS + 1):

    # --------------------------------------
    # TRAINING
    # --------------------------------------

    model.train()

    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)

        train_loss_sum += loss.item() * batch_size

        train_correct += (
            outputs.argmax(dim=1) == labels
        ).sum().item()

        train_total += batch_size

    train_loss = train_loss_sum / train_total
    train_accuracy = train_correct / train_total

    # --------------------------------------
    # VALIDATION
    # --------------------------------------

    model.eval()

    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)

            val_loss_sum += loss.item() * batch_size

            val_correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()

            val_total += batch_size

    val_loss = val_loss_sum / val_total
    val_accuracy = val_correct / val_total

    # Adjust learning rate when validation loss stalls.
    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch {epoch}/{EPOCHS} | "
        f"Train loss: {train_loss:.4f} | "
        f"Train accuracy: {train_accuracy:.4f} | "
        f"Val loss: {val_loss:.4f} | "
        f"Val accuracy: {val_accuracy:.4f} | "
        f"LR: {current_lr:.6f}"
    )

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "validation_loss": val_loss,
        "validation_accuracy": val_accuracy,
        "learning_rate": current_lr
    })

    pd.DataFrame(history).to_csv(
        HISTORY_PATH,
        index=False
    )

    # --------------------------------------
    # SAVE BEST CHECKPOINT
    # --------------------------------------

    # Select by validation loss, consistent with
    # the original training script.
    if val_loss < best_val_loss:

        best_val_loss = val_loss
        best_val_accuracy = val_accuracy
        epochs_without_improvement = 0

        torch.save({
            "model_state_dict": model.state_dict(),
            "class_names": CLASS_NAMES,
            "epoch": epoch,
            "validation_loss": val_loss,
            "validation_accuracy": val_accuracy
        }, FINETUNED_PATH)

        print("  Saved new best fine-tuned checkpoint.")

    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print("Early stopping.")
        break


# ==========================================
# FINISHED
# ==========================================

print("\nFine-tuning finished.")
print(f"Best validation loss: {best_val_loss:.6f}")
print(f"Accuracy at that checkpoint: {best_val_accuracy:.4%}")
print("Checkpoint:", FINETUNED_PATH)
print("History:", HISTORY_PATH)
