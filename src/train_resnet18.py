
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
EPOCHS = 12
LEARNING_RATE = 0.0001
PATIENCE = 3
SEED = 42

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

    def __init__(self, image_file, label_file=None,
                 transform=None):

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


# ==========================================
# PRETRAINED RESNET18
# ==========================================

weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# Replace ImageNet's 1000-class output with our 8 classes.
model.fc = nn.Linear(
    model.fc.in_features,
    len(CLASS_NAMES)
)

model = model.to(DEVICE)

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
# TRAINING
# ==========================================

best_val_loss = float("inf")
epochs_without_improvement = 0
history = []

checkpoint_path = MODEL_DIR / "resnet18_best.pth"

for epoch in range(1, EPOCHS + 1):

    # ----- Training -----
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

    # ----- Validation -----
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
        MODEL_DIR / "resnet18_history.csv",
        index=False
    )

    # Save the checkpoint with the lowest validation loss.
    if val_loss < best_val_loss:

        best_val_loss = val_loss
        epochs_without_improvement = 0

        torch.save({
            "model_state_dict": model.state_dict(),
            "class_names": CLASS_NAMES,
            "epoch": epoch,
            "validation_loss": val_loss,
            "validation_accuracy": val_accuracy
        }, checkpoint_path)

        print("  Saved new best ResNet18 checkpoint.")

    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print("Early stopping.")
        break


print("\nTraining finished.")
print("Best validation loss:", best_val_loss)
print("Checkpoint:", checkpoint_path)
print("History:", MODEL_DIR / "resnet18_history.csv")
