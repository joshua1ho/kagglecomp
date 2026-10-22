
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ==================================================
# 1. SETTINGS
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
PATIENCE = 5
RANDOM_STATE = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

print("=" * 60)
print("FACIAL FEATURE CLASSIFICATION")
print("=" * 60)
print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "GPU memory:",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / (1024 ** 3),
            2
        ),
        "GB"
    )


# ==================================================
# 2. LOAD DATASET USING MEMORY MAPPING
# ==================================================

class FacialFeatureDataset(Dataset):

    def __init__(self, image_file, label_file=None,
                 augmentation=None):

        # Read image arrays from disk without loading
        # the entire dataset into normal RAM.
        self.images = np.load(
            image_file,
            mmap_mode="r"
        )

        self.labels = None

        if label_file is not None:
            self.labels = np.load(label_file)

            if len(self.images) != len(self.labels):
                raise ValueError(
                    "Image and label counts do not match."
                )

        self.augmentation = augmentation

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        # Copy one image so its NumPy memory is writable.
        image = np.array(
            self.images[index],
            dtype=np.uint8,
            copy=True
        )

        # Convert H x W x C into C x H x W.
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1).float()
        image = image / 255.0

        if self.augmentation is not None:
            image = self.augmentation(image)

        if self.labels is None:
            return image

        label = int(self.labels[index])

        return image, label


# Augmentation is applied only to training images.
train_augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=8)
])


# ==================================================
# 3. LOAD TRAINING AND VALIDATION DATA
# ==================================================

required_files = [
    "X_train.npy",
    "X_validation.npy",
    "y_train.npy",
    "y_validation.npy",
    "class_names.npy"
]

for filename in required_files:
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

CLASS_NAMES = np.load(
    DATA_DIR / "class_names.npy"
).tolist()

NUM_CLASSES = len(CLASS_NAMES)

train_dataset = FacialFeatureDataset(
    DATA_DIR / "X_train.npy",
    DATA_DIR / "y_train.npy",
    augmentation=train_augmentation
)

validation_dataset = FacialFeatureDataset(
    DATA_DIR / "X_validation.npy",
    DATA_DIR / "y_validation.npy"
)

print("\nDataset sizes:")
print("Training images:", len(train_dataset))
print("Validation images:", len(validation_dataset))
print("Number of classes:", NUM_CLASSES)
print("Classes:", CLASS_NAMES)

# num_workers=0 is a reliable starting point on Windows.
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


# ==================================================
# 4. DEFINE A CNN FROM SCRATCH
# ==================================================

class FacialFeatureCNN(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        self.features = nn.Sequential(

            # Block 1: 112 x 112 -> 56 x 56
            nn.Conv2d(
                3, 32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 56 x 56 -> 28 x 28
            nn.Conv2d(
                32, 64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 28 x 28 -> 14 x 14
            nn.Conv2d(
                64, 128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4: 14 x 14 -> 7 x 7
            nn.Conv2d(
                128, 256,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Reduce each feature map to one value.
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

        x = self.features(x)
        x = self.classifier(x)

        return x


model = FacialFeatureCNN(NUM_CLASSES).to(DEVICE)

print("\nModel architecture:")
print(model)

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print("\nTrainable parameters:", total_parameters)


# ==================================================
# 5. LOSS FUNCTION AND OPTIMIZER
# ==================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)

BEST_MODEL_PATH = MODEL_DIR / "best_model.pth"
HISTORY_PATH = MODEL_DIR / "training_history.csv"


# ==================================================
# 6. TRAINING FUNCTION
# ==================================================

def train_one_epoch():

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_index, (images, labels) in enumerate(
        train_loader
    ):

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        predictions = outputs.argmax(dim=1)

        total_correct += (
            predictions == labels
        ).sum().item()

        total_samples += batch_size

        if (batch_index + 1) % 200 == 0:
            print(
                f"  Training batch "
                f"{batch_index + 1}/{len(train_loader)}",
                flush=True
            )

    average_loss = total_loss / total_samples

    accuracy = total_correct / total_samples

    return average_loss, accuracy


# ==================================================
# 7. VALIDATION FUNCTION
# ==================================================

def validate():

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():

        for images, labels in validation_loader:

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

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size

            predictions = outputs.argmax(dim=1)

            total_correct += (
                predictions == labels
            ).sum().item()

            total_samples += batch_size

    average_loss = total_loss / total_samples

    accuracy = total_correct / total_samples

    return average_loss, accuracy


# ==================================================
# 8. TRAIN THE MODEL
# ==================================================

print("\nStarting training...")
print("Batch size:", BATCH_SIZE)
print("Maximum epochs:", EPOCHS)
print("Early stopping patience:", PATIENCE)

history = []

best_validation_loss = float("inf")
epochs_without_improvement = 0

for epoch in range(EPOCHS):

    print("\n" + "=" * 60)
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print("=" * 60, flush=True)

    train_loss, train_accuracy = train_one_epoch()

    validation_loss, validation_accuracy = validate()

    scheduler.step(validation_loss)

    current_lr = optimizer.param_groups[0]["lr"]

    print(f"\nTraining loss: {train_loss:.4f}")
    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Validation loss: {validation_loss:.4f}")
    print(f"Validation accuracy: {validation_accuracy:.4f}")
    print(f"Learning rate: {current_lr:.6f}")

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "validation_loss": validation_loss,
        "validation_accuracy": validation_accuracy,
        "learning_rate": current_lr
    })

    # Save the model whenever validation loss improves.
    if validation_loss < best_validation_loss:

        best_validation_loss = validation_loss
        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_names": CLASS_NAMES,
                "epoch": epoch + 1,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy
            },
            BEST_MODEL_PATH
        )

        print("Saved new best model.")

    else:

        epochs_without_improvement += 1

    # Save the training history after each epoch.
    pd.DataFrame(history).to_csv(
        HISTORY_PATH,
        index=False
    )

    if epochs_without_improvement >= PATIENCE:

        print(
            "\nEarly stopping: validation loss "
            "has not improved."
        )

        break


# ==================================================
# 9. FINISH
# ==================================================

print("\n" + "=" * 60)
print("TRAINING FINISHED")
print("=" * 60)

print("Best model:", BEST_MODEL_PATH)
print("Training history:", HISTORY_PATH)
print("Best validation loss:", best_validation_loss)

print(
    "\nThe best model checkpoint contains the model "
    "weights and its best validation metrics."
)

print(
    "The provided test dataset has not been used "
    "for training or validation."
)
