
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "models" / "resnet18_finetuned_v2_best.pth"
OUTPUT_PATH = ROOT / "submission_resnet18_v2.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


class TestDataset(Dataset):
    def __init__(self, images):
        self.images = images
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = np.asarray(self.images[index])

        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = image * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)

        image = Image.fromarray(image).convert("RGB")
        return self.transform(image)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"V2 checkpoint not found: {MODEL_PATH}"
        )

    X_test = np.load(DATA_DIR / "X_test.npy", mmap_mode="r")
    test_ids = np.load(DATA_DIR / "test_ids.npy")
    class_names = np.load(DATA_DIR / "class_names.npy").tolist()

    print("Test images:", len(X_test))
    print("Classes:", class_names)
    print("Checkpoint:", MODEL_PATH.name)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))

    checkpoint = torch.load(
        MODEL_PATH, map_location=device, weights_only=False
    )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    dataset = TestDataset(X_test)
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    predictions = []
    print("Generating V2 predictions...")

    with torch.no_grad():
        for batch_index, images in enumerate(loader):
            images = images.to(device)
            outputs = model(images)
            predicted_indices = outputs.argmax(dim=1)
            predictions.extend(predicted_indices.cpu().tolist())

            if (batch_index + 1) % 100 == 0:
                print(
                    f"Processed {batch_index + 1}/{len(loader)} batches"
                )

    predicted_classes = [class_names[i] for i in predictions]

    submission = pd.DataFrame({
        "image_id": test_ids,
        "class": predicted_classes
    })

    submission.to_csv(OUTPUT_PATH, index=False)

    print("\nV2 submission created:", OUTPUT_PATH)
    print("Rows:", len(submission))
    print("Missing predictions:", submission["class"].isna().sum())
    print("\nFirst five predictions:")
    print(submission.head())

    assert len(submission) == len(X_test)
    assert len(submission) == len(test_ids)
    assert submission["class"].notna().all()
    assert submission["image_id"].is_unique
    assert set(submission["class"]).issubset(set(class_names))

    print("\nAll submission checks passed.")


if __name__ == "__main__":
    main()
