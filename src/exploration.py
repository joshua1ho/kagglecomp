
from pathlib import Path
import pandas as pd
from PIL import Image

# Project and dataset paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

LABELS_FILE = DATA_DIR / "train_labels.csv"
SUBMISSION_FILE = DATA_DIR / "sample_submission.csv"


def main():
    print("Dataset exists:", DATA_DIR.exists())
    print("Training folder exists:", TRAIN_DIR.exists())
    print("Test folder exists:", TEST_DIR.exists())

    # Read the CSV files
    labels = pd.read_csv(LABELS_FILE)
    submission = pd.read_csv(SUBMISSION_FILE)

    print("\nTraining labels:")
    print(labels.head())
    print("\nTraining CSV columns:")
    print(labels.columns.tolist())
    print("\nTraining CSV shape:", labels.shape)

    print("\nSample submission:")
    print(submission.head())
    print("\nSubmission columns:")
    print(submission.columns.tolist())
    print("\nSubmission shape:", submission.shape)

    # Count image files
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    train_images = [
        p for p in TRAIN_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    ]

    test_images = [
        p for p in TEST_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    ]

    print("\nNumber of training images:", len(train_images))
    print("Number of test images:", len(test_images))

    # Inspect one image
    if train_images:
        with Image.open(train_images[0]) as image:
            print("\nExample image:", train_images[0].name)
            print("Image dimensions:", image.size)
            print("Image mode:", image.mode)


if __name__ == "__main__":
    main()
