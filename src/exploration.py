
from pathlib import Path

import pandas as pd
from PIL import Image


# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

LABELS_FILE = DATA_DIR / "train_labels.csv"
SUBMISSION_FILE = DATA_DIR / "sample_submission.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --------------------------------------------------
# 2. Find image files
# --------------------------------------------------

def get_images(folder):
    """Find supported image files, including subfolders."""
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


# --------------------------------------------------
# 3. Explore the dataset
# --------------------------------------------------

def main():
    print("=" * 50)
    print("DATASET PATHS")
    print("=" * 50)

    print("Project directory:", PROJECT_DIR)
    print("Dataset exists:", DATA_DIR.exists())
    print("Training folder exists:", TRAIN_DIR.exists())
    print("Test folder exists:", TEST_DIR.exists())

    # Check that the required files and folders exist
    required_paths = [
        TRAIN_DIR,
        TEST_DIR,
        LABELS_FILE,
        SUBMISSION_FILE,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file or folder not found: {path}"
            )

    # --------------------------------------------------
    # 4. Load CSV files
    # --------------------------------------------------

    labels = pd.read_csv(LABELS_FILE)
    submission = pd.read_csv(SUBMISSION_FILE)

    print("\n" + "=" * 50)
    print("TRAINING LABELS")
    print("=" * 50)

    print("\nFirst five rows:")
    print(labels.head())

    print("\nColumn names:")
    print(labels.columns.tolist())

    print("\nDataset shape (rows, columns):")
    print(labels.shape)

    print("\n" + "=" * 50)
    print("SAMPLE SUBMISSION")
    print("=" * 50)

    print("\nFirst five rows:")
    print(submission.head())

    print("\nColumn names:")
    print(submission.columns.tolist())

    print("\nDataset shape (rows, columns):")
    print(submission.shape)

    # --------------------------------------------------
    # 5. Check CSV structure
    # --------------------------------------------------

    print("\n" + "=" * 50)
    print("CSV VALIDATION")
    print("=" * 50)

    required_columns = {"image_id", "class"}

    print(
        "\nTraining CSV has required columns:",
        required_columns.issubset(labels.columns),
    )

    print(
        "Submission CSV has required columns:",
        required_columns.issubset(submission.columns),
    )

    print("\nMissing values in training CSV:")
    print(labels.isnull().sum())

    print("\nMissing values in submission CSV:")
    print(submission.isnull().sum())

    if "image_id" in labels.columns:
        print(
            "\nDuplicate training image IDs:",
            labels["image_id"].duplicated().sum(),
        )

    if "image_id" in submission.columns:
        print(
            "Duplicate submission image IDs:",
            submission["image_id"].duplicated().sum(),
        )

    # --------------------------------------------------
    # 6. Inspect class labels
    # --------------------------------------------------

    if "class" in labels.columns:
        print("\n" + "=" * 50)
        print("CLASS LABELS")
        print("=" * 50)

        print("\nUnique class labels:")
        print(sorted(labels["class"].dropna().unique()))

        print("\nNumber of examples per class:")
        print(labels["class"].value_counts().sort_index())

        print(
            "\nTotal number of distinct classes:",
            labels["class"].nunique(),
        )

    # --------------------------------------------------
    # 7. Count training and test images
    # --------------------------------------------------

    train_images = get_images(TRAIN_DIR)
    test_images = get_images(TEST_DIR)

    print("\n" + "=" * 50)
    print("IMAGE FILES")
    print("=" * 50)

    print("\nNumber of training images:", len(train_images))
    print("Number of test images:", len(test_images))

    print("\nFirst 10 training image paths:")
    for path in train_images[:10]:
        print(path.relative_to(PROJECT_DIR))

    print("\nFirst 10 test image paths:")
    for path in test_images[:10]:
        print(path.relative_to(PROJECT_DIR))

    # --------------------------------------------------
    # 8. Inspect one example from each folder
    # --------------------------------------------------

    print("\n" + "=" * 50)
    print("IMAGE PROPERTIES")
    print("=" * 50)

    examples = [
        ("Training", train_images),
        ("Test", test_images),
    ]

    for name, images in examples:
        if images:
            image_path = images[0]

            try:
                with Image.open(image_path) as image:
                    print(f"\n{name} example:")
                    print("Filename:", image_path.name)
                    print("Format:", image.format)
                    print("Dimensions (width, height):", image.size)
                    print("Colour mode:", image.mode)

            except Exception as error:
                print(f"\nCould not open {name.lower()} image:")
                print("Filename:", image_path.name)
                print("Error:", error)
        else:
            print(f"\nNo {name.lower()} images found.")

    # --------------------------------------------------
    # 9. Compare CSV row counts with image counts
    # --------------------------------------------------

    print("\n" + "=" * 50)
    print("COUNT CHECKS")
    print("=" * 50)

    print("Training CSV rows:", len(labels))
    print("Training image files:", len(train_images))
    print("Submission CSV rows:", len(submission))
    print("Test image files:", len(test_images))

    print(
        "\nTraining CSV row count matches image count:",
        len(labels) == len(train_images),
    )

    print(
        "Submission row count matches test image count:",
        len(submission) == len(test_images),
    )

    print("\nDataset exploration finished.")


if __name__ == "__main__":
    main()
