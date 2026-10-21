
from pathlib import Path
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
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

EXPECTED_CLASSES = {
    "CatEye", "CatFace", "CatMouth", "CatNose",
    "DogEye", "DogFace", "DogMouth", "DogNose",
}


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


def image_filename(image_id):
    """Convert an image ID to its expected five-digit JPEG filename."""
    return f"{int(image_id):05d}.jpg"


# --------------------------------------------------
# 3. Explore the dataset
# --------------------------------------------------

def main():

    # --------------------------------------------------
    # 4. Check paths and required files
    # --------------------------------------------------

    print("=" * 60)
    print("DATASET PATHS")
    print("=" * 60)

    print("Project directory:", PROJECT_DIR)
    print("Dataset exists:", DATA_DIR.exists())
    print("Training folder exists:", TRAIN_DIR.exists())
    print("Test folder exists:", TEST_DIR.exists())

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
    # 5. Load and inspect CSV files
    # --------------------------------------------------

    labels = pd.read_csv(LABELS_FILE)
    submission = pd.read_csv(SUBMISSION_FILE)

    print("\n" + "=" * 60)
    print("TRAINING LABELS")
    print("=" * 60)

    print("\nFirst five rows:")
    print(labels.head())

    print("\nColumns:", labels.columns.tolist())
    print("Shape (rows, columns):", labels.shape)

    print("\nData types:")
    print(labels.dtypes)

    print("\nMissing values:")
    print(labels.isnull().sum())

    print(
        "Duplicate training IDs:",
        labels["image_id"].duplicated().sum()
        if "image_id" in labels.columns else "image_id column missing"
    )

    print("\n" + "=" * 60)
    print("SAMPLE SUBMISSION")
    print("=" * 60)

    print("\nFirst five rows:")
    print(submission.head())

    print("\nColumns:", submission.columns.tolist())
    print("Shape (rows, columns):", submission.shape)

    print("\nMissing values:")
    print(submission.isnull().sum())

    print(
        "Duplicate submission IDs:",
        submission["image_id"].duplicated().sum()
        if "image_id" in submission.columns else "image_id column missing"
    )

    # Check required CSV columns before using them
    required_label_columns = {"image_id", "class"}
    required_submission_columns = {"image_id", "class"}

    if not required_label_columns.issubset(labels.columns):
        raise ValueError(
            f"Training CSV must contain {required_label_columns}"
        )

    if not required_submission_columns.issubset(submission.columns):
        raise ValueError(
            f"Submission CSV must contain {required_submission_columns}"
        )

    # --------------------------------------------------
    # 6. Inspect class labels
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("CLASS LABELS")
    print("=" * 60)

    actual_classes = set(labels["class"].dropna().unique())

    print("Unique class labels:", sorted(actual_classes))
    print("Number of distinct classes:", len(actual_classes))

    print("\nUnexpected classes:")
    print(sorted(actual_classes - EXPECTED_CLASSES))

    print("\nExpected classes with no examples:")
    print(sorted(EXPECTED_CLASSES - actual_classes))

    class_counts = labels["class"].value_counts().reindex(
        sorted(EXPECTED_CLASSES), fill_value=0
    )

    print("\nNumber of examples per class:")
    print(class_counts)

    # Plot class distribution
    class_counts.plot(kind="bar")
    plt.title("Training Images per Class")
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------
    # 7. Count image files
    # --------------------------------------------------

    train_images = get_images(TRAIN_DIR)
    test_images = get_images(TEST_DIR)

    print("\n" + "=" * 60)
    print("IMAGE FILES")
    print("=" * 60)

    print("Training image files:", len(train_images))
    print("Test image files:", len(test_images))

    print("\nTraining file extensions:")
    print(Counter(path.suffix.lower() for path in train_images))

    print("\nTest file extensions:")
    print(Counter(path.suffix.lower() for path in test_images))

    print("\nFirst 10 training paths:")
    for path in train_images[:10]:
        print(path.relative_to(PROJECT_DIR))

    print("\nFirst 10 test paths:")
    for path in test_images[:10]:
        print(path.relative_to(PROJECT_DIR))

    # --------------------------------------------------
    # 8. Compare CSV row counts with image counts
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("COUNT CHECKS")
    print("=" * 60)

    print("Training CSV rows:", len(labels))
    print("Training image files:", len(train_images))
    print("Submission CSV rows:", len(submission))
    print("Test image files:", len(test_images))

    print(
        "Training row count matches image count:",
        len(labels) == len(train_images)
    )

    print(
        "Submission row count matches test image count:",
        len(submission) == len(test_images)
    )

    # --------------------------------------------------
    # 9. Verify image IDs against filenames
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("IMAGE ID MAPPING")
    print("=" * 60)

    train_paths_by_name = {}
    test_paths_by_name = {}

    for path in train_images:
        train_paths_by_name.setdefault(path.name, []).append(path)

    for path in test_images:
        test_paths_by_name.setdefault(path.name, []).append(path)

    expected_train_names = {
        image_filename(image_id)
        for image_id in labels["image_id"].dropna()
    }

    expected_test_names = {
        image_filename(image_id)
        for image_id in submission["image_id"].dropna()
    }

    missing_train = expected_train_names - set(train_paths_by_name)
    missing_test = expected_test_names - set(test_paths_by_name)

    print("Expected training filenames:", len(expected_train_names))
    print("Missing training files:", len(missing_train))
    print("Expected test filenames:", len(expected_test_names))
    print("Missing test files:", len(missing_test))

    print("\nFirst five training ID mappings:")

    for image_id in labels["image_id"].head():
        filename = image_filename(image_id)
        paths = train_paths_by_name.get(filename, [])
        print(f"{image_id} -> {filename} -> {paths}")

    if missing_train:
        print("\nExamples of missing training files:")
        print(sorted(missing_train)[:10])

    if missing_test:
        print("\nExamples of missing test files:")
        print(sorted(missing_test)[:10])

    # Check for duplicate basenames in different folders
    duplicate_train_names = {
        name: paths
        for name, paths in train_paths_by_name.items()
        if len(paths) > 1
    }

    duplicate_test_names = {
        name: paths
        for name, paths in test_paths_by_name.items()
        if len(paths) > 1
    }

    print("\nDuplicate training filenames:", len(duplicate_train_names))
    print("Duplicate test filenames:", len(duplicate_test_names))

    # --------------------------------------------------
    # 10. Inspect image properties across all images
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("IMAGE PROPERTIES")
    print("=" * 60)

    dimension_counts = Counter()
    mode_counts = Counter()
    format_counts = Counter()
    unreadable_images = []

    for image_path in train_images + test_images:
        try:
            with Image.open(image_path) as image:
                dimensions = image.size
                mode = image.mode
                image_format = image.format

                image.verify()

            dimension_counts[dimensions] += 1
            mode_counts[mode] += 1
            format_counts[image_format] += 1

        except Exception as error:
            unreadable_images.append(
                (str(image_path), str(error))
            )

    print("Image dimensions (width, height):")

    for dimensions, count in dimension_counts.most_common():
        print(f"{dimensions}: {count}")

    print("\nColour modes:")
    print(mode_counts)

    print("\nImage formats:")
    print(format_counts)

    print("\nUnreadable image count:", len(unreadable_images))

    if unreadable_images:
        print("\nExamples of unreadable files:")
        for path, error in unreadable_images[:10]:
            print(path, "->", error)

    # --------------------------------------------------
    # 11. Display one sample image per class
    # --------------------------------------------------

    print("\nDisplaying representative training images...")

    fig, axes = plt.subplots(2, 4, figsize=(12, 7))
    axes = axes.flatten()

    for ax in axes:
        ax.axis("off")

    for ax, class_name in zip(axes, sorted(EXPECTED_CLASSES)):

        class_rows = labels[labels["class"] == class_name]

        if class_rows.empty:
            ax.set_title(f"{class_name}\nNo examples")
            continue

        image_id = class_rows.iloc[0]["image_id"]
        filename = image_filename(image_id)
        matching_paths = train_paths_by_name.get(filename, [])

        if not matching_paths:
            ax.set_title(f"{class_name}\nImage missing")
            continue

        try:
            with Image.open(matching_paths[0]) as image:
                ax.imshow(image.convert("RGB"))

            ax.set_title(f"{class_name}\nID: {image_id}")

        except Exception as error:
            ax.set_title(f"{class_name}\nCould not open")
            print(f"Could not display {filename}: {error}")

    plt.tight_layout()
    plt.show()

    print("\nDataset exploration finished.")


if __name__ == "__main__":
    main()
