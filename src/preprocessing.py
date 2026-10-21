
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# 1. Define project paths and settings
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

LABELS_FILE = DATA_DIR / "train_labels.csv"
SUBMISSION_FILE = DATA_DIR / "sample_submission.csv"

# All images will be resized to this size.
IMAGE_SIZE = (112, 112)

# Reproducible train/validation split.
VALIDATION_SIZE = 0.2
RANDOM_STATE = 42

EXPECTED_CLASSES = [
    "CatEye",
    "CatFace",
    "CatMouth",
    "CatNose",
    "DogEye",
    "DogFace",
    "DogMouth",
    "DogNose",
]


# --------------------------------------------------
# 2. Helper functions
# --------------------------------------------------

def get_image_path(folder, image_id):
    """Find an image by its ID, including nested directories."""

    filename = f"{int(image_id):05d}.jpg"
    matches = list(folder.rglob(filename))

    if len(matches) == 0:
        raise FileNotFoundError(
            f"Could not find image {filename} in {folder}"
        )

    if len(matches) > 1:
        raise ValueError(
            f"Multiple files named {filename} found in {folder}"
        )

    return matches[0]


def load_image(image_path):
    """Load an image, convert it to RGB, resize it, and scale pixels."""

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize(
            IMAGE_SIZE,
            Image.Resampling.BILINEAR
        )

        image_array = np.asarray(image, dtype=np.float32)

    # Convert pixel values from 0-255 to 0-1.
    image_array = image_array / 255.0

    return image_array


def load_images_from_ids(image_ids, folder):
    """Load a list of image IDs into a NumPy array."""

    images = []

    total = len(image_ids)

    for index, image_id in enumerate(image_ids, start=1):
        image_path = get_image_path(folder, image_id)
        image_array = load_image(image_path)
        images.append(image_array)

        if index % 5000 == 0 or index == total:
            print(f"Loaded {index}/{total} images from {folder.name}")

    return np.stack(images).astype(np.float32)


# --------------------------------------------------
# 3. Main preprocessing workflow
# --------------------------------------------------

def main():

    print("=" * 60)
    print("PREPROCESSING")
    print("=" * 60)

    # Check paths.
    for path in [
        TRAIN_DIR,
        TEST_DIR,
        LABELS_FILE,
        SUBMISSION_FILE,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required path does not exist: {path}"
            )

    # Load CSV files.
    labels_df = pd.read_csv(LABELS_FILE)
    submission_df = pd.read_csv(SUBMISSION_FILE)

    # Validate CSV columns.
    if not {"image_id", "class"}.issubset(labels_df.columns):
        raise ValueError(
            "Training labels CSV must contain image_id and class."
        )

    if "image_id" not in submission_df.columns:
        raise ValueError(
            "Sample submission CSV must contain image_id."
        )

    if labels_df[["image_id", "class"]].isnull().any().any():
        raise ValueError(
            "Training labels contain missing image IDs or classes."
        )

    if labels_df["image_id"].duplicated().any():
        raise ValueError(
            "Training labels contain duplicate image IDs."
        )

    if submission_df["image_id"].isnull().any():
        raise ValueError(
            "Sample submission contains missing image IDs."
        )

    if submission_df["image_id"].duplicated().any():
        raise ValueError(
            "Sample submission contains duplicate image IDs."
        )

    # Verify class labels.
    actual_classes = set(labels_df["class"].unique())
    expected_classes = set(EXPECTED_CLASSES)

    if actual_classes != expected_classes:
        raise ValueError(
            f"Unexpected class labels. Found: {sorted(actual_classes)}"
        )

    # Use a fixed class-to-number mapping.
    class_to_index = {
        class_name: index
        for index, class_name in enumerate(EXPECTED_CLASSES)
    }

    index_to_class = {
        index: class_name
        for class_name, index in class_to_index.items()
    }

    print("\nClass-to-index mapping:")
    for class_name, index in class_to_index.items():
        print(f"{class_name}: {index}")

    # --------------------------------------------------
    # 4. Split labelled data before preprocessing
    # --------------------------------------------------

    train_ids, validation_ids = train_test_split(
        labels_df["image_id"].to_numpy(),
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels_df["class"],
    )

    # Keep labels aligned with their image IDs.
    labels_by_id = labels_df.set_index("image_id")["class"]

    y_train_names = labels_by_id.loc[train_ids].to_numpy()
    y_validation_names = labels_by_id.loc[validation_ids].to_numpy()

    y_train = np.array(
        [class_to_index[name] for name in y_train_names],
        dtype=np.int64,
    )

    y_validation = np.array(
        [class_to_index[name] for name in y_validation_names],
        dtype=np.int64,
    )

    print("\nSplit sizes:")
    print("Training samples:", len(train_ids))
    print("Validation samples:", len(validation_ids))
    print("Provided test samples:", len(submission_df))

    print("\nTraining class counts:")
    print(pd.Series(y_train_names).value_counts().sort_index())

    print("\nValidation class counts:")
    print(pd.Series(y_validation_names).value_counts().sort_index())

    # --------------------------------------------------
    # 5. Load and preprocess training images
    # --------------------------------------------------

    print("\nLoading training images...")

    X_train = load_images_from_ids(train_ids, TRAIN_DIR)

    print("\nLoading validation images...")

    X_validation = load_images_from_ids(
        validation_ids,
        TRAIN_DIR,
    )

    # --------------------------------------------------
    # 6. Check the processed arrays
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("PROCESSED DATA CHECKS")
    print("=" * 60)

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)

    print("X_validation shape:", X_validation.shape)
    print("y_validation shape:", y_validation.shape)

    print("Image data type:", X_train.dtype)
    print("Label data type:", y_train.dtype)

    print("Minimum pixel value:", X_train.min())
    print("Maximum pixel value:", X_train.max())

    assert X_train.ndim == 4
    assert X_validation.ndim == 4

    assert X_train.shape[1:] == (
        IMAGE_SIZE[1],
        IMAGE_SIZE[0],
        3,
    )

    assert X_validation.shape[1:] == X_train.shape[1:]

    assert len(X_train) == len(y_train)
    assert len(X_validation) == len(y_validation)

    assert np.isfinite(X_train).all()
    assert np.isfinite(X_validation).all()

    assert X_train.min() >= 0.0
    assert X_train.max() <= 1.0

    # --------------------------------------------------
    # 7. Optional: preprocess test images separately
    # --------------------------------------------------

    # This prepares the test images for later prediction.
    # It does not use test labels or include test images
    # in training or validation.

    print("\nLoading provided test images...")

    test_ids = submission_df["image_id"].to_numpy()

    X_test = load_images_from_ids(
        test_ids,
        TEST_DIR,
    )

    print("\nX_test shape:", X_test.shape)
    print("Test data type:", X_test.dtype)

    # --------------------------------------------------
    # 8. Finish
    # --------------------------------------------------

    print("\nPreprocessing completed successfully.")
    print("Image size:", IMAGE_SIZE)
    print("Number of classes:", len(class_to_index))

    print(
        "\nNote: Arrays currently exist in memory only. "
        "No files have been saved."
    )


if __name__ == "__main__":
    main()
