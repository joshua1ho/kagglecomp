
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


# ==================================================
# 1. PROJECT PATHS AND SETTINGS
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

LABELS_FILE = DATA_DIR / "train_labels.csv"
SUBMISSION_FILE = DATA_DIR / "sample_submission.csv"

OUTPUT_DIR = DATA_DIR / "processed"

# Image dimensions: width, height
IMAGE_SIZE = (112, 112)

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


# ==================================================
# 2. BUILD IMAGE INDEX
# ==================================================

def build_image_index(folder):
    """
    Find images once and create a dictionary mapping
    each image ID to its file path.
    """

    print(f"\nIndexing images in: {folder}", flush=True)

    if not folder.exists():
        raise FileNotFoundError(
            f"Image folder does not exist: {folder}"
        )

    image_index = {}

    supported_extensions = {".jpg", ".jpeg", ".png"}

    for path in folder.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() not in supported_extensions:
            continue

        image_id = path.stem

        if image_id in image_index:
            raise ValueError(
                f"Duplicate image ID {image_id} in {folder}"
            )

        image_index[image_id] = path

    print(
        f"Found {len(image_index)} images.",
        flush=True
    )

    return image_index


# ==================================================
# 3. LOAD AND PREPROCESS ONE IMAGE
# ==================================================

def load_image(image_path):
    """
    Open an image, convert it to RGB, resize it,
    and return uint8 pixel values from 0 to 255.
    """

    with Image.open(image_path) as image:

        image = image.convert("RGB")

        image = image.resize(
            IMAGE_SIZE,
            Image.Resampling.BILINEAR
        )

        image_array = np.asarray(
            image,
            dtype=np.uint8
        )

    return image_array


# ==================================================
# 4. PROCESS IMAGES DIRECTLY TO DISK
# ==================================================

def save_images(image_ids, image_index, output_path):
    """
    Process images one at a time and save them to
    a NumPy memory-mapped file.

    This avoids storing the complete image dataset
    in ordinary RAM.
    """

    image_ids = np.asarray(image_ids)
    total = len(image_ids)

    if total == 0:
        raise ValueError(
            f"No image IDs supplied for {output_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"\nProcessing {total} images...",
        flush=True
    )

    # uint8 uses 1 byte per colour channel.
    # open_memmap writes to disk rather than requiring
    # the complete array to fit in ordinary RAM.
    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.uint8,
        shape=(
            total,
            IMAGE_SIZE[1],
            IMAGE_SIZE[0],
            3
        )
    )

    try:

        for i, image_id in enumerate(image_ids):

            key = f"{int(image_id):05d}"

            if key not in image_index:
                raise FileNotFoundError(
                    f"Image {key} not found."
                )

            image_path = image_index[key]

            output[i] = load_image(image_path)

            if (
                (i + 1) % 1000 == 0
                or i + 1 == total
            ):
                print(
                    f"Processed {i + 1}/{total} images",
                    flush=True
                )

        output.flush()

    finally:
        del output

    print(
        f"Saved images to: {output_path}",
        flush=True
    )


# ==================================================
# 5. MAIN PREPROCESSING WORKFLOW
# ==================================================

def main():

    print("=" * 60)
    print("FACIAL FEATURE PREPROCESSING")
    print("=" * 60, flush=True)

    # --------------------------------------------------
    # Check required files and folders
    # --------------------------------------------------

    required_paths = [
        TRAIN_DIR,
        TEST_DIR,
        LABELS_FILE,
        SUBMISSION_FILE,
    ]

    for path in required_paths:

        if not path.exists():
            raise FileNotFoundError(
                f"Required path does not exist: {path}"
            )

    # --------------------------------------------------
    # Load and validate CSV files
    # --------------------------------------------------

    labels_df = pd.read_csv(LABELS_FILE)
    submission_df = pd.read_csv(SUBMISSION_FILE)

    if not {"image_id", "class"}.issubset(
        labels_df.columns
    ):
        raise ValueError(
            "Training CSV must contain image_id and class."
        )

    if "image_id" not in submission_df.columns:
        raise ValueError(
            "Sample submission must contain image_id."
        )

    if labels_df[
        ["image_id", "class"]
    ].isnull().any().any():

        raise ValueError(
            "Training labels contain missing values."
        )

    if labels_df["image_id"].duplicated().any():
        raise ValueError(
            "Duplicate training image IDs found."
        )

    if submission_df["image_id"].isnull().any():
        raise ValueError(
            "Test submission contains missing image IDs."
        )

    if submission_df["image_id"].duplicated().any():
        raise ValueError(
            "Duplicate test image IDs found."
        )

    actual_classes = set(
        labels_df["class"].unique()
    )

    if actual_classes != set(EXPECTED_CLASSES):
        raise ValueError(
            "Unexpected class labels found: "
            f"{sorted(actual_classes)}"
        )

    print("\nCSV validation passed.", flush=True)

    print(
        "Training label rows:",
        len(labels_df)
    )

    print(
        "Provided test rows:",
        len(submission_df)
    )

    # --------------------------------------------------
    # Create class mappings
    # --------------------------------------------------

    class_to_index = {
        class_name: index
        for index, class_name
        in enumerate(EXPECTED_CLASSES)
    }

    index_to_class = {
        index: class_name
        for class_name, index
        in class_to_index.items()
    }

    print("\nClass-to-index mapping:")

    for class_name, index in class_to_index.items():
        print(f"{class_name}: {index}")

    # --------------------------------------------------
    # Split training data
    # --------------------------------------------------

    train_ids, validation_ids = train_test_split(
        labels_df["image_id"].to_numpy(),
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels_df["class"]
    )

    # Preserve label-to-image correspondence.
    labels_by_id = labels_df.set_index(
        "image_id"
    )["class"]

    y_train_names = labels_by_id.loc[
        train_ids
    ].to_numpy()

    y_validation_names = labels_by_id.loc[
        validation_ids
    ].to_numpy()

    y_train = np.array(
        [
            class_to_index[name]
            for name in y_train_names
        ],
        dtype=np.int64
    )

    y_validation = np.array(
        [
            class_to_index[name]
            for name in y_validation_names
        ],
        dtype=np.int64
    )

    test_ids = submission_df[
        "image_id"
    ].to_numpy()

    print("\nSplit sizes:")
    print("Training:", len(train_ids))
    print("Validation:", len(validation_ids))
    print("Provided test:", len(test_ids))

    print("\nTraining class counts:")
    print(
        pd.Series(
            y_train_names
        ).value_counts().sort_index()
    )

    print("\nValidation class counts:")
    print(
        pd.Series(
            y_validation_names
        ).value_counts().sort_index()
    )

    # --------------------------------------------------
    # Index image files once
    # --------------------------------------------------

    train_index = build_image_index(TRAIN_DIR)
    test_index = build_image_index(TEST_DIR)

    # Verify all required image IDs exist.
    for ids, index, name in [
        (
            labels_df["image_id"].to_numpy(),
            train_index,
            "training"
        ),
        (
            test_ids,
            test_index,
            "test"
        )
    ]:

        missing = [
            f"{int(image_id):05d}"
            for image_id in ids
            if f"{int(image_id):05d}" not in index
        ]

        if missing:
            raise FileNotFoundError(
                f"Missing {name} images. "
                f"Examples: {missing[:10]}"
            )

    print(
        "\nAll requested image files were found.",
        flush=True
    )

    # --------------------------------------------------
    # Create output folder
    # --------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------
    # Process training images
    # --------------------------------------------------

    save_images(
        train_ids,
        train_index,
        OUTPUT_DIR / "X_train.npy"
    )

    # --------------------------------------------------
    # Process validation images
    # --------------------------------------------------

    save_images(
        validation_ids,
        train_index,
        OUTPUT_DIR / "X_validation.npy"
    )

    # --------------------------------------------------
    # Process provided test images separately
    # --------------------------------------------------

    save_images(
        test_ids,
        test_index,
        OUTPUT_DIR / "X_test.npy"
    )

    # --------------------------------------------------
    # Save labels, IDs and class mapping
    # --------------------------------------------------

    np.save(
        OUTPUT_DIR / "y_train.npy",
        y_train
    )

    np.save(
        OUTPUT_DIR / "y_validation.npy",
        y_validation
    )

    np.save(
        OUTPUT_DIR / "train_ids.npy",
        train_ids
    )

    np.save(
        OUTPUT_DIR / "validation_ids.npy",
        validation_ids
    )

    np.save(
        OUTPUT_DIR / "test_ids.npy",
        test_ids
    )

    np.save(
        OUTPUT_DIR / "class_names.npy",
        np.array(EXPECTED_CLASSES)
    )

    # --------------------------------------------------
    # Verify saved arrays without loading all data
    # into ordinary RAM
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("VERIFYING SAVED ARRAYS")
    print("=" * 60)

    for name in [
        "X_train",
        "X_validation",
        "X_test"
    ]:

        array = np.load(
            OUTPUT_DIR / f"{name}.npy",
            mmap_mode="r"
        )

        print(f"\n{name}:")
        print("Shape:", array.shape)
        print("Data type:", array.dtype)
        print(
            "Pixel range:",
            int(array.min()),
            "to",
            int(array.max())
        )

        expected_shape = (
            IMAGE_SIZE[1],
            IMAGE_SIZE[0],
            3
        )

        assert array.ndim == 4
        assert array.shape[1:] == expected_shape
        assert array.dtype == np.uint8

        del array

    print("\nLabel array checks:")

    print("y_train shape:", y_train.shape)
    print("y_validation shape:", y_validation.shape)

    assert len(y_train) == len(train_ids)
    assert len(y_validation) == len(validation_ids)

    assert y_train.min() >= 0
    assert y_train.max() < len(EXPECTED_CLASSES)

    assert y_validation.min() >= 0
    assert y_validation.max() < len(EXPECTED_CLASSES)

    print("\nPreprocessing completed successfully.")
    print("Saved files in:", OUTPUT_DIR)

    print(
        "\nImages are stored as uint8 pixels (0-255). "
        "Convert batches to float32 and divide by 255 "
        "when training the model."
    )


if __name__ == "__main__":
    main()
