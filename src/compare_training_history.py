
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# SETTINGS
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "models"

CNN_HISTORY = MODEL_DIR / "training_history.csv"
RESNET_HISTORY = MODEL_DIR / "resnet18_history.csv"

OUTPUT_DIR = MODEL_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# LOAD TRAINING HISTORIES
# ==========================================

def load_history(path, model_name):

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}\n"
            "Check the filename in the settings section."
        )

    df = pd.read_csv(path)

    required_columns = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "validation_loss",
        "validation_accuracy"
    ]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{model_name} history is missing columns: {missing}"
        )

    print(f"\n{model_name} history loaded: {path}")
    print("Epochs recorded:", len(df))

    return df


cnn = load_history(CNN_HISTORY, "Custom CNN")
resnet = load_history(RESNET_HISTORY, "ResNet18")


# ==========================================
# PLOT LOSS CURVES
# ==========================================

plt.figure(figsize=(10, 6))

plt.plot(
    cnn["epoch"],
    cnn["train_loss"],
    label="Custom CNN - Training"
)
plt.plot(
    cnn["epoch"],
    cnn["validation_loss"],
    label="Custom CNN - Validation"
)

plt.plot(
    resnet["epoch"],
    resnet["train_loss"],
    label="ResNet18 - Training"
)
plt.plot(
    resnet["epoch"],
    resnet["validation_loss"],
    label="ResNet18 - Validation"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Comparison")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

loss_path = OUTPUT_DIR / "model_comparison_loss.png"
plt.savefig(loss_path, dpi=200)
plt.close()


# ==========================================
# PLOT ACCURACY CURVES
# ==========================================

plt.figure(figsize=(10, 6))

plt.plot(
    cnn["epoch"],
    cnn["train_accuracy"],
    label="Custom CNN - Training"
)
plt.plot(
    cnn["epoch"],
    cnn["validation_accuracy"],
    label="Custom CNN - Validation"
)

plt.plot(
    resnet["epoch"],
    resnet["train_accuracy"],
    label="ResNet18 - Training"
)
plt.plot(
    resnet["epoch"],
    resnet["validation_accuracy"],
    label="ResNet18 - Validation"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy Comparison")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

accuracy_path = OUTPUT_DIR / "model_comparison_accuracy.png"
plt.savefig(accuracy_path, dpi=200)
plt.close()


# ==========================================
# SUMMARISE BEST VALIDATION METRICS
# ==========================================

def summarize_model(df, model_name):

    # Best epoch based on lowest validation loss.
    best_loss_index = df["validation_loss"].idxmin()
    best_loss_row = df.loc[best_loss_index]

    # Also record the highest validation accuracy.
    best_accuracy_index = df["validation_accuracy"].idxmax()
    best_accuracy_row = df.loc[best_accuracy_index]

    return {
        "model": model_name,
        "epochs_recorded": len(df),
        "best_loss_epoch": int(best_loss_row["epoch"]),
        "lowest_validation_loss": float(
            best_loss_row["validation_loss"]
        ),
        "accuracy_at_best_loss": float(
            best_loss_row["validation_accuracy"]
        ),
        "best_accuracy_epoch": int(
            best_accuracy_row["epoch"]
        ),
        "highest_validation_accuracy": float(
            best_accuracy_row["validation_accuracy"]
        )
    }


summary = pd.DataFrame([
    summarize_model(cnn, "Custom CNN"),
    summarize_model(resnet, "ResNet18")
])

summary_path = OUTPUT_DIR / "model_comparison_summary.csv"
summary.to_csv(summary_path, index=False)

print("\nMODEL COMPARISON SUMMARY")
print("=" * 65)
print(summary.to_string(index=False))

print("\nSaved files:")
print(loss_path)
print(accuracy_path)
print(summary_path)
