from datasets import load_dataset
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 64

MODEL_PATH = "models/plant_disease_model.keras"

# ============================================================
# HEADER
# ============================================================

print("========================================")
print("CONFUSION MATRIX & CLASSIFICATION REPORT")
print("========================================")

# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading trained model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully!")

# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading PlantVillage dataset...")

dataset = load_dataset(
    "geraldmc/plantvillage-full",
    revision="v0.1.0",
    split="train"
)

print("Dataset loaded successfully!")

# ============================================================
# GET TEST DATA
# ============================================================

test_indices = [
    i
    for i, split in enumerate(dataset["split"])
    if split == "test"
]

print("\nTest images:", len(test_indices))

# ============================================================
# CREATE CLASS NAME MAPPING
# ============================================================

print("\nCreating class name mapping...")

class_mapping = {}

for class_idx, class_label in zip(
    dataset["class_idx"],
    dataset["class_label"]
):
    class_mapping[class_idx] = class_label

class_names = [
    class_mapping[i]
    for i in range(len(class_mapping))
]

print(
    "Number of classes:",
    len(class_names)
)

print("\nClass mapping:")

for i, name in enumerate(class_names):
    print(f"{i}: {name}")

# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_images(images):

    processed_images = []

    for image in images:

        image = tf.image.resize(
            image,
            IMAGE_SIZE
        )

        image = tf.cast(
            image,
            tf.float32
        )

        image = tf.keras.applications.mobilenet_v2.preprocess_input(
            image
        )

        processed_images.append(
            image.numpy()
        )

    return np.array(processed_images)


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

print("\n========================================")
print("GENERATING PREDICTIONS")
print("========================================")

print(
    "\nThis may take some time because "
    "the test set contains 10,948 images."
)

y_true = []
y_pred = []

total_images = len(test_indices)

for start in range(
    0,
    total_images,
    BATCH_SIZE
):

    end = min(
        start + BATCH_SIZE,
        total_images
    )

    batch_indices = test_indices[start:end]

    batch_images = []
    batch_labels = []

    for index in batch_indices:

        batch_images.append(
            np.array(
                dataset[index]["image"]
            )
        )

        batch_labels.append(
            dataset[index]["class_idx"]
        )

    batch_images = preprocess_images(
        batch_images
    )

    predictions = model.predict(
        batch_images,
        verbose=0
    )

    predicted_labels = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(
        batch_labels
    )

    y_pred.extend(
        predicted_labels
    )

    processed = end

    if (
        processed % 640 == 0
        or processed == total_images
    ):

        percentage = (
            processed / total_images
        ) * 100

        print(
            f"Processed: {processed}/{total_images} "
            f"({percentage:.1f}%)"
        )

# Convert to NumPy arrays

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print("\nPrediction generation completed!")

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n========================================")
print("CLASSIFICATION REPORT")
print("========================================")

report = classification_report(
    y_true,
    y_pred,
    labels=np.arange(len(class_names)),
    target_names=class_names,
    digits=4
)

print(report)

# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

with open(
    "models/classification_report.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "PLANT DISEASE CLASSIFICATION REPORT\n"
    )

    file.write(
        "====================================\n\n"
    )

    file.write(report)

print(
    "\nClassification report saved to:"
)

print(
    "models/classification_report.txt"
)

# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\nGenerating confusion matrix...")

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=np.arange(len(class_names))
)

# ============================================================
# PLOT CONFUSION MATRIX
# ============================================================

plt.figure(
    figsize=(18, 16)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "Plant Disease Detection - Confusion Matrix"
)

plt.colorbar()

tick_marks = np.arange(
    len(class_names)
)

plt.xticks(
    tick_marks,
    class_names,
    rotation=90,
    fontsize=7
)

plt.yticks(
    tick_marks,
    class_names,
    fontsize=7
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)

plt.tight_layout()

# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

plt.savefig(
    "models/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

print(
    "\nConfusion matrix saved to:"
)

print(
    "models/confusion_matrix.png"
)

plt.show()

# ============================================================
# FINAL SUMMARY
# ============================================================

accuracy = np.mean(
    y_true == y_pred
)

print("\n========================================")
print("FINAL SUMMARY")
print("========================================")

print(
    f"\nTest Accuracy: {accuracy * 100:.2f}%"
)

print(
    "\nTotal test images:",
    len(y_true)
)

print(
    "Correct predictions:",
    np.sum(y_true == y_pred)
)

print(
    "Incorrect predictions:",
    np.sum(y_true != y_pred)
)

print("\n========================================")
print("ANALYSIS COMPLETED")
print("========================================")