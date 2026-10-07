from datasets import load_dataset
import tensorflow as tf
import numpy as np

# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 64

MODEL_PATH = "models/plant_disease_model.keras"

print("========================================")
print("PLANT DISEASE MODEL EVALUATION")
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
# GET TEST INDICES
# ============================================================

test_indices = [
    i for i, split in enumerate(dataset["split"])
    if split == "test"
]

print(
    "\nTest images:",
    len(test_indices)
)

# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_image(image, label):

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    # MobileNetV2 preprocessing
    image = tf.keras.applications.mobilenet_v2.preprocess_input(
        image
    )

    return image, label

# ============================================================
# TEST GENERATOR
# ============================================================

def test_generator():

    for index in test_indices:

        image = np.array(
            dataset[index]["image"]
        )

        label = dataset[index]["class_idx"]

        yield image, label

# ============================================================
# CREATE TEST DATASET
# ============================================================

test_dataset = tf.data.Dataset.from_generator(
    test_generator,
    output_signature=(
        tf.TensorSpec(
            shape=(None, None, 3),
            dtype=tf.uint8
        ),
        tf.TensorSpec(
            shape=(),
            dtype=tf.int32
        )
    )
)

# ============================================================
# PREPROCESS TEST DATA
# ============================================================

test_dataset = test_dataset.map(
    preprocess_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

# ============================================================
# BATCH + PREFETCH
# ============================================================

test_dataset = (
    test_dataset
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

print("\nTest data pipeline ready!")

# ============================================================
# EVALUATE MODEL
# ============================================================

print("\n========================================")
print("EVALUATING MODEL")
print("========================================")

results = model.evaluate(
    test_dataset,
    verbose=1
)

# ============================================================
# DISPLAY RESULTS
# ============================================================

test_loss = results[0]
test_accuracy = results[1]

print("\n========================================")
print("FINAL TEST RESULTS")
print("========================================")

print(
    "\nTest loss:",
    f"{test_loss:.4f}"
)

print(
    "Test accuracy:",
    f"{test_accuracy:.4f}"
)

print(
    "\nTest accuracy percentage:",
    f"{test_accuracy * 100:.2f}%"
)

print("\n========================================")
print("EVALUATION COMPLETED")
print("========================================")