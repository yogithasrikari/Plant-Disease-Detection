from datasets import load_dataset
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.model_selection import train_test_split
import numpy as np
import os

# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 38

# Small dataset for QUICK TESTING
TRAIN_SIZE = 2000
VALIDATION_SIZE = 400

EPOCHS = 1
RANDOM_STATE = 42

print("========================================")
print("QUICK PLANT DISEASE MODEL TEST")
print("========================================")

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
print("Total images:", len(dataset))

# ============================================================
# GET ORIGINAL TRAIN INDICES
# ============================================================

all_train_indices = [
    i for i, split in enumerate(dataset["split"])
    if split == "train"
]

print(
    "\nOriginal training images:",
    len(all_train_indices)
)

# ============================================================
# GET LABELS
# ============================================================

all_train_labels = np.array([
    dataset[i]["class_idx"]
    for i in all_train_indices
])

# ============================================================
# SELECT SMALL STRATIFIED SUBSET
# ============================================================

print("\nSelecting small stratified dataset...")

small_indices, _ = train_test_split(
    all_train_indices,
    train_size=TRAIN_SIZE + VALIDATION_SIZE,
    random_state=RANDOM_STATE,
    stratify=all_train_labels
)

small_labels = np.array([
    dataset[i]["class_idx"]
    for i in small_indices
])

# ============================================================
# SPLIT SMALL DATASET
# ============================================================

train_indices, validation_indices = train_test_split(
    small_indices,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE,
    stratify=small_labels
)

print("\nQuick training split:")
print(
    "Training images:",
    len(train_indices)
)

print(
    "Validation images:",
    len(validation_indices)
)

# ============================================================
# VERIFY CLASSES
# ============================================================

train_class_count = len(set(
    dataset[i]["class_idx"]
    for i in train_indices
))

validation_class_count = len(set(
    dataset[i]["class_idx"]
    for i in validation_indices
))

print(
    "\nClasses in training:",
    train_class_count
)

print(
    "Classes in validation:",
    validation_class_count
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
    image = preprocess_input(image)

    return image, label


# ============================================================
# GENERATORS
# ============================================================

def train_generator():

    for index in train_indices:

        image = np.array(
            dataset[index]["image"]
        )

        label = dataset[index]["class_idx"]

        yield image, label


def validation_generator():

    for index in validation_indices:

        image = np.array(
            dataset[index]["image"]
        )

        label = dataset[index]["class_idx"]

        yield image, label


# ============================================================
# CREATE TF.DATA DATASETS
# ============================================================

train_dataset = tf.data.Dataset.from_generator(
    train_generator,
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

validation_dataset = tf.data.Dataset.from_generator(
    validation_generator,
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
# PREPROCESS
# ============================================================

train_dataset = train_dataset.map(
    preprocess_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

validation_dataset = validation_dataset.map(
    preprocess_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

# ============================================================
# BATCH + PREFETCH
# ============================================================

train_dataset = (
    train_dataset
    .shuffle(
        500,
        reshuffle_each_iteration=True
    )
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

validation_dataset = (
    validation_dataset
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# ============================================================
# TEST PIPELINE
# ============================================================

print("\nTesting data pipeline...")

for images, labels in train_dataset.take(1):

    print(
        "Image batch shape:",
        images.shape
    )

    print(
        "Label batch shape:",
        labels.shape
    )

    print(
        "Pixel range:",
        float(tf.reduce_min(images)),
        "to",
        float(tf.reduce_max(images))
    )

print("\nData pipeline ready!")

# ============================================================
# BUILD MOBILENETV2
# ============================================================

print("\nLoading MobileNetV2...")

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers
base_model.trainable = False

model = models.Sequential([
    layers.Input(
        shape=(224, 224, 3)
    ),

    base_model,

    layers.GlobalAveragePooling2D(),

    layers.Dropout(0.3),

    layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])

# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)

# ============================================================
# MODEL SUMMARY
# ============================================================

print("\nModel created successfully!")

print(
    "Trainable parameters:",
    model.count_params()
)

# ============================================================
# TRAIN
# ============================================================

print("\n========================================")
print("STARTING QUICK TRAINING")
print("========================================")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)

# ============================================================
# RESULTS
# ============================================================

print("\n========================================")
print("QUICK TEST COMPLETED")
print("========================================")

train_accuracy = history.history["accuracy"][0]
validation_accuracy = history.history["val_accuracy"][0]

train_loss = history.history["loss"][0]
validation_loss = history.history["val_loss"][0]

print(
    "\nTraining accuracy:",
    f"{train_accuracy:.4f}"
)

print(
    "Training loss:",
    f"{train_loss:.4f}"
)

print(
    "Validation accuracy:",
    f"{validation_accuracy:.4f}"
)

print(
    "Validation loss:",
    f"{validation_loss:.4f}"
)

print("\n========================================")
print("END OF QUICK TEST")
print("========================================")