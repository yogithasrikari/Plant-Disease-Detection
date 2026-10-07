from datasets import load_dataset
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import numpy as np
import os

# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 38

TRAIN_SIZE = 10000
VALIDATION_SIZE = 2000

EPOCHS = 3
RANDOM_STATE = 42

print("========================================")
print("PLANT DISEASE DETECTION")
print("FINAL MODEL TRAINING")
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
# GET TRAIN / TEST INDICES
# ============================================================

all_train_indices = [
    i for i, split in enumerate(dataset["split"])
    if split == "train"
]

test_indices = [
    i for i, split in enumerate(dataset["split"])
    if split == "test"
]

print("\nDataset:")
print("Available training images:", len(all_train_indices))
print("Test images:", len(test_indices))

# ============================================================
# GET TRAINING LABELS
# ============================================================

all_train_labels = np.array([
    dataset[i]["class_idx"]
    for i in all_train_indices
])

# ============================================================
# SELECT STRATIFIED SUBSET
# ============================================================

print("\nSelecting stratified training subset...")

selected_indices, _ = train_test_split(
    all_train_indices,
    train_size=TRAIN_SIZE + VALIDATION_SIZE,
    random_state=RANDOM_STATE,
    stratify=all_train_labels
)

selected_labels = np.array([
    dataset[i]["class_idx"]
    for i in selected_indices
])

# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

train_indices, validation_indices = train_test_split(
    selected_indices,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE,
    stratify=selected_labels
)

print("\nFinal training setup:")
print("Training images:", len(train_indices))
print("Validation images:", len(validation_indices))
print("Test images:", len(test_indices))

# ============================================================
# VERIFY CLASSES
# ============================================================

train_classes = len(set(
    dataset[i]["class_idx"]
    for i in train_indices
))

validation_classes = len(set(
    dataset[i]["class_idx"]
    for i in validation_indices
))

print("\nTraining classes:", train_classes)
print("Validation classes:", validation_classes)

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


def test_generator():

    for index in test_indices:

        image = np.array(
            dataset[index]["image"]
        )

        label = dataset[index]["class_idx"]

        yield image, label

# ============================================================
# CREATE DATASETS
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

test_dataset = test_dataset.map(
    preprocess_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

# ============================================================
# BATCH + PREFETCH
# ============================================================

train_dataset = (
    train_dataset
    .shuffle(
        1000,
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

test_dataset = (
    test_dataset
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

# Freeze pretrained feature extractor
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
# MODEL INFORMATION
# ============================================================

print("\nModel created successfully!")

print(
    "Total parameters:",
    model.count_params()
)

print(
    "Trainable parameters:",
    sum(
        np.prod(variable.shape)
        for variable in model.trainable_variables
    )
)

# ============================================================
# CREATE MODELS DIRECTORY
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

# ============================================================
# CALLBACKS
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_accuracy",
    patience=1,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    "models/plant_disease_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

# ============================================================
# TRAIN
# ============================================================

print("\n========================================")
print("STARTING FINAL MODEL TRAINING")
print("========================================")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=[
        early_stopping,
        checkpoint
    ]
)

# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(
    "models/plant_disease_model_final.keras"
)

print("\n========================================")
print("TRAINING COMPLETED")
print("========================================")

print(
    "\nFinal model saved at:"
)

print(
    "models/plant_disease_model_final.keras"
)

# ============================================================
# RESULTS
# ============================================================

best_train_accuracy = max(
    history.history["accuracy"]
)

best_validation_accuracy = max(
    history.history["val_accuracy"]
)

print(
    "\nBest training accuracy:",
    f"{best_train_accuracy:.4f}"
)

print(
    "Best validation accuracy:",
    f"{best_validation_accuracy:.4f}"
)

print("\n========================================")
print("MODEL READY FOR EVALUATION")
print("========================================")