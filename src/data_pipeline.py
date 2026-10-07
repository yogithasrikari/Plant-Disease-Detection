from datasets import load_dataset
import tensorflow as tf
import numpy as np


# --------------------------------------------------
# Configuration
# --------------------------------------------------

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 38


# --------------------------------------------------
# Load PlantVillage dataset
# --------------------------------------------------

print("Loading PlantVillage dataset...")

dataset = load_dataset(
    "geraldmc/plantvillage-full",
    revision="v0.1.0",
    split="train"
)

print("Dataset loaded successfully!")
print("Total images:", len(dataset))


# --------------------------------------------------
# Get train and test indices
# --------------------------------------------------

train_indices = [
    i for i, split in enumerate(dataset["split"])
    if split == "train"
]

test_indices = [
    i for i, split in enumerate(dataset["split"])
    if split == "test"
]

print("\nTraining images:", len(train_indices))
print("Testing images:", len(test_indices))


# --------------------------------------------------
# Image preprocessing function
# --------------------------------------------------

def preprocess_image(image, label):
    """
    Resize image and normalize pixel values.
    """

    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0

    return image, label


# --------------------------------------------------
# Generator for TensorFlow
# --------------------------------------------------

def train_generator():
    for index in train_indices:
        image = np.array(dataset[index]["image"])
        label = dataset[index]["class_idx"]

        yield image, label


def test_generator():
    for index in test_indices:
        image = np.array(dataset[index]["image"])
        label = dataset[index]["class_idx"]

        yield image, label


# --------------------------------------------------
# Create TensorFlow datasets
# --------------------------------------------------

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


# --------------------------------------------------
# Apply preprocessing
# --------------------------------------------------

train_dataset = train_dataset.map(
    preprocess_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

test_dataset = test_dataset.map(
    preprocess_image,
    num_parallel_calls=tf.data.AUTOTUNE
)


# --------------------------------------------------
# Shuffle, batch and prefetch
# --------------------------------------------------

train_dataset = (
    train_dataset
    .shuffle(1000)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

test_dataset = (
    test_dataset
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)


# --------------------------------------------------
# Test the pipeline
# --------------------------------------------------

print("\nTesting data pipeline...")

for images, labels in train_dataset.take(1):

    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    print("Pixel value range:",
          float(tf.reduce_min(images)),
          "to",
          float(tf.reduce_max(images)))

print("\nData pipeline created successfully!")