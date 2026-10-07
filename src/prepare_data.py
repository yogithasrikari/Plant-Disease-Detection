from datasets import load_dataset
from collections import Counter


# Load PlantVillage dataset
print("Loading PlantVillage dataset...")

dataset = load_dataset(
    "geraldmc/plantvillage-full",
    revision="v0.1.0",
    split="train"
)


# Basic dataset information
print("\nDataset loaded successfully!")
print("Total images:", len(dataset))


# Count the number of images in each class
class_counts = Counter(dataset["class_label"])


# Display number of classes
print("\nNumber of classes:", len(class_counts))


# Display images per class
print("\nImages per class:")

for class_name, count in sorted(class_counts.items()):
    print(f"{class_name}: {count}")