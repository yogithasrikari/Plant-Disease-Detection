from datasets import load_dataset
from collections import Counter


# ============================================================
# Load Dataset
# ============================================================

print("Loading PlantVillage dataset...")

dataset = load_dataset(
    "geraldmc/plantvillage-full",
    revision="v0.1.0",
    split="train"
)

print("\nDataset loaded successfully!")
print("Total images:", len(dataset))


# ============================================================
# Check Split Values
# ============================================================

print("\nChecking split values...")

split_counts = Counter(dataset["split"])

for split_name, count in split_counts.items():
    print(f"{split_name}: {count}")


# ============================================================
# Check Class Information
# ============================================================

print("\nChecking class information...")

class_labels = dataset["class_label"]
class_indices = dataset["class_idx"]


print("Number of unique class names:",
      len(set(class_labels)))

print("Number of unique class indices:",
      len(set(class_indices)))


# ============================================================
# Check Class Name -> Index Mapping
# ============================================================

print("\nClass name -> class index mapping:")

mapping = {}

for name, index in zip(class_labels, class_indices):

    if name not in mapping:
        mapping[name] = index

for name, index in sorted(mapping.items()):
    print(f"{index:2d} -> {name}")


# ============================================================
# Check Train/Test Class Distribution
# ============================================================

print("\nChecking train/test class distribution...")

train_counts = Counter()
test_counts = Counter()

for i in range(len(dataset)):

    class_name = dataset[i]["class_label"]
    split = dataset[i]["split"]

    if split == "train":
        train_counts[class_name] += 1

    elif split == "test":
        test_counts[class_name] += 1


# ============================================================
# Display Distribution
# ============================================================

print("\nTRAINING CLASS COUNTS:")

for class_name, count in sorted(train_counts.items()):
    print(f"{class_name}: {count}")


print("\nTESTING CLASS COUNTS:")

for class_name, count in sorted(test_counts.items()):
    print(f"{class_name}: {count}")


# ============================================================
# Final Checks
# ============================================================

print("\n========================================")
print("DATASET DIAGNOSTIC SUMMARY")
print("========================================")

print("Total images:", len(dataset))
print("Training images:", sum(train_counts.values()))
print("Testing images:", sum(test_counts.values()))
print("Number of classes:", len(mapping))

print("\nDataset diagnostic completed successfully!")