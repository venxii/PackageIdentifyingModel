import tensorflow as tf # type: ignore
import os
import json

DATA_DIR = "data/train"
LABELS_FILE = "labels.json"

# Load just to get class names
ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    image_size=(224, 224),
    batch_size=1
)

# Extract and save
labels = ds.class_names
print("✅ Classes:", labels)

with open(LABELS_FILE, "w") as f:
    json.dump(labels, f)

print(f"✅ Saved labels to {LABELS_FILE}")
