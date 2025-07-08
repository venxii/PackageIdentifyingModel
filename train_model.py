print("👋 Starting training script...")


import tensorflow as tf # type: ignore
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D # type: ignore
from tensorflow.keras.models import Model # type: ignore
import os

print("✅ TensorFlow version:", tf.__version__)
print("📂 Current working directory:", os.getcwd())
print("📁 Contents of ./data:", os.listdir("data"))
print("📁 Contents of ./data/train:", os.listdir("data/train"))

# ==== CONFIG =====
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10  # You can increase if needed
DATA_DIR = "data"  # change if your folder is elsewhere
MODEL_NAME = "packaging_classifier.h5"

# ==== LOAD DATA =====
train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "train"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "val"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print("Class labels:", class_names)

# ==== BUILD MODEL =====
base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze base model

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
outputs = Dense(len(class_names), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ==== TRAIN =====
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# ==== SAVE MODEL =====
model.save(MODEL_NAME)
print(f"✅ Model saved as {MODEL_NAME}")
