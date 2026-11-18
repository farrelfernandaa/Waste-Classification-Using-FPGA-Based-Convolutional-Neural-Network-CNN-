"""
Skrip Training & Kuantisasi Tiny-CNN (INT8) untuk Implementasi FPGA Verilog
Dataset: TrashNet (garythung/trashnet) - 3 Kelas (Metal, Paper, Plastic)
Input: 32x32 Grayscale
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from datasets import load_dataset
from PIL import Image

# 1. Memuat Dataset TrashNet dari Hugging Face
print("Mengunduh dataset TrashNet...")
dataset = load_dataset("garythung/trashnet", split="train")

# Filter 3 kelas utama: 2=metal, 3=paper, 4=plastic
TARGET_CLASSES = {4: 0, 3: 1, 2: 2}  # 0: Plastic, 1: Paper, 2: Metal
CLASS_NAMES = ["Plastic", "Paper", "Metal"]

X_data, y_data = [], []
for item in dataset:
    label = item["label"]
    if label in TARGET_CLASSES:
        img = item["image"].convert("L").resize((32, 32))
        arr = np.array(img, dtype=np.float32) / 128.0 - 1.0  # Normalisasi [-1, 1]
        X_data.append(arr[..., np.newaxis])
        y_data.append(TARGET_CLASSES[label])

X_data = np.array(X_data)
y_data = np.array(y_data)

# Acak dan bagi data train/val (80:20)
indices = np.arange(len(X_data))
np.random.seed(42)
np.random.shuffle(indices)
X_data, y_data = X_data[indices], y_data[indices]

split = int(0.8 * len(X_data))
X_train, X_val = X_data[:split], X_data[split:]
y_train, y_val = y_data[:split], y_data[split:]

print(f"Total sampel (3 kelas): {len(X_data)} | Train: {len(X_train)} | Val: {len(X_val)}")

# 2. Arsitektur Tiny-CNN Ramah Hardware (Total 1.203 Parameter)
model = models.Sequential([
    layers.Input(shape=(32, 32, 1)),
    layers.Conv2D(4, (3, 3), activation="relu", name="conv1"),
    layers.MaxPooling2D((2, 2), name="pool1"),
    layers.Conv2D(8, (3, 3), activation="relu", name="conv2"),
    layers.MaxPooling2D((2, 2), name="pool2"),
    layers.Flatten(),
    layers.Dense(3, name="fc_out")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

model.summary()

# 3. Training Model
print("\nMemulai training...")
model.fit(X_train, y_train, epochs=25, batch_size=32, validation_data=(X_val, y_val), verbose=1)

val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"\nAkurasi Validasi (FP32): {val_acc * 100:.2f}%")

# 4. Kuantisasi Bobot ke 8-bit Signed Integer (INT8) & Ekspor ke .mem
def export_weights_to_mem(keras_model, filename="weights.mem"):
    all_quantized = []
    for layer in keras_model.layers:
        weights = layer.get_weights()
        if len(weights) > 0:
            for w in weights:
                w_flat = w.flatten()
                max_val = np.max(np.abs(w_flat)) + 1e-8
                scale = 127.0 / max_val
                w_int8 = np.clip(np.round(w_flat * scale), -128, 127).astype(np.int8)
                all_quantized.extend(w_int8)

    with open(filename, "w") as f:
        f.write("// Quantized INT8 Weights & Biases for FPGA ROM ($readmemh)\n")
        f.write(f"// Total Parameters: {len(all_quantized)} bytes\n")
        for val in all_quantized:
            hex_val = format(int(val) & 0xFF, "02x")
            f.write(f"{hex_val}\n")
    print(f"Berhasil mengekspor {len(all_quantized)} parameter INT8 ke '{filename}'")

export_weights_to_mem(model, "weights.mem")
