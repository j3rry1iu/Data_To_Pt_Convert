import numpy as np
import torch
import re

# === Step 1: Load the raw text file ===
with open("accelerometerData.txt", "r") as f:
    raw_text = f.read()

# === Step 2: Clean and extract numbers ===
# Remove ellipses (which break parsing)
cleaned_text = raw_text.replace("...", "")

# Match all valid numeric patterns: int, float, scientific notation
numbers = re.findall(r'-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?', cleaned_text)

# Convert string numbers to floats
data_flat = np.array([float(num) for num in numbers], dtype=np.float32)

# === Step 3: Reshape ===
# You can configure this based on your data structure
num_samples = 3
sequence_length = len(data_flat) // (num_samples * 3)

try:
    data_np = data_flat.reshape(num_samples, sequence_length, 3)
except ValueError as e:
    print("❌ Reshape failed:", e)
    exit(1)

# === Step 4: Save as .pt ===
data_tensor = torch.tensor(data_np)
torch.save(data_tensor, "accelerometer_data.pt")

print("Data saved to 'accelerometer_data.pt'")
print("Shape:", data_tensor.shape)