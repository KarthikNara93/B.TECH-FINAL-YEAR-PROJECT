import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Define dataset path
data_path = r"C:\Users\Karthik\Desktop\latestlib\trsined_model\data_set" # Change if needed
gesture_labels = ["A", "B", "C", "D", "E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","0","1","2","3","4","5","6","7","8","9","10"] # Modify based on your dataset
num_classes = len(gesture_labels)

X, y = [], []

# Step 1: Load each class data
print("🔄 Loading dataset...")
for idx, label in enumerate(gesture_labels):
    label_path = os.path.join(data_path, label)  # Ensure path is correct

    if not os.path.exists(label_path):
        print(f"⚠️ Warning: Folder '{label}' not found! Skipping...")
        continue

    for file in os.listdir(label_path):
        if file.endswith(".npy"):
            full_path = os.path.join(label_path, file)
            try:
                data = np.load(full_path)
                X.append(data)
                y.append(idx)
            except Exception as e:
                print(f"❌ Error loading {file}: {e}")

# Step 2: Convert lists to numpy arrays
if len(X) == 0:
    print("❌ No data found! Check dataset path and file structure.")
    exit()

X = np.array(X, dtype=np.float32)
y = np.array(y)

print(f"✅ Loaded {len(X)} samples across {num_classes} classes.")

# Step 3: Normalize data
X = X / np.max(X)  # Scale data to range [0,1]
print("✅ Data normalized.")

# Step 4: One-hot encode labels
y = tf.one_hot(y, num_classes).numpy()
print("✅ Labels one-hot encoded.")

# Step 5: Split into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✅ Data split: {X_train.shape[0]} training, {X_test.shape[0]} testing.")

# Step 6: Save preprocessed data
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

print("🎉 Preprocessing completed! Saved as X_train.npy, X_test.npy, y_train.npy, y_test.npy.")