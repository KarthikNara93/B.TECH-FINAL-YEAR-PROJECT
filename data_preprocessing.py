import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

data_dir = r'C:\Users\Karthik\Desktop\project_testing_2\dataset'
dataset_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]

X = []
y = []

# Loop through each dataset file (one per class)
for file in dataset_files:
    data_path = os.path.join(data_dir, file)
    class_data = np.load(data_path, allow_pickle=True)
    
    # Extract landmarks (X) and corresponding labels (y) for each class
    for item in class_data:
        X.append(item[0])  # Landmark data
        y.append(file.split('.')[0])  # Class label is the filename (e.g., '0', '1', '2', etc.)

X = np.array(X)
y = np.array(y)

print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")

lb = LabelBinarizer()
y_encoded = lb.fit_transform(y)

print(f"Shape of one-hot encoded y: {y_encoded.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y
)

np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

print("\nTraining Summary:")
print(f"Number of training samples: {X_train.shape[0]}")
print(f"Number of test samples: {X_test.shape[0]}")
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}")

print("\nData preprocessing completed and saved as 'X_train.npy', 'X_test.npy', 'y_train.npy', 'y_test.npy'!")
