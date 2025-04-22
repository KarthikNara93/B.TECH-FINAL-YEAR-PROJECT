import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load preprocessed data
X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

# Reshape input for LSTM (Samples, Timesteps, Features)
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Define the CNN-LSTM Model
model = keras.Sequential([
    layers.Conv1D(64, kernel_size=1, activation='relu', input_shape=(1, X_train.shape[2])),
    layers.MaxPooling1D(pool_size=1),
    layers.LSTM(64, return_sequences=True),
    layers.LSTM(32),
    layers.Dense(32, activation='relu'),
    layers.Dense(y_train.shape[1], activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
epochs = 75
batch_size = 32
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

# Save the trained model
model.save("cnn-lstm-model2.h5")

print("Model training completed and saved as 'cnn-lstm-model.h5'!")
