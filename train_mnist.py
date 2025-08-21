import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# Reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# One-hot encode labels
y_train_ohe = to_categorical(y_train, 10)
y_test_ohe = to_categorical(y_test, 10)

# Build a simple neural network
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(">>> Starting training...")
model.fit(x_train, y_train_ohe, epochs=5, validation_data=(x_test, y_test_ohe), verbose=2)

# Save model using TensorFlow SavedModel format
out_dir = 'mnist_model'
model.save(out_dir)
print(f">>> Model saved to: {out_dir}")

# Save test data for evaluation elsewhere
np.save('x_test.npy', x_test)
np.save('y_test.npy', y_test)
print(">>> Test data saved: x_test.npy, y_test.npy")
