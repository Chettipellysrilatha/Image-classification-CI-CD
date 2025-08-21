import numpy as np
import tensorflow as tf

# Load test data
x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')

# Load saved model
model = tf.keras.models.load_model('mnist_model')

# Evaluate
loss, acc = model.evaluate(x_test, tf.keras.utils.to_categorical(y_test), verbose=0)
print(f"Test Accuracy: {acc:.4f}")
print(f"Test Loss: {loss:.4f}")
