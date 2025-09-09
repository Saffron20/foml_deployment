import tensorflow as tf
import numpy as np

# Generate some simple training data (y = 2x + 1)
X = np.array([-1.0, 0.0, 1.0, 2.0, 3.0], dtype=float)
y = np.array([-1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# Define a simple Sequential model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# Compile the model
model.compile(optimizer='sgd', loss='mean_squared_error')

# Train the model
print("Training the model...")
model.fit(X, y, epochs=500, verbose=0)

# Save the model
model.save("model.h5")
print("Model saved as model.h5")
