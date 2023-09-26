# TensorFlow is an open source machine learning library
!pip install tensorflow==2.11.1 
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# We'll use Keras to create a simple model architecture
# from tf.keras import layers #
from tensorflow.keras import layers
model_1 = tf.keras.Sequential()

# First layer takes a scalar input and feeds it through 16 "neurons." The
# neurons decide whether to activate based on the 'relu' activation function.
model_1.add(layers.Dense(16, activation='relu', input_shape=(1,)))

# First layer is a single neuron, since we want to output a single value
model_1.add(layers.Dense(1))

# Compile the model using a standard optimizer and loss function for regression
model_1.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

# Print a summary of the model's architecture
model_1.summary()
