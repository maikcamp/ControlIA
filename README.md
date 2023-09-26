# ControlIA
Codigos del capitulo 4
Import dependencias
#TensorFlow is an open source machine learning library
!pip install tensorflow==2.11.1 
import tensorflow as tf
#NumPy is a math library
import numpy as np
#Matplotib is a graphing library
import matplotlib.pyplot as plt
#math is python's math library
import math

# Grafica Senoidal
#We'll generate this many sample datapoints
samples = 1000
#Set a "seed" value, so we get the same random numbers each time we run this
#noteboo. ANy number can be user here
SEED=1337
np.random.seed(SEED)
tf.random.set_seed(SEED)
#Generate a uniformly distributed set of random numbers in the range from
# 0 to 2n, which covers a complete sine wave oscillation
x_values = np.random.uniform(low=0, high=2*math.pi, size=samples)
#Shuffle the values to guarantee they're not in order
np.random.shuffle(x_values)
#Calcule the corresponding sine values
y_values=np.sin(x_values)
#Plot our data. The 'b.' argument tells the library to print blue dots.
plt.plot(x_values, y_values,'b.')
plt.show()

# Add a small random number to each y value
y_values += 0.1 * np.random.rand(*y_values.shape)
# Plot out data
plt.plot(x_values, y_values, 'b.')
plt.show() 

# Different color
# We'll use 60% of our data for training and 20% for testing. The remaining 20%
# will be used for validation. Calculate the indices of each section.
TRAIN_SPLIT = int(0.6* samples)
TEST_SPLIT = int(0.2* samples + TRAIN_SPLIT)

#Use np.split to chop out data into three parts.
#the second argument to no.split is an array of indices where the data will be
#split. We provide two indices, so the data will be divided into three chunks
x_train, x_validate, x_test = np.split(x_values,[TRAIN_SPLIT, TEST_SPLIT])
y_train, y_validate, y_test = np.split(y_values,[TRAIN_SPLIT, TEST_SPLIT])
#Double check that our splits add up correctly
assert (x_train.size + x_validate.size + x_test.size) == samples

# PLOT the data in each partition in differents colors
plt.plot(x_train, y_train, 'b.', label="Train")
plt.plot(x_validate, y_validate, 'y.', label="Validate")
plt.plot(x_test, y_test, 'r.', label="Test")
plt.legend()
plt.show()
