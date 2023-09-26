# ControlIA
Codigos del capitulo 4
Import dependencias
# TensorFlow is an open source machine learning library
!pip install tensorflow==2.11.1 
import tensorflow as tf
# NumPy is a math library
import numpy as np
# Matplotib is a graphing library
import matplotlib.pyplot as plt
# math is python's math library
import math

# Grafica Senoidal
#We'll generate this many sample datapoints
samples = 1000
# Set a "seed" value, so we get the same random numbers each time we run this
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
