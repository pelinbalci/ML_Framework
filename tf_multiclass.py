# Week2 assignment

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import linear, relu, sigmoid
import matplotlib.pyplot as plt


def load_data():
    X = np.load("Data/X.npy")
    y = np.load("Data/y.npy")
    X = X[0:1000]
    y = y[0:1000]
    return X, y


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def my_softmax(z):
    """ Softmax converts a vector of values to a probability distribution.
    Args:
      z (ndarray (N,))  : input data, N features
    Returns:
      a (ndarray (N,))  : softmax of z
    """
    ### START CODE HERE ###
    a =np.zeros_like(z)
    summary = 0
    for i in z:
        summary += np.exp(i)
    for i in range(len(z)):
        a_i = np.exp(z[i])/summary
        a[i] = a_i

    ### END CODE HERE ###
    return a


# load dataset
X, y = load_data()

tf.random.set_seed(1234) # for consistent results
model = Sequential(
    [
        ### START CODE HERE ###
        tf.keras.layers.Dense(25, input_shape=(400,), activation='relu'),
        tf.keras.layers.Dense(15, activation='relu'),
        tf.keras.layers.Dense(10, activation='linear')

        ### END CODE HERE ###
    ], name = "my_model"
)

model.summary()

[layer1, layer2, layer3] = model.layers

#### Examine Weights shapes
W1,b1 = layer1.get_weights()
W2,b2 = layer2.get_weights()
W3,b3 = layer3.get_weights()
print(f"W1 shape = {W1.shape}, b1 shape = {b1.shape}")
print(f"W2 shape = {W2.shape}, b2 shape = {b2.shape}")
print(f"W3 shape = {W3.shape}, b3 shape = {b3.shape}")


# defines a loss function, SparseCategoricalCrossentropy and indicates the softmax should be included with the
# loss calculation by adding from_logits=True)
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
)

history = model.fit(
    X,y,
    epochs=40
)

image_of_two = X[1015]
prediction = model.predict(image_of_two.reshape(1,400))  # prediction
print(f" predicting a Two: \n{prediction}")
print(f" Largest Prediction index: {np.argmax(prediction)}")

prediction_p = tf.nn.softmax(prediction)
print(f" predicting a Two. Probability vector: \n{prediction_p}")
print(f"Total of predictions: {np.sum(prediction_p):0.3f}")
yhat = np.argmax(prediction_p)
print(f"np.argmax(prediction_p): {yhat}")