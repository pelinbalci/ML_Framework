import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def load_data():
    X = np.load("Data/X.npy")
    y = np.load("Data/y.npy")
    X = X[0:1000]
    y = y[0:1000]
    return X, y


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


###############
# LOAD DATA
###############
X, y = load_data()

###############
# CREATE MODEL
###############
model = Sequential(
    [
        tf.keras.Input(shape=(400,)),  # specify input size
        ### START CODE HERE ###
        tf.keras.layers.Dense(25, activation='sigmoid'),
        tf.keras.layers.Dense(15, activation='sigmoid'),
        tf.keras.layers.Dense(1, activation='sigmoid')

        ### END CODE HERE ###
    ], name="my_model"
)
model.summary()

###############
# NUMBER OF PARAMETERS
###############
L1_num_params = 400 * 25 + 25  # W1 parameters  + b1 parameters
L2_num_params = 25 * 15 + 15   # W2 parameters  + b2 parameters
L3_num_params = 15 * 1 + 1     # W3 parameters  + b3 parameters
print("L1 params = ", L1_num_params, ", L2 params = ", L2_num_params, ",  L3 params = ", L3_num_params )

###############
# GET WEIGHTS
###############

[layer1, layer2, layer3] = model.layers

# Examine Weights shapes
W1,b1 = layer1.get_weights()
W2,b2 = layer2.get_weights()
W3,b3 = layer3.get_weights()
print(f"W1 shape = {W1.shape}, b1 shape = {b1.shape}")
print(f"W2 shape = {W2.shape}, b2 shape = {b2.shape}")
print(f"W3 shape = {W3.shape}, b3 shape = {b3.shape}")

w_x, b_x = model.layers[1].get_weights()

###############
# COMPILE - LOSS AND OPTIMIZATION
###############
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.001),
)

###############
# TRAIN
###############
model.fit(X,y,epochs=20)

###############
# PREDICT
###############
prediction = model.predict(X[0].reshape(1,400))
if prediction >= 0.5:
    yhat = 1
else:
    yhat = 0


def my_dense(a_in, W, b, g):
    """
    Computes dense layer
    Args:
      a_in (ndarray (n, )) : Data, 1 example
      W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
      b    (ndarray (j, )) : bias vector, j units
      g    activation function (e.g. sigmoid, relu..)
    Returns
      a_out (ndarray (j,))  : j units
    """
    units = W.shape[1]
    a_out = np.zeros(units)
    ### START CODE HERE ###
    for j in range(units):
        w = W[:, j]  # take the all weights to the jth neuron in the layer.
        z = np.dot(w, a_in) + b[j]  # all inputs times all weights to jth neuron + bias for jth neuron.
        a_out[j] = g(z)  # activation function

    ### END CODE HERE ###
    return (a_out)


def my_dense_v(A_in, W, b, g):
    """
    Computes dense layer
    Args:
      A_in (ndarray (m,n)) : Data, m examples, n features each
      W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
      b    (ndarray (1,j)) : bias vector, j units
      g    activation function (e.g. sigmoid, relu..)
    Returns
      A_out (tf.Tensor or ndarray (m,j)) : m examples, j units
    """
    ### START CODE HERE ###
    z = np.matmul(A_in, W) + b
    A_out = g(z)

    ### END CODE HERE ###
    return (A_out)


def my_sequential(x, W1, b1, W2, b2, W3, b3):
    a1 = my_dense(x,  W1, b1, sigmoid)
    a2 = my_dense(a1, W2, b2, sigmoid)
    a3 = my_dense(a2, W3, b3, sigmoid)
    return(a3)


###############
# GET WEIGHTS AND DEFINE THEM IN MY SEQUENTIAL
###############
W1_tmp,b1_tmp = layer1.get_weights()
W2_tmp,b2_tmp = layer2.get_weights()
W3_tmp,b3_tmp = layer3.get_weights()


prediction = my_sequential(X[0], W1_tmp, b1_tmp, W2_tmp, b2_tmp, W3_tmp, b3_tmp )
if prediction >= 0.5:
    yhat = 1
else:
    yhat = 0
print( "yhat = ", yhat, " label= ", y[0,0])


##############
# GET NAME TO LAYERS
#############

model = Sequential(
    [
        tf.keras.layers.Dense(3, input_dim=2,  activation = 'sigmoid', name='L1'),
        tf.keras.layers.Dense(1,  activation = 'sigmoid', name='L2')
    ]
)

model.summary()
logistic_layer = model.get_layer('L1')
w,b = logistic_layer.get_weights()
print('w:', w, 'b:', b)
