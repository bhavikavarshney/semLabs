# This code actually works :)
# Neural networks
# Importing the essential modules in the hidden layer
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math, random

tf.random.set_seed(1000)

function_to_learn = lambda x: np.cos(x) + 0.1*np.random.randn(*x.shape)
layer_1_neurons = 10
NUM_points = 1000

# Train the parameters of the hidden layer
batch_size = 100
NUM_EPOCHS = 1500

all_x = np.float32(np.random.uniform(-2*math.pi, 2*math.pi, (1, NUM_points))).T
np.random.shuffle(all_x)
train_size = int(900)

# Train the first 700 points in the set
x_training = all_x[:train_size]
y_training = function_to_learn(x_training)

# Training the last 300 points in the given set
x_validation = all_x[train_size:]
y_validation = function_to_learn(x_validation)

plt.figure(1)
plt.scatter(x_training, y_training, c='blue', label='train')
plt.scatter(x_validation, y_validation, c='pink', label='validation')
plt.legend()
plt.show()

X = tf.constant(x_training, dtype=tf.float32)
Y = tf.constant(y_training, dtype=tf.float32)

# first layer
# Number of neurons = 10
w_h = tf.Variable(tf.random.uniform([1, layer_1_neurons], minval=-1, maxval=1, dtype=tf.float32))
b_h = tf.Variable(tf.zeros([1, layer_1_neurons], dtype=tf.float32))
h = tf.nn.sigmoid(tf.matmul(X, w_h) + b_h)

# output layer
# Number of neurons = 10
w_o = tf.Variable(tf.random.uniform([layer_1_neurons, 1], minval=-1, maxval=1, dtype=tf.float32))
b_o = tf.Variable(tf.zeros([1, 1], dtype=tf.float32))

# Training loop
errors = []
optimizer = tf.keras.optimizers.Adam()

for i in range(NUM_EPOCHS):
    with tf.GradientTape() as tape:
        h = tf.nn.sigmoid(tf.matmul(X, w_h) + b_h)
        model = tf.matmul(h, w_o) + b_o
        loss = tf.nn.l2_loss(model - Y)

    gradients = tape.gradient(loss, [w_h, b_h, w_o, b_o])
    optimizer.apply_gradients(zip(gradients, [w_h, b_h, w_o, b_o]))

    cost = tf.nn.l2_loss(model - Y)
    errors.append(cost)

    if i % 100 == 0:
        print("epoch %d, cost = %g" % (i, cost))

plt.plot(errors, label='MLP Function Approximation')
plt.xlabel('epochs')
plt.ylabel('cost')
plt.legend()
plt.show()