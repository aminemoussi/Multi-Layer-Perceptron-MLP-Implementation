import numpy as np
import math
from scipy.special import softmax

import time



class MLP:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights_input_hidden = None
        self.bias_input_hidden = None
        self.weights_hidden_output = None
        self.bias_hidden_output = None



    def init_params(self, ni, nh, no):
        self.weights_input_hidden = np.random.normal(0, 0.3, (ni, nh))             #ask her about this
        self.bias_input_hidden = np.zeros((1, nh))

        self.weights_hidden_output = np.random.normal(0, 0.3, (nh, no))
        self.bias_hidden_output = np.zeros((1, no))

        print("** Initializing weights and biases **")
        print("Input -> Hidden Weights: ", self.weights_input_hidden, "\n")
        print("Weights: ", self.bias_input_hidden, "\n")

        time.sleep(3)

        print("Hidden -> Output Weights: ", self.weights_hidden_output, "\n")
        print("Weights: ", self.bias_hidden_output, "\n")




    def hyperbolic_tan(self, x):    #Hyperbolic Tang
        return np.tanh(x)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Subtract max for numerical stability
        return e_x / np.sum(e_x, axis=1, keepdims=True)
        # return softmax(x)

    def forward(self, X):
        print("**** Going through the Input-hidden layer: ")
        time.sleep(1)

        input_hidden = np.dot(X.T, self.weights_input_hidden) + self.bias_input_hidden
        print("Dot product results: ", input_hidden)
        input_hidden = self.hyperbolic_tan(input_hidden)
        print("Hidden Layer's final output: ", input_hidden)


        print("**** Going through the hidden-output layer: ")
        # time.sleep(1)

        hidden_output = np.dot(input_hidden, self.weights_hidden_output) + self.bias_hidden_output
        print("Dot product results: ", hidden_output)
        hidden_output = self.softmax(hidden_output)
        print("Final output: ", hidden_output)

        # input("")


        return hidden_output

    def one_hot_encode(self, y):


        encoded_arr = np.zeros((y.shape), dtype=int)

        print("1.... :One Hot encoded array: ", encoded_arr)

        print("*****")

        mask = np.max(y, axis=1, keepdims=True)
        print(mask)

        print("2....: One Hot encoded array: ", encoded_arr)

        return encoded_arr



    def loss_accuracy(self, y, y_pred):
        y_one_hot = self.one_hot_encode(y, y_pred.shape[1])        #convert y to one-hot
        return -np.sum(y_one_hot * np.log(y_pred + 1e-10))     #add small epsilon to avoid log(0)


                   #0               1                2               3
X = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9]])
print(X)
y = np.array([0, 0, 1, 1, 2, 2, 3, 3, 3])

neuron = MLP()
neuron.init_params(2, 3, 4)

input("")

y_pred = []

for item in X:
    print(item)

    i = neuron.forward(item)

    item_reshaped = i.reshape(1, -1)  # Reshape to (1, 2)


    y_pred.append(item_reshaped)

y_pred = np.vstack(y_pred)

print(y_pred)


input("")
# loss = tf.keras.losses.categorical_crossentropy(y, y_pred)
y_pred_encoded = neuron.one_hot_encode(y_pred)



input("")





