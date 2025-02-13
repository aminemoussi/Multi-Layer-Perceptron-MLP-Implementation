import numpy as np
import math
from softmax import softmax
import time
import math



class MLP:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights_input_hidden = None
        self.bias_input_hidden = None
        self.weights_hidden_output = None
        self.bias_hidden_output = None



    def init_params(self, ni, nh, no):
        self.weights_input_hidden = np.random.normal(0, 0.3, (nh, ni))
        self.bias_input_hidden = np.zeros((1, nh))                             # because we'll use it for each value of x, otherwise use nhxm

        self.weights_hidden_output = np.random.normal(0, 0.3, (no, nh))
        self.bias_hidden_output = np.zeros((1, no))     #shouldn't exist nrmlmn

        print("** Initializing weights and biases **")
        print("Input -> Hidden Weights: ", self.weights_input_hidden, "\n")
        print("Weights: ", self.bias_input_hidden, "\n")

        time.sleep(3)

        print("Hidden -> Output Weights: ", self.weights_hidden_output, "\n")
        print("Weights: ", self.bias_hidden_output, "\n")




    def hyperbolic_tan(self, x):    #Hyperbolic Tang
        return np.tanh(x)





    def forward(self, X, A1, A2):
        print("**** Going through the Input-hidden layer: ")
        time.sleep(1)

        input_hidden = np.dot(self.weights_input_hidden, X.T) + self.bias_input_hidden
        print("Z1: ", input_hidden)
        input_hidden = self.hyperbolic_tan(input_hidden)
        input_hidden = input_hidden.T

        A1 = np.hstack((A1, input_hidden))

        print("A1: ", A1)

        # input("")


        # print(input_hidden)

        print("**** Going through the hidden-output layer: ")
        # time.sleep(1)
        hidden_output = np.dot(self.weights_hidden_output, input_hidden)                                    #no output bias
        hidden_output = hidden_output + self.bias_hidden_output.T

        # hidden_output = np.dot(self.weights_hidden_output, input_hidden) + self.bias_hidden_output        #with output bias
        # print("hidden output bias: ", self.bias_hidden_output)
        # print("hidden to output values: ", hidden_output)
        # hidden_output = hidden_output + self.bias_hidden_output.T

        print("Z2: ", hidden_output)

        hidden_output = softmax(hidden_output)

        A2 = np.hstack((A2, hidden_output))

        print("A2: ", A2)
        print("\n")

        input("")
        return A1, A2



    def one_hot_encode(self, y, classes):
        rows = y.shape[0]

        encoded_arr = np.zeros((rows, classes))

        encoded_arr[np.arange(y.shape[0]), y] = 1

        return encoded_arr





    def loss(self, y_one_hot_encoded, y_predicted):
        selected_vals = y_predicted[y_one_hot_encoded == 1]

        sum = 0

        for val in selected_vals:
            sum += math.log1p(val)       #log is the natural logarithm (base e)

        sum = sum / y_one_hot_encoded.shape[0]

        return sum





    def accuracy(self, y, y_predicted):

        true = 0
        total = y.shape[0]

        max_indeces = np.argmax(y_predicted, axis=1)

        print(max_indeces)

        for i in range(9):
            if max_indeces[i] == y[i]:
                true += 1

        return true, total




                   #0               1                2               3
X = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9]])
print(X)

y = np.array([0, 0, 1, 1, 2, 2, 3, 3, 3])

neuron = MLP()
neuron.init_params(2, 3, 4)




input("")

A1 = np.empty((3, 0))
A2 = np.empty((4, 0))
y_pred = []

for item in X:
    print(item)

    A1, A2 = neuron.forward(item, A1, A2)



print("The Final A1 Matrix:", A1)
print("\n")
A2 = A2.T
print("The Final A2 Matrix:", A2)





input("")

print("One Hot Encoding:")
print("\n")

input("")

# loss = tf.keras.losses.categorical_crossentropy(y, y_pred)
y_encoded = neuron.one_hot_encode(y, 4)

print("Y before: ", y)
print("\n")
print("Y encoded: ")
print(y_encoded)





input("")
print("Loss function calculation:")
loss = neuron.loss(y_encoded, A2)
print("The optained Loss function: ", loss)

accuracy = neuron.accuracy(y, A2)
print("The optained Accuracy: ", accuracy)

