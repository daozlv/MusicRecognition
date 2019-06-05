import numpy as np
import random
import os
import cv2

# Convention 2/matrix format: (Features, Samples)
class FeedForwardNN:
    """
    Some loose notes:
    -activation_functions defaults to using sigmoid only for all activations
    -cost_function defaults to cross entropy
    -deltal = lower case delta = error
    -layers order: layer 0 = input layer, layer 1 = 1st hidden layer, layer n-1 = final hidden layer, layer n = output layer
    -Scale of a NN from small to big: neuron -> layer -> NN -> sample -> batch -> epoch

    nn_sizes = [input neurons, hidden L1 neurons, ....., hidden Ln neurons, output neurons]
    activation_functions = ['sigmoid', 'sigmoid', 'relu', ...., 'sigmoid'] where len(activation_functions) = number of layers - 1
    """

    def __init__(self, nn_sizes, activation_functions=None, cost_function='mse'):
        self.layers = len(nn_sizes)

        # Initialze Parameters
        self.weight_sizes = zip(nn_sizes[1:],
                                nn_sizes[:-1])  # matrix shape: (next_layer_neurons, current_layer_neurons)
        self.weights = [np.random.randn(row, col) for row, col in self.weight_sizes]
        self.biases = [np.random.rand(row, 1) for row in nn_sizes[1:]]

        # Initialize activation and cost functions
        # Cost function default to MSE if none are given
        # Activation functions default to sigmoid if none are given
        self.cost_function = cost_function
        if activation_functions == None:
            self.activation_functions = ['sigmoid'] * (self.layers - 1)
        else:
            self.activation_functions = activation_functions

        # These dictionaries stores all the function options(will add more functions later to expand the options)
        self.cost_prime_dic = {'mse': self.msePrime}
        self.activations_dic = {'sigmoid': self.sigmoid}
        self.activations_prime_dic = {'sigmoid': self.sigmoidPrime}

    def sigmoid(self, z):
        a = 1 / (1 + np.exp(-z))
        return a

    def relu(self, z):
        a = z * (z > 0)
        return a

    def sigmoidPrime(self, z):
        da_dz = self.sigmoid(z) * (1 - self.sigmoid(z))
        return da_dz

    def msePrime(self, a_out, y_label):
        dc_da = (a_out - y_label)
        return dc_da

    def feedForward(self, a):
        """
        -Operation: Layer by layer
        -Forward feed the Input neuron activations through the the network and returns the linear and activation function
        values calculated at teach layer in a list.

        z_list = [ [[sample 1 layer input z],....,[sample n layer output z]],...,[] ] #len = len(layers-1)
        a_list = [ [[sample 1 layer input a],....,[sample n layer output a]],...,[] ] #len = len(layers)
        """

        z_list = []  # for use in Backprop
        a_list = [a]  # for use in Backprop
        for w, b, func in zip(self.weights, self.biases, self.activation_functions):
            z = np.dot(w, a) + b
            z_list.append(z)

            a = self.activations_dic[func](z)
            a_list.append(a)
        return z_list, a_list  # activations[-1] will be the activation of the output layer

    def backprop(self, z_list, a_list, deltal):
        """
        -Operation: Layer by layer
        -Calculate the parameter gradients of the Output layer first utilizing the deltal matrix that is passed into this function.
         Then iterate through all the other layers, except for the Input layer, backwards to update the deltal matrix and calculate
         the parameter gradients of each layer.

        z_list: Each element in the list is a matrix of z values of neurons in a layer. Each element is a layer in the network
        a_list: Each element in the list is a matrix of a values of neurons in a layer. Each element is a layer in the network
        """

        # Initialize parameter gradients for the sample
        # Note: nabla_w = dC/dw where C is the cost function
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        # Compute nabla_w & nabla_b connecting to the Output layer
        nabla_w[-1] = np.dot(deltal, a_list[-2].transpose())  # deltal * a_list[-2].transpose()
        nabla_b[-1] = deltal

        # Compute nabla_w & nabla_b between all layers except the Output layer
        # Iterating the layers backwards starting from the 2nd to last layer(since we already done the last layer above)
        # Note: the 0th layer is left out in the iterations
        for layer in reversed(range(1, self.layers - 1)):
            # Note: don't focus too much on the indexing since it can be a little confusing. Focus on the variable names instead when
            # just reading over these calculations.
            a_previous_layer = a_list[layer - 1]
            z_current_layer = z_list[layer - 1]
            da_dz_current_layer = self.activations_prime_dic[self.activation_functions[layer - 1]](z_current_layer)
            deltal = np.dot(self.weights[layer].transpose(), deltal) * da_dz_current_layer
            # deltal * np.dot(self.weights[layer].transpose(), da_dz_current_layer)
            # np.dot(self.weights[layer].transpose(), deltal) * da_dz_current_layer

            nabla_w[layer - 1] = np.dot(deltal, a_previous_layer.transpose())
            nabla_b[layer - 1] = deltal
        return nabla_w, nabla_b

    def update_mini_batch(self, batch, eta):
        """
        -Operation: Sample by sample
        -THIS IS WHERE THE MAIN BODY OF CALCULATIONS ARE DONE

        batch: (features, label) * samples
        """

        # Initialize parameter gradients for the batch
        batch_nabla_w = [np.zeros(w.shape) for w in self.weights]
        batch_nabla_b = [np.zeros(b.shape) for b in self.biases]

        batch_size = len(batch)
        for x_features, y_label in batch:
            # Feed foward
            z_list, a_list = self.feedForward(
                x_features)  # note:input data, x, are also the activations of the input layer

            # Compute Error aka deltal of the output layer
            deltal_output_layer = self.cost_prime_dic[self.cost_function](a_list[-1], y_label) \
                                  * self.activations_prime_dic[self.activation_functions[-1]](z_list[-1])

            # Backward propagate
            sample_nabla_w, sample_nabla_b = self.backprop(z_list, a_list, deltal_output_layer)

            # Update parameter gradients for the batch
            # Sum up the gradients across all the samples in the batch. The average batch parameter gradients will be used to
            # update the parameters after this step.
            batch_nabla_w = [bnw + snw for bnw, snw in zip(batch_nabla_w, sample_nabla_w)]
            batch_nabla_b = [bnb + snb for bnb, snb in zip(batch_nabla_b, sample_nabla_b)]

        # Update parameters
        self.weights = [w - eta * (bnw / batch_size) for w, bnw in zip(self.weights, batch_nabla_w)]
        self.biases = [b - eta * (bnb / batch_size) for b, bnb in zip(self.biases, batch_nabla_b)]

    def fit(self, training_data, epochs, batch_size, eta):
        """
        -Operation: Eopch by eopch & batch by batch
        training_data: (x_features, y_label) * samples
        eta = Learning rate
        """

        # Each epoch is 1 iteration through the entire training_data stochasticaly using mini-batches
        for epoch in range(0, epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[start_idx: start_idx + batch_size]
                for start_idx in range(0, len(training_data), batch_size)
            ]
            # The Gradient is updated after each mini-batch update
            for batch in mini_batches:
                self.update_mini_batch(batch, eta)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedForward(x)[1][-1]), y) for x, y in test_data]
        return str((sum(int(x == y) for (x, y) in test_results) / len(test_results)) * 100) + '%'

    def predict(self, test_image):
        res = self.feedForward(test_image)
        return res


# Get Training Data:
eight_path = './' + 'note_images/eight'
quarter_path = './' + 'note_images/quarter'
half_path = './' + 'note_images/half'
sixteen_path = './' + 'note_images/sixteen'

eight = os.listdir(eight_path)
quarter = os.listdir(quarter_path)
half = os.listdir(half_path)
sixteen = os.listdir(sixteen_path)

X_train = []
Images = sixteen[:7] + eight[:7] + quarter[:7] + half[:7]
paths = [sixteen_path, eight_path, quarter_path, half_path]
paths_counter = 0
index_counter = 0
for im in Images:
    image = cv2.imread(paths[paths_counter] + '/' + im, cv2.IMREAD_GRAYSCALE)
    X_train.append(image)

    index_counter += 1
    if index_counter == 7:
        index_counter = 0
        paths_counter += 1

X_train = [np.reshape(x, (len(X_train[0][1]) * len(X_train[0]), 1)) for x in X_train]

Y_train = []
encoded_labels = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
for label in encoded_labels:
    for j in range(7):
        temp = []
        for i in label:
            temp.append([i])
            Y_train.append(temp)

train_data = zip(X_train, np.array(Y_train))

# Get a sample image to test:
test_image = cv2.imread('./' + 'note_images/9.jpg', cv2.IMREAD_GRAYSCALE)
test_image = np.reshape(test_image, (400, 1))
test_image = test_image.tolist()

# Train model:
ffnn = FeedForwardNN([40*10, 30, 30, 30, 4]) #3 hidden layers with 30 nodes each
train_data = list(train_data)
ffnn.fit(train_data, 20, 50, 0.001)

result = ffnn.predict(test_image)
result(print)
