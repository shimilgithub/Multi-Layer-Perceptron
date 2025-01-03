"""This module implements multilayer perceptron"""
import numpy as np

class MultiLayerPerceptron:
    """This class implements Multi-Layer Perceptron (MLP) neural network.
    Attributes:net (dict): Dictionary containing weights and biases for each layer of the network."""
    
    def __init__(self):
        """Initializes the MultiLayerPerceptron instance and creates an empty network dictionary."""
        self.net = {}
        pass

    def init_network(self):
        """Initializes the neural network with predefined weights and biases for three layers.
        Layers:
            - Layer 1: 2 inputs, 3 neurons
            - Layer 2: 3 inputs, 2 neurons
            - Layer 3: Output layer with 2 inputs, 2 outputs"""
        net = {}

        # layer 1
        net['w1'] = np.array([[0.7, 0.9, 0.3],[0.5, 0.4, 0.1]])
        net['b1'] = np.array([1, 1, 1])

        # layer 2
        net['w2'] = np.array([[0.2, 0.3], [0.4, 0.5], [0.22, 0.1234]])
        net['b2'] = np.array([0.5, 0.5])

        # layer 3(output layer)
        net['w3'] = np.array([[0.7, 0.1], [0.123, 0.314]])
        net['b3'] = np.array([0.1, 0.2])

        self.net = net

    def forward(self, x):
        """Performs a forward pass through the network, applying weights, biases, and activation functions.
        Parameters: x (numpy array): Input vector for the network.
        Returns:numpy array: Output of the network after the forward pass."""

        w1, w2, w3 = self.net['w1'], self.net['w2'], self.net['w3']
        b1, b2, b3 = self.net['b1'], self.net['b2'], self.net['b3']

        a1 = np.dot(x, w1) + b1
        z1 = self.sigmoid(a1)

        a2 = np.dot(z1, w2) + b2
        z2 = self.sigmoid(a2)

        a3 = np.dot(z2, w3) + b3
        y = self.identity(a3)

        return y
    
    def identity(self, x):
        """Identity function used for the output layer.
        Parameters:x (numpy array): Input to the identity function.
        Returns:numpy array: The same input x as output."""
        return x
    
    def sigmoid(self, x):
        """Sigmoid activation function used for the hidden layers of the network.
        Parameters:x (numpy array): Input to the sigmoid function.
        Returns:numpy array: Transformed output where each element is between 0 and 1."""
        return 1/(1 + np.exp(-x))
