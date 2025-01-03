"""This module used for testing the Multi-Layer Perceptron (MLP)."""

#import library and module
import numpy as np
from multilayer_perceptron import MultiLayerPerceptron

# Create an instance of MLP
mlp = MultiLayerPerceptron()

# Initialize the neural network
mlp.init_network()

# Perform a forward pass with a single input
y = mlp.forward(np.array([7.0, 2.0]))

# Print output
print("Single Input result:",y)

#Testing with multiple inputs
multiple_inputs = np.array([
    [7.0, 2.0],
    [1.5, 3.5],
    [2.1, 1.8],
    [6.3, 7.2]
])

# Perform forward pass for each input in multiple inputs
for i, input_data in enumerate(multiple_inputs):
    y_multiple = mlp.forward(input_data)
    print(f"Output for input {i+1} ({input_data}): {y_multiple}")