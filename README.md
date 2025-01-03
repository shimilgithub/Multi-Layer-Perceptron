# Implementation of MultiLayer Perceptron 

## Overview

A MultiLayer Perceptron (MLP) is a type of artificial neural network consisting of multiple layers of neurons. Each neuron uses an activation function to process inputs and produce outputs. MLPs are commonly used for tasks such as classification and regression. This assignment involves implementing an MLP class, validating it through tests, and documenting your work in a script and a Jupyter notebook.

## Repository Structure

The repository structured as follows:

```
├── multilayer_perceptron.py    # Implementation of the MultiLayerPerceptron class
├── module4.py                  # Script to test and validate the MLP class
├── module4.ipynb               # Jupyter Notebook for experimentation and demonstration
```

## Technologies Used

- **Python**: Core programming language
- **NumPy**: For numerical computations (if required)
- **Jupyter Notebook**: For interactive testing and demonstration

## File Details

### 1. `multilayer_perceptron.py`
This file contains the implementation of the `MultiLayerPerceptron` class. This class includes methods to:
- Initialize the network structure
- Perform forward propagation
- Train the model using backpropagation
- Make predictions

### 2. `module4.py`
This script is for testing and validating the functionality of the `MultiLayerPerceptron` class. Include:
- Sample datasets
- Code to initialize and train the MLP
- Validation results

### 3. `module4.ipynb`
This Jupyter Notebook demonstrates how the `MultiLayerPerceptron` class works. Include:
- A brief explanation of the steps
- Code cells for training and validating the MLP
- Visualizations (if applicable)

## How to Run

1. **Clone the Repository**
   ```bash
   git clone https://github.com/shimilgithub/Multi-Layer-Perceptron.git
   cd Multi-Layer-Perceptron
   ```

2. **Run the Script**
   ```bash
   python module4.py
   ```

3. **Run the Jupyter Notebook**
   ```bash
   jupyter notebook module4.ipynb
   ```
