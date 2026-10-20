# Multi-layer perceptron for non-linear XOR classification

This laboratory implements a foundational Multi-Layer Perceptron (MLP) from
first principles using NumPy. The implementation solves the non-linearly
separable Exclusive OR (XOR) problem through manual analytical derivations,
vectorized forward propagation, backward propagation via the chain rule, and
batch gradient descent optimization.

## Contents

- Implementation notebook: [main.ipynb](./main.ipynb) containing manual
  analytical verification, vectorized forward/backward passes, and full XOR
  convergence training.

## Architecture

The network consists of a two-layer feedforward architecture:
- Input layer: 2 neurons ($x \in \mathbb{R}^{1 \times 2}$) representing the
  binary input features $x_1$ and $x_2$.
- Hidden layer: 2 neurons with logistic sigmoid activation $\sigma(z) =
  \frac{1}{1 + e^{-z}}$, parameterized by weight matrix $W^{(1)} \in
  \mathbb{R}^{2 \times 2}$ and bias vector $b^{(1)} \in \mathbb{R}^{1 \times 2}$.
- Output layer: 1 neuron with logistic sigmoid activation, parameterized by
  weight vector $W^{(2)} \in \mathbb{R}^{2 \times 1}$ and bias scalar $b^{(2)}
  \in \mathbb{R}^{1 \times 1}$.

## Methodology

### 1. Analytical derivation and verification
The implementation defines explicit mathematical steps for a sample $[1, 0]$
with target $y = 1$:
- Hidden pre-activations: $z^{(1)} = x W^{(1)} + b^{(1)}$
- Hidden activations: $a^{(1)} = \sigma(z^{(1)})$
- Output pre-activation: $z^{(2)} = a^{(1)} W^{(2)} + b^{(2)}$
- Prediction: $\hat{y} = \sigma(z^{(2)})$
- Loss: Half Mean Squared Error $L = \frac{1}{2}(\hat{y} - y)^2$
- Error deltas and parameter gradients:
  $\delta^{(2)} = (\hat{y} - y) \odot \sigma'(z^{(2)})$,
  $\delta^{(1)} = (\delta^{(2)} (W^{(2)})^T) \odot \sigma'(z^{(1)})$

The computed values in Python are verified against manual analytical ground
truth to confirm mathematical exactness.

### 2. Modular neural network class
The logic is encapsulated in `TwoLayerNeuralNetwork`, supporting:
- Vectorized batch matrix operations.
- Exact backpropagation gradient computations.
- Parameter updates via gradient descent: $\theta \leftarrow \theta - \eta
  \nabla_{\theta} L$.

## Convergence and results

When trained on the full XOR truth table over 10,000 epochs with learning rate
$\eta = 1.0$, the network separates the non-linear decision boundary:

| $x_1$ | $x_2$ | Target $y$ | Predicted $\hat{y}$ | Rounded output |
| :---: | :---: | :---: | :---: | :---: |
| 0 | 0 | 0 | 0.0526 | 0 |
| 0 | 1 | 1 | 0.9490 | 1 |
| 1 | 0 | 1 | 0.9490 | 1 |
| 1 | 1 | 0 | 0.0566 | 0 |

## Tech stack

- Language: Python 3
- Computation: NumPy (vectorized array operations)
- Data verification: Pandas

Authored by Youssef Fellah.
Developed for the Engineering Cycle at Mundiapolis University.
