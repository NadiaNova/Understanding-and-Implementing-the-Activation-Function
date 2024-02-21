# Understanding-and-Implementing-the-Activation-Function

# 1. Theoretical Understanding:
## Explanation of Activation Function:

An activation function is a crucial component of artificial neural networks that introduces non-linearity into the model. It operates on the weighted sum of inputs plus a bias and determines the output of a neuron. Mathematically, it can be represented as 

y=f(x), where 
x is the input, 
f(x) is the activation function, and 
y is the output.

Activation functions commonly used in neural networks include sigmoid, tanh, ReLU (Rectified Linear Unit), and softmax, among others.

## Purpose of Activation Functions:

Activation functions are utilized in neural networks for several reasons:

**Introduction of Non-linearity:** Activation functions allow neural networks to approximate non-linear functions, which increases the model's expressive power, enabling it to learn complex patterns.

**Normalization of Output:** Activation functions normalize the output of each neuron, constraining it within a certain range, which aids in stabilizing and speeding up the learning process.

**Feature Extraction:** Activation functions can also act as feature detectors, helping the network to focus on relevant features of the input data.

# 2. Mathematical Exploration:

## Derivation of Activation Function Formula:

Let's derive the formula for the sigmoid activation function:

f(x)= 1+e −x
 
1


To derive this, we start with the logistic function:

σ(z)= 
1+e 
−z
 
1
​
 
where 
z=wx+b (weighted sum of inputs plus bias).

Now, replacing z with x for simplicity, we get:

�
(
�
)
=
1
1
+
�
−
�
f(x)= 
1+e 
−x
 
1
​
 

This function outputs values between 0 and 1, making it suitable for binary classification tasks.

## Calculation of Derivative:

The derivative of the sigmoid function with respect to x can be calculated as follows:

f 
′
 (x)=f(x)∗(1−f(x))

This derivative is particularly significant in the backpropagation process during neural network training. It determines the magnitude of adjustments made to the weights and biases during gradient descent.


```bash
pip install foobar
```

# 3. Usage

```python
import numpy as np
import matplotlib.pyplot as plt

# Sigmoid Activation Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Tanh Activation Function
def tanh(x):
    return np.tanh(x)

# ReLU Activation Function
def relu(x):
    return np.maximum(0, x)

# Leaky ReLU Activation Function
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

# Generate data
x = np.linspace(-10, 10, 100)
y_sigmoid = sigmoid(x)
y_tanh = tanh(x)
y_relu = relu(x)
y_leaky_relu = leaky_relu(x)

# Plot
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(x, y_sigmoid)
plt.title('Sigmoid Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(x, y_tanh)
plt.title('Tanh Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(x, y_relu)
plt.title('ReLU Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(x, y_leaky_relu)
plt.title('Leaky ReLU Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True)

plt.tight_layout()
plt.show()

```


Analysis:
The sigmoid function squeezes input values to a range between 0 and 1.
It has a smooth gradient, which aids in gradient-based optimization algorithms.
However, it suffers from the vanishing gradient problem, where gradients become very small for large input values, hindering the training of deep neural networks.

# 4. Analysis:
## Advantages of Activation Functions:

Enable non-linear transformations, allowing neural networks to learn complex patterns.
Normalize output, aiding in convergence during training.
Act as feature detectors, enhancing the network's ability to extract relevant information.

## Disadvantages of Activation Functions:

Some activation functions (like sigmoid and tanh) suffer from the vanishing gradient problem, which can slow down or stall the learning process, particularly in deep networks.
Computational cost: Certain activation functions, especially those involving exponentials or complex operations, can be computationally expensive.

## Impact on Gradient Descent and Vanishing Gradients:

Activation functions influence the gradient descent process by determining the direction and magnitude of weight updates.
The problem of vanishing gradients, often encountered with sigmoid and tanh functions, can impede convergence, especially in deep networks. This occurs because the gradients diminish as they propagate backward through layers, making it challenging to update the weights of earlier layers effectively.

## Contributing



## License

[GNU GENERAL PUBLIC LICENSE](https://www.gnu.org/licenses/why-not-lgpl.html)

![Alt text](URL "Title")
