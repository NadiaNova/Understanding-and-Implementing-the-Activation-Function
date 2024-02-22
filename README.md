# Understanding-and-Implementing-the-Activation-Function

![App Screenshot](https://www.geeksforgeeks.org/wp-content/uploads/33-1-1.png)

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

![App Screenshot](https://github.com/NadiaNova/Understanding-and-Implementing-the-Activation-Function/blob/main/Image%20File/image-1.png)


# 2. Mathematical Exploration:

## Derivation of Activation Function Formula:

Let's derive the formula for the sigmoid activation function:

![App Screenshot](https://github.com/NadiaNova/Understanding-and-Implementing-the-Activation-Function/blob/main/Image%20File/formula-1.png)
​
To derive this, we start with the logistic function:

![App Screenshot](https://github.com/NadiaNova/Understanding-and-Implementing-the-Activation-Function/blob/main/Image%20File/formula-2.jpg)
​ 
where, z=wx+b (weighted sum of inputs plus bias).

Now, replacing z with x for simplicity, we get:

![App Screenshot](https://github.com/NadiaNova/Understanding-and-Implementing-the-Activation-Function/blob/main/Image%20File/formula-3.png)

This function outputs values between 0 and 1, making it suitable for binary classification tasks.

## Calculation of Derivative:

The derivative of the sigmoid function with respect to x can be calculated as follows:

![App Screenshot](https://github.com/NadiaNova/Understanding-and-Implementing-the-Activation-Function/blob/main/Image%20File/formula-4.png)

This derivative is particularly significant in the backpropagation process during neural network training. It determines the magnitude of adjustments made to the weights and biases during gradient descent.


![App Screenshot](https://github.com/NadiaNova/Understanding-and-Implementing-the-Activation-Function/blob/main/Image%20File/image-2.png)



## Different Types of Activation Function:

**1. Sigmoid Activation Function:**

Definition:
The sigmoid activation function, also known as the logistic function, transforms any real-valued number into a value between 0 and 1. It has an S-shaped curve and is given by the formula:

![App Screenshot](https://github.com/NadiaNova/Understanding-and-Implementing-the-Activation-Function/blob/main/Image%20File/formula-1.png)

Properties:

    - Range: (0,1)
    - Smooth gradient: This facilitates stable learning during gradient descent optimization.
    - Squashes input to a range suitable for binary classification tasks.
    - However, it suffers from the vanishing gradient problem, especially for extreme input values, which can slow down training.

**2. Tanh Activation Function:**

Definition:
The hyperbolic tangent (tanh) activation function is similar to the sigmoid but squashes input values to a range between -1 and 1. It is given by the formula:

![App Screenshot](https://github.com/NadiaNova/Understanding-and-Implementing-the-Activation-Function/blob/main/Image%20File/formula-5.png)

Properties:

    - Range: (−1,1)
    - Similar S-shaped curve to the sigmoid but centered at 0.
    - It addresses the vanishing gradient problem better than sigmoid as it has non-zero derivatives across its entire range.
    - It is commonly used in hidden layers of neural networks.

**3. ReLU (Rectified Linear Unit) Activation Function:**

Definition:
The ReLU activation function replaces all negative input values with zero, resulting in a piecewise linear function. It is defined as:

![App Screenshot](https://github.com/NadiaNova/Understanding-and-Implementing-the-Activation-Function/blob/main/Image%20File/formula-6.png)

Properties:

   - Simple and computationally efficient.
   - It facilitates faster convergence during training compared to sigmoid and tanh.
   - However, ReLU neurons can "die" during training if they consistently output zero for all inputs, leading to dead neurons with zero gradients.

**4. Leaky ReLU Activation Function:**

Definition:
Leaky ReLU is a variant of ReLU that introduces a small slope for negative input values, instead of zero. It is defined as:

![App Screenshot](https://github.com/NadiaNova/Understanding-and-Implementing-the-Activation-Function/blob/main/Image%20File/formula-7.png)

otherwise​
where α is a small positive constant (typically around 0.01).

Properties:

    - Addresses the "dying ReLU" problem by allowing a small gradient for negative inputs.
    - It prevents neurons from being completely inactive, ensuring they can still contribute to the learning process.
    - It can be beneficial in scenarios where ReLU leads to dead neurons.

In summary, each activation function has its own characteristics and advantages. The choice of activation function depends on the specific requirements and nature of the problem at hand, as well as considerations such as computational efficiency and gradient behavior during training.



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

![App Screenshot](https://github.com/NadiaNova/Understanding-and-Implementing-the-Activation-Function/blob/main/Image%20File/image-3.png)


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

## References:

1. Visit my Kaggle profile for code: https://www.kaggle.com/afsananadia
2. Image source: https://machinelearningmastery.com/wp-content/uploads/2020/12/How-to-Choose-an-Hidden-Layer-Activation-Function.png
3. Image source: https://www.geeksforgeeks.org/wp-content/uploads/33-1-1.png
4. Image source: https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.analyticsvidhya.com%2Fblog%2F2021%2F04%2Factivation-functions-and-their-derivatives-a-quick-complete-guide%2F&psig=AOvVaw3r28Ww7UjXBmGuNguMR0N2&ust=1708662246092000&source=images&cd=vfe&opi=89978449&ved=0CBMQjRxqFwoTCMDlhoSNvoQDFQAAAAAdAAAAABAE



```bash
pip install foobar
```

![Alt text](URL "Title")
