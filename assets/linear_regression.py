import numpy as np
import matplotlib.pyplot as plt

# Generate the test problem

# Set dimensionality
N = 10

# y = wx + noise in the interval [a, b]
a = 0
b = 1

w1 = 0
w2 = 2
noise_level = 0.1

# Define X
X = np.linspace(0, 1, N)

# Define Y
Y = w2 * X + w1 + noise_level * (b - a) * np.random.normal(0, 1, X.shape)

# Define the real line 
Y_real = w2 * X + w1

# Plot
plt.plot(X, Y, 'o')
plt.plot(X, Y_real, 'r')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.title('An Example of Regression')
plt.show()