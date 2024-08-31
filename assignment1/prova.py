from control.matlab import place
import numpy as np

# Given matrices
Ad = np.array([[1.0, 0.01],
               [0.0, 1.0]])

Bd = np.array([[2.3923445e-06],
               [4.7846890e-04]])

# Desired poles very close to zero
desired_poles = [0.001, 0.001]

# Place the poles
L = place(Ad, Bd, desired_poles)

print("State feedback gain matrix L:", L)
