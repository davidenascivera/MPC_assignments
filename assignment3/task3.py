import numpy as np

from astrobee import Astrobee
from dlqr import DLQR
from simulation import EmbeddedSimEnvironment
import matplotlib.pyplot as plt

# ------------------------------
# Part I - LQR Design
# ------------------------------
# Instantiate an Astrobee
abee = Astrobee(h=0.1)

# Linearization around reference point
x_star = np.zeros((12, 1))
x_star[0] = 1
x_star[1] = 0.5
x_star[2] = 0.1
x_star[6] = 0.087
x_star[7] = 0.077
x_star[8] = 0.067

A, B = abee.create_linearized_dynamics(x_bar=x_star)

C = np.diag(np.ones(12))
D = np.zeros((12, 6))

Ad, Bd, Cd, Dd = abee.c2d(A, B, C, D)

ctl = DLQR(Ad, Bd, C)

abee.set_discrete_dynamics(Ad, Bd)

E, V = np.linalg.eig(Ad.T) #We need to transpose the Ad in order to fine the left eigenvector 
#In the matrix E we have all the left eigenvector as a columns
print(np.round(Bd,2))

print(f"eigenvector {E}")
print(f"eigenvalue matrix shape: {E.shape}, eigenvector matrix shape: {V.shape}, matrix Bd shape: {Bd.shape}")
print("eigenvectors")
print(np.round(V,3))
print("vector product")
print(np.round(V.T @ Bd,3))

'''
We have verified that no row is zero, so we do not have any loss of controllability.
We could have also verify the controllability of the matrix using

result = np.hstack((Ad @ Bd, Bd))
print(result.shape)
print(np.linalg.matrix_rank(result))

'''


R_coefficients = np.ones(6)
Q_coefficients = np.ones(12)
result = np.hstack((Ad @ Bd, Bd))
print(result.shape)
print(np.linalg.matrix_rank(result))




Q_coefficients[0:3]  = 1   # coeff pos
Q_coefficients[3:6]  = 1   # coeff v
Q_coefficients[6:9]  = 1   # coeff theta
Q_coefficients[9:12] = 1  # coeff w

R_coefficients[0:3] = 1   # coeff f
R_coefficients[3:6] = 1   # coeff T

Q = np.diag(Q_coefficients)
R = np.diag(R_coefficients)

print("mat Q ---------------------------------")
print(np.round(Q))

print("mat R ---------------------------------")
print(R)

K, P = ctl.get_lqr_gain(Q, R)
#print(np.round(K,2))

#-------------Using M to reduce dimensionality -----------------------

M = np.zeros((9,12))
for i in range(9):
    M[i,i]=1
print("Matrix M:")
print(M)


Q_diag = np.ones(9)
Q_diag[0:3] = (1/0.06)**2
Q_diag[3:6] = (1/0.03)**2
Q_diag[6:9] = (10e-5/10e-7)**2

R_coefficients[0:3] = (1.85/0.85)**2   # coeff force
R_coefficients[3:6] = (1/0.04)**2   # coeff torque
R = np.diag(R_coefficients)

Qp = M.T @ np.diag(Q_diag) @ M

R = 100*R
print("Qp____________________________________")
print(np.round(Qp,5))   

print("R____________________________________")
print(np.round(R,2))


Kp, Pp = ctl.get_lqr_gain(Qp, R)

# Starting pose
x0 = np.zeros((12, 1))
# Set reference for controller
ctl.set_reference(x_star)
sim_env = EmbeddedSimEnvironment(model=abee,
                                 dynamics=abee.linearized_discrete_dynamics,
                                 controller=ctl.feedback,
                                 time=20)

t, y, u_og = sim_env.run(x0)
print(u_og)
sim_env.evaluate_performance(t, y, u_og)

import matplotlib.pyplot as plt

# Assuming u_og has been calculated in the previous simulation, let's simulate a u_og vector here for demonstration purposes
# For the real case, the u_og vector would b   e from the actual simulation environment's result
# Plot the control vector (u_og) over time for each control input
time_steps = np.arange(u_og.shape[1])

# Plot the control vector (u_og) over time for each control input with specific colors for groups
plt.figure(figsize=(10, 6))

# Plot the first 3 control inputs in red
for i in range(3):
    plt.plot(time_steps, u_og[i, :], label=f'u_og{i+1}', color='red')

# Plot the remaining control inputs in blue
for i in range(3, u_og.shape[0]):
    plt.plot(time_steps, u_og[i, :], label=f'u_og{i+1}', color='blue')

plt.title('Control Vector (u_og) Over Time')
plt.xlabel('Time Step')
plt.ylabel('Control Input')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()



# ------------------------------
# Part II - LQG Design
# ------------------------------
# Output feedback - measure position, attitude and angular velocity
#             Goal - estimate linear velocity
C = np.eye(3)
C = np.hstack((C, np.zeros((3, 3))))



# Create the matrices for Qn and Rn
# TODO: adjust the values of Qn and Rn to answer Q4 and Q5 - they start at 0
Q_diag = np.vstack((np.ones((3, 1)) * 1  , np.zeros((3, 1))))
R_diag = np.vstack((np.ones((3, 1)) * 1))


Qn = np.diag(Q_diag.reshape(6, ))
Rn = np.diag(R_diag.reshape(3, ))

print(np.round(Qn,1))
print(np.round(Rn,1))


abee.set_kf_params(C, Qn, Rn)
abee.init_kf(x0[0:6].reshape(6, 1))

sim_env_lqg = EmbeddedSimEnvironment(model=abee,
                                     dynamics=abee.linearized_discrete_dynamics,
                                     controller=ctl.feedback,
                                     time=20)
sim_env_lqg.set_estimator(True)
t, y, u = sim_env_lqg.run(x0)

#print(sim_env_lqg.trace_P_list)

import matplotlib.pyplot as plt

# Assuming u has been calculated in the previous simulation, let's simulate a u vector here for demonstration purposes
# For the real case, the u vector would be from the actual simulation environment's result
# Plot the control vector (u) over time for each control input
time_steps = np.arange(u.shape[1])
# Plot the control vector (u) over time for each control input with specific colors for groups
plt.figure(figsize=(10, 6))

# Plot the first 3 control inputs in red
for i in range(3):
    plt.plot(time_steps, u[i, :], label=f'u{i+1}', color='red')

# Plot the remaining control inputs in blue
for i in range(3):
    plt.plot(time_steps, u_og[i, :], label=f'u_og{i+1}', color='blue', linestyle='--')

plt.title('Control Vector (u) Over Time')
plt.xlabel('Time Step')
plt.ylabel('Control Input')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()
