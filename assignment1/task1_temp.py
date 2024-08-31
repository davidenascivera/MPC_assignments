import numpy as np

from astrobee_1d import Astrobee
from controller import Controller
from simulation import EmbeddedSimEnvironment
from control.matlab import place

# Create pendulum and controller objects
abee = Astrobee(h=3)
ctl = Controller()


# Get the system discrete-time dynamics
# TODO: the method 'one_axis_ground_dynamics' needs to be completed inside the class Astrobee! (Q1)
A, B = abee.one_axis_ground_dynamics()

C = np.array([1,0])
print(C)



D = np.array([0])


Ad, Bd, Cd, Dd = abee.c2d(A,B,C,D)
# abee.poles_zeros(Ad, Bd, Cd, Dd) #PLOT Zeros and Poles

print(f"Ad = {Ad} \n Bn = {Bd}")

ctl.set_system(Ad, Bd, Cd, Dd)
abee.set_discrete_dynamics(Ad, Bd)

'''
in questa parte dobbiamo selezionare i poli per cui dopo L viene calcolata. 
'''
ctl.set_poles(p= 0.5, p2 = 0.55)
L = ctl.get_closed_loop_gain()
print(f"l1 e l2 :{L}")

Afeed = Ad - Bd @L
print(f"Mat retrazionata: {Afeed}")
#abee.poles_zeros(Afeed, Bd, Cd, Dd)


dock_target = np.array([[0.0, 0.0]]).T
ctl.set_reference(dock_target)   #cambia la var ref nell'oggetto

# Starting position
x0 = [1.0, 0.0]

xp = np.array([[2.0], 
              [0.5]])

# Example control input vector u (1x1)
up = np.array([[0.1]])



# Initialize simulation environment
sim_env = EmbeddedSimEnvironment(model=abee,
                                 dynamics=abee.linearized_discrete_dynamics,
                                 controller=ctl.control_law,
                                 time=40.0)
t, y, u = sim_env.run(x0)
sim_env.visualize() #funzione che serve per visualizzare


""""
introduction of disturbance
"""

# Disturbance effect
abee.set_disturbance()
sim_env = EmbeddedSimEnvironment(model=abee,
                                 dynamics=abee.linearized_discrete_dynamics,
                                 controller=ctl.control_law,
                                 time=40.0)
t, y, u = sim_env.run(x0)
sim_env.visualize()


# Activate feed-forward gain
ctl.activate_integral_action(dt=3, ki=0.01)
t, y, u = sim_env.run(x0)
sim_env.visualize()
