import numpy as np

from astrobee_1d import Astrobee
from controller import Controller
from simulation import EmbeddedSimEnvironment

# Create pendulum and controller objects
abee = Astrobee(h=0.1)
ctl = Controller()

# Get the system discrete-time dynamics
# TODO: the method 'one_axis_ground_dynamics' needs to be completed inside the class Astrobee! (Q1)
A, B = abee.one_axis_ground_dynamics()

# TODO: Get the discrete time system with c2d. Four matrices are required here :)
Ad, Bd, Cd, Dd = abee.c2d( )
abee.set_discrete_dynamics(Ad, Bd)

# Plot poles and zeros
abee.poles_zeros(Ad, Bd, Cd, Dd)
# TODO: check the poles and zero an answer the question (Q3)

# Get control gains
ctl.set_system( ) # TODO: Set the discrete time system matrices
K = ctl.get_closed_loop_gain()

# Set the desired reference based on the dock position and zero velocity on docked position
dock_target = np.array([[0.0, 0.0]]).T
ctl.set_reference(dock_target)

# Starting position
x0 = [1.0, 0.0]

# Initialize simulation environment
sim_env = EmbeddedSimEnvironment(model=abee,
                                 dynamics=abee.linearized_discrete_dynamics,
                                 controller=ctl.control_law,
                                 time=40.0)
t, y, u = sim_env.run(x0)
sim_env.visualize()


# Disturbance effect
abee.set_disturbance()
sim_env = EmbeddedSimEnvironment(model=abee,
                                 dynamics=abee.linearized_discrete_dynamics,
                                 controller=ctl.control_law,
                                 time=40.0)
t, y, u = sim_env.run(x0)
sim_env.visualize()

# Activate feed-forward gain
# TODO: To activate the integral action you need to change class Controller() first !
ctl.activate_integral_action(dt=0.1, ki=0.045)
t, y, u = sim_env.run(x0)
sim_env.visualize()
