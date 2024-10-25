from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import control 



class Astrobee(object):
    def __init__(self,
                 mass=9.6,
                 mass_ac=11.3,
                 inertia=0.25,
                 h=0.1,
                 **kwargs):
        """
        Astrobee Robot, NMPC tester class.

        :param mass: mass of the Astrobee
        :type mass: float
        :param inertia: inertia tensor of the Astrobee
        :type inertia: np.diag
        :param h: sampling time of the discrete system, defaults to 0.01
        :type h: float, optional
        """

        # Model
        self.n = None
        self.m = None
        self.dt = h

        # Model prperties
        self.mass = mass + mass_ac
        self.inertia = inertia

        # Linearized model for continuous and discrete time
        self.Ac = None
        self.Bc = None
        self.Ad = None
        self.Bd = None


    def cartesian_ground_dynamics(self):
        """
        Helper function to populate Ac and Bc with continuous-time
        dynamics of the system.
        """

        # Jacobian of exact discretization
        self.n = 6
        self.m = 3

        Ac = np.array([
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ])

        # Define matrix B
        Bc = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [1/self.mass, 0, 0],
            [0, 0, 0],
            [0, 1/self.mass, 0],
            [0, 0, 1/self.inertia]
        ])
        
        self.Ac = Ac
        self.Bc = Bc

        return self.Ac, self.Bc

    def cartesian_3d_dynamics(self):
        """
        Helper function to populate Ac and Bc with continuous-time
        dynamics of the system.
        """

        # Jacobian of exact discretization
        self.n = 8
        self.m = 4

        Ac = np.zeros((self.n, self.n))
        Bc = np.zeros((self.n, self.m))

        # TODO: Fill the matrices Ac and Bc according to the model in (1), adding
        #       the proper component for translation on Z

        self.Ac = Ac
        self.Bc = Bc

        return self.Ac, self.Bc

    def linearized_dynamics(self, x, u):
        """
        Linear dynamics for the Astrobee, continuous time.

        :param x: state
        :type x: np.ndarray, ca.DM, ca.MX
        :param u: control input
        :type u: np.ndarray, ca.DM, ca.MX
        :return: state derivative
        :rtype: np.ndarray, ca.DM, ca.MX
        """

        xdot = self.Ac @ x + self.Bc @ u

        return xdot

    def c2d(self, A, B, C, D):
        """
        Continuous to Discrete-time dynamics
        """
        # create a continuous time system in state space form
        continuous_system = control.ss(A, B, C, D)
        # create a discrete time system in state space form
        discrete_system   = control.c2d(continuous_system, self.dt, method='zoh')
        # extract the discrete time matrices
        ( Ad_list , Bd_list , Cd_list , Dd_list ) = control.ssdata( discrete_system  )
        
        # convret the list to numpy arrays
        Ad = np . array ( Ad_list )
        Bd = np . array ( Bd_list )
        Cd = np . array ( Cd_list )
        Dd = np . array ( Dd_list )
        
        return Ad,Bd,Cd,Dd

    def set_discrete_dynamics(self, Ad, Bd):
        """
        Helper function to populate discrete-time dynamics

        :param Ad: discrete-time transition matrix
        :type Ad: np.ndarray, ca.DM
        :param Bd: discrete-time control input matrix
        :type Bd: np.ndarray, ca.DM
        """

        self.Ad = Ad
        self.Bd = Bd

    def linearized_discrete_dynamics(self, x, u):
        """
        Method to propagate discrete-time dynamics for Astrobee

        :param x: state
        :type x: np.ndarray, ca.DM
        :param u: control input
        :type u: np.ndarray, ca.DM
        :return: state after dt seconds
        :rtype: np.ndarray, ca.DM
        """

        if self.Ad is None or self.Bd is None:
            print("Set discrete-time dynamics with set_discrete_dynamcs(Ad, Bd) method.")
            return np.zeros(x.shape[0])

        x_next = self.Ad @ x + self.Bd @ u

        return x_next

    def set_trajectory(self, time, type="2d", x_off=np.zeros((4, 1)), fp=0.1, ft=0.01):
        """
        Helper methjod to create a trajectory for Astrobee to perform in open-loop

        :param time: total length of the trajectory
        :type time: float
        """
        '''
        In questa sezione andiamo a creare la traiettoria da delle funzioni seno e coseno.
        '''
        
        t = np.linspace(0, time, int(time / self.dt))
        px = 0.025 * np.cos(2 * np.pi * fp * t) + x_off[0]  # 0.05 * np.ones(t.shape)
        py = 0.025 * np.sin(2 * np.pi * fp * t) + x_off[1]  # np.zeros(t.shape)
        theta = 0.05 * np.cos(2 * np.pi * ft * t + x_off[3])  # np.zeros(t.shape)
        if type != "2d":
            pz = 0.025 * np.cos(2 * np.pi * fp * t) + x_off[2]
            self.trajectory = np.vstack((px, py, pz, theta))
        else:
            self.trajectory = np.vstack((px, py, theta))

    def get_trajectory(self, t_start, t_end=None):
        """
        Get part of the trajectory created previously.

        :param t_start: starting time
        :type t_start: float
        :param t_end: ending time, defaults to None
        :type t_end: float, optional
        """

        start_idx = int(t_start / self.dt)

        if t_end is None:
            piece = self.trajectory[:, start_idx:]
        else:
            end_idx = int(t_end / self.dt)
            piece = self.trajectory[:, start_idx:end_idx]

        return piece



astrobee = Astrobee()

# Print the value of inertia
astrobee.inertia
