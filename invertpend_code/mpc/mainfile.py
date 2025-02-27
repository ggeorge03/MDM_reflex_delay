from predcontrol2 import MPC
import numpy as np
from anim import animPendulum

def initialise_variables():
    M = 4.0   # mass of the cart (kg)
    m_p = 0.1 # mass of the pendulum (kg)
    g = 9.81  # gravity (m/s^2)
    l = 5.0   # length of the pendulum (m)

    def initialise_matricies():
        A = [
            [0, 1, 0, 0],
            [0, -1/M, -m_p*g/M, 0],
            [0, 0, 0, 1],
            [0, -1/(M*l), -(M+m_p)*g/(M*l), 0]
            ]

        B = [
            [0],
            [1/M+m_p],
            [0],
            [-1 / M*l]
            ]
        C = np.eye(4)

        Q = np.diag([1,1,1000,10])
        
        R = 0.0001
        return A,B,C,Q,R
    return initialise_matricies()

# linearise system around equilibrium point theta = 0.
matricies = initialise_variables()
mpc = MPC(*matricies,pred_hor=2, cont_hor= 0 ,desired_state=[0,0,0,0])
mpc.set_bounds([-5,-5,-np.pi/2, -2], [5,5,np.pi/2, 2]) # set constraints for the state.
x0 = np.array([0,0,-0.1,0])
states, inputs = mpc.sim_model(x0,n_iter=1000,dt=0.1)

x, x_dot, theta, theta_dot = states[:, 0], states[:, 1], states[:, 2], states[:, 3]
pend=animPendulum(theta, x, 1, 5, m=4)