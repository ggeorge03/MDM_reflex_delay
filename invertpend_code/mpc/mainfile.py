from mpc_control import MPC
import numpy as np
from anim import animPendulum
import matplotlib.pyplot as plt
def initialise_variables():
    M = 2.0   # mass of the cart (kg)
    m_p = 1 # mass of the pendulum (kg)
    g = 9.81  # gravity (m/s^2)
    l = 1.0   # length of the pendulum (m)
    d = 1 # damping
    s = 1 # orientation ( s=1 means upright, -1 means hanging dir down)

    def initialise_matricies():
        A = np.array([ # state transition matrix
            [0, 1, 0, 0],
            [0, 0, 0, 0],  # normally theta would affect the card position, but because the cart is a finger, I don't think this should be the case.
            [0, 0, 0, 1],
            [0, -s*d/(M*l), -s*(m_p+M)*g/(M*l), 0]
        ])


        B = [ # this might not be totally right ( both for A and ), but it's the closest thing i can think of
            [0],
            [1/M+m_p],
            [0],
            [1 / M*l]
            ]
        C = np.eye(4)

        Q = np.diag([5000,1,10,10]) # large penalty on distance to match physical experiment
        
        R = 0.001
        return A,B,C,Q,R
    return initialise_matricies()

# linearise system around equilibrium point theta = 0.
matricies = initialise_variables()
mpc = MPC(*matricies,pred_hor=5, cont_hor= 0 ,desired_state=[0,0,0,0]) # dw about control horizon for now.
mpc.set_bounds([-5,-5,-0.8, -2], [5,5,0.8, 2]) # set constraints for the state.
x0 = np.array([0,0,-0.5,0]) # ics
states, inputs = mpc.sim_model(x0,n_iter=1000,dt=0.1, delay=5) # specify delay in timesteps ( for now, might change.)
# 1 timestep = 0.1 seconds ( for now )

x, x_dot, theta, theta_dot = states[:, 0], states[:, 1], states[:, 2], states[:, 3]
pend=animPendulum(theta, x, 1, 5, m=4)
plt.plot(theta)
plt.title("theta against time")
plt.show()
