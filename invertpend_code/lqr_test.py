import numpy as np
import scipy.linalg
# import scipy.integrate
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from ddeint import ddeint
from Eqs_of_motion import cartpend

# System parameters
m = 0.191  # Mass of pendulum
M = 1  # Mass of cart
L = 3  # Length of pendulum
g = -10  # Gravity
d = 1  # Damping coefficient
tau = 0.01

s = 1  # System mode

# State-space matrices
A = np.array([[0, 1, 0, 0],
              [0, -d/M, -m*g/M, 0],
              [0, 0, 0, 1],
              [0, -s*d/(M*L), -s*(m+M)*g/(M*L), 0]])

B = np.array([[0],
              [1/M],
              [0],
              [s*1/(M*L)]])

# LQR cost matrices
Q = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 10, 0],
              [0, 0, 0, 100]])

R = np.array([[0.001]])

# Solve the continuous-time algebraic Riccati equation (CARE)
P = scipy.linalg.solve_continuous_are(A, B, Q, R)

# Compute LQR gain K
K = np.linalg.inv(R) @ B.T @ P


def cartpend_lqr(t, Y, y0):
    '''.'''
    if s == -1:
        y_ref = np.array([4, 0, 0, 0])
    else:
        y_ref = np.array([1, 0, np.pi, 0])

    if t > tau:
        y_tau = Y(t - tau)  # Get state from history
    else:
        y_tau = y0  # Use the initial state if time <= tau

    u_tau = (-K @ (y_tau - y_ref)).item()

    # # Get delayed state Y(t - tau), use initial history if t < tau
    # y_tau = y(t - tau) if t > tau else y(0)

    # # Compute delayed control input
    # u_delayed = (-K @ (y_tau - y_ref)).item()

    # u = (-K @ (y - y_ref)).item()  # Compute control input
    # Call cart-pendulum dynamics
    return cartpend(Y(t), m, M, L, g, d, u_tau)


tspan = np.arange(0, 10, 0.001)
if s == -1:
    y0 = np.array([0, 0, 0, 0])
else:
    y0 = np.array([-3, 3, np.pi + 0.1, 2])
# Define history function


def history(t):
    return y0


# Time span and initial conditions

# if s == -1:
#     y0 = np.array([0, 0, 0, 0])
# else:
#     y0 = np.array([-3, 3, np.pi + 0.1, 2])

# sol = ddeint(cartpend_lqr, history, tspan)
sol = ddeint(lambda Y, t: cartpend_lqr(t, Y, y0), history, tspan)

# Solve the differential equation
# sol = scipy.integrate.solve_ivp(
#     cartpend_lqr, [0, 10], y0, t_eval=tspan, method='RK45')

# Define history function (for times t < 0)


# def history(t):
#     return y0


# # Solve the delay differential equation
# sol = ddeint(cartpend_lqr, history, tspan)

# Extract cart position and pendulum angle
cart_x = sol[0]  # Cart position
pendulum_theta = sol[2]  # Pendulum angle

# Compute pendulum end coordinates
pendulum_x = cart_x + L * np.sin(pendulum_theta)
pendulum_y = -L * np.cos(pendulum_theta)

# Set up figure for animation
fig, ax = plt.subplots()
ax.set_xlim(min(cart_x) - 1, max(cart_x) + 1)
ax.set_ylim(-L - 1, L + 1)
ax.set_aspect('equal')
ax.grid()

cart, = ax.plot([], [], 'ks', markersize=10)
pendulum, = ax.plot([], [], 'ro-', lw=2)


def init():
    '''Initialize animation objects.'''
    cart.set_data([], [])
    pendulum.set_data([], [])
    return cart, pendulum


def update(frame):
    '''Update the animation with the new positions of cart and pendulum.'''
    cart.set_data(cart_x[frame], 0)
    pendulum.set_data([cart_x[frame], pendulum_x[frame]],
                      [0, pendulum_y[frame]])
    return cart, pendulum


# Get the number of time steps in the solution
frames = len(sol[0])

# Create the animation
ani = animation.FuncAnimation(
    fig, update, frames=frames, init_func=init, blit=True, interval=0)
plt.show()
