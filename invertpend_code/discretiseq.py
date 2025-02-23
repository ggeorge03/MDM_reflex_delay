import numpy as np
from ddeint import ddeint
import matplotlib.pyplot as plt
from matplotlib import animation

# system parameters:
M = 3.0  # Mass of the cart
m = 1.0  # Mass of the pendulum
l = 3.0  # Length of the pendulum
g = 9.81  # Acceleration due to gravity
y_cart = 5

s, d = 1, 1

A = np.array([
    [0, 1, 0, 0],
    [0, -d/M, -m*g/M, 0],
    [0, 0, 0, 1],
    [0, -s*d/(M*l), -s*(m+M)*g/(M*l), 0]
])

# Define B matrix
B = np.array([
    [0],
    [1/M],
    [0],
    [s/(M*l)]
])
B=B.reshape(-1,1)

# Define Q matrix
Q = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 10, 0],
    [0, 0, 0, 100]
])


def syseq(Y, t, tau):

    # Linearise eq:
    u = 1 # change later, placeholder. Ideally will get for different controllers it will change..
    state = np.dot(A,Y(t))
    control = np.dot(B,u)
    dxdt, dx_dotdt, dthetadt, dtheta_dotdt = state.reshape(-1,1) + control.reshape(-1,1)

    
    return [dxdt, dx_dotdt, dthetadt, dtheta_dotdt]


def ics(t): # ddeint requires initial conditions to a callable func. essentially a history func idk tho
    return [0, 0, 0, 0] # x, x_dot, theta, theta_dot respectively

def control_input(t, tau, n_steps):
    u = np.zeros_like(t)
    u[t < tau] = 0 # input before delay should be nothing as human won't have realised to make a change.
    print(u)
    return u

t = np.linspace(0,100, 1000)
delay = 2
yy = ddeint(func=syseq, g = ics, tt=t, fargs=(delay,)) # fargs needs to be passed a a tuple, thf we place a comma 
# ( even though theres only one element )

# yy will be a numpy array by default due to how ddeint does it's calculations.


x, x_dot, theta, theta_dot = yy[:, 0], yy[:, 1], yy[:, 2], yy[:, 3]
plt.title("Position")
plt.plot(t,x, label = " position of cart "); plt.plot(t, x_dot , label = " velocity of cart "); plt.legend()
plt.show()

plt.title("Angle")
plt.plot(t,theta, label="anglular disp"); plt.plot(t, theta_dot, label = "angluar vel"), plt.legend()
plt.show()


# Now to animate it -----
def bob_xy(x_pos, theta_val):
    x_val = l * np.sin(theta_val) + x_pos
    y_val = -l * np.cos(theta_val) + y_cart
    return x_val, y_val

bx, by = bob_xy(x, theta)

def animate(i):
    time_text.set_text(f'Time = {t[i]:.1f} s')
    x_bob, y_bob = bx[i], by[i]
    x_cart = x[i]
    ln1.set_data([x_bob], [y_bob])
    rod.set_data([x[i], x_bob], [y_cart, y_bob])
    trolley.set_data([x_cart-1, x_cart+1], [y_cart])

fig, ax = plt.subplots(1,1)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
time_text.set_text('')
ax.grid()
ln1, = plt.plot([],[], 'ro', markersize = (m*5)**0.5)
rod, = ax.plot([], [], 'k-', lw=2)  # black line for the rod
trolley, = ax.plot([],[], 'k-', lw=2)
ax.set_ylim(-1,16)
ax.set_xlim(-20,20)
ax.set_aspect('equal')

ani = animation.FuncAnimation(
    fig,
    animate,
    frames=len(t),
    interval = 200,
    repeat = True
    )

plt.show()
