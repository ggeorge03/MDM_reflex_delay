# code largely taken from https://www.youtube.com/watch?v=ENNyltVTJaE&t=507s, with a few adjustments..

import  numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import odeint
from matplotlib import animation
import sympy as sm
xs = np.linspace(0,100,1000)


# ideally want to plot the frequency against phase for an inverted pendulum... why

t, m, g = sm.symbols("t m g")
the = sm.symbols('\theta', cls = sm.Function)
the = the(t)
the_d = sm.diff(the, t) # d theta d wrt time.
the_dd = sm.diff(the_d, t) # 2nd diff wrt time.
l = 4 # pendulum length.
k = 0.1 # damping coefficient
pivot_origin = (0,8)

x, y = sm.symbols("x y", cls=sm.Function)
# movement wrt origin
x =  l * sm.sin(the) 
y = -l * sm.cos(the)


#x_f = sm.lambdify(the, x)
#y_f = sm.lambdify(the, y)


# lagrangian mechanics:

T = 0.5*m*(sm.diff(x,t)**2 + sm.diff(y,t)**2) # storing it in cartesian coordinates accounts for moment of inertia anyway
U = m*g*y
L = T-U

LEq = (sm.diff(L, the) - sm.diff(sm.diff(L, the_d), t) ).simplify()

d_2 = sm.solve(LEq, the_dd)[0]  - k*the_d # the_dd as subj. 2nd term adds a slight dampening to the system.
d_1 = the_d # omega as subj

d_2f = sm.lambdify((g, the, the_d), d_2)
d_1f = sm.lambdify(the_d, the_d) # ( func returns itself )


# create ODE:

def dSdt(S, t): # S[0] is theta, S[1] will be omega / w
    return [
        d_1f(S[1]), # dtheta/dt = w
        d_2f(g,S[0],S[1]) # dw/dt
    ]
times = np.linspace(0,20,1000)
g = 9.81
m = 5
ans = odeint(dSdt, y0=[sm.pi,0.1], t=times) # put the ics in here

plt.plot(times, ans.T[0]) # theta against time
plt.xlabel("Time / s")
plt.ylabel("Theta / rad")
plt.show()

def xy(theta):
    x_val = l * np.sin(theta) + pivot_origin[0]
    y_val = -l * np.cos(theta) + pivot_origin[1]
    return x_val, y_val


x1, y1 = xy(ans.T[0])

def animate(i):
    x_bob, y_bob = x1[i], y1[i]
    ln1.set_data([x_bob], [y_bob])
    rod.set_data([pivot_origin[0], x_bob], [pivot_origin[1], y_bob])
    trolley.set_data([pivot_origin[0]-1, pivot_origin[0]+1], [pivot_origin[1]])

fig, ax = plt.subplots(1,1)
ax.grid()
ln1, = plt.plot([],[], 'ro', markersize = (m*5)**0.5)
rod, = ax.plot([], [], 'k-', lw=2)  # black line for the rod
trolley, = ax.plot([],[], 'k-', lw=2)
ax.set_ylim(-1,16)
ax.set_xlim(-8,8)
ax.set_aspect('equal')

ani = animation.FuncAnimation(
    fig,
    animate,
    frames=len(times),
    interval = 50,
    repeat = True
    )

plt.show()

# need to get model in state space form...



# will likely solve system numerically using scipy.integrate.solve_ivp , or a github ddeint package perhaps.:
# this requires:

# frequency against phase
# displacement ( x and y ) against time..