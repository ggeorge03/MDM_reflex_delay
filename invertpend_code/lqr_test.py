import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from ddeint import ddeint
from Eqs_of_motion import cartpend

# System parameters
m = 0.191  # Mass of pendulum
M = 4.5  # Mass of human arm
g = -9.81  # Acceleration due to gravity
d = 0.01  # Damping coefficient
s = 1  # System mode

# LQR cost matrices
Q = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 10, 0],
              [0, 0, 0, 100]])

R = np.array([[1]])


def is_unstable(sol):
    """Check if the system is unstable by checking if theta is far from pi at the end."""
    theta_final = sol[-1, 2]
    return np.abs(theta_final - np.pi) > 0.1


def run_simulation(L, tau):
    """Simulate the system for given L and tau."""
    A = np.array([[0, 1, 0, 0],
                  [0, -d/M, -m*g/M, 0],
                  [0, 0, 0, 1],
                  [0, -s*d/(M*L), -s*(m+M)*g/(M*L), 0]])

    B = np.array([[0],
                  [1/M],
                  [0],
                  [s*1/(M*L)]])

    # Solve LQR
    P = scipy.linalg.solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P

    def cartpend_lqr_with_delay(t, Y, y0):
        y_ref = np.array([0, 0, np.pi - 0.05, 0])
        y_tau = Y(t - tau) if t > tau else y0
        u_tau = (-K @ (y_tau - y_ref)).item()
        return cartpend(Y(t), m, M, L, g, d, u_tau)

    def history(t):
        return np.array([0, 0, np.pi - 0.05, 0])

    tspan = np.arange(-tau, 10, 0.02)
    sol = ddeint(lambda Y, t: cartpend_lqr_with_delay(
        t, Y, history(0)), history, tspan)
    return sol, is_unstable(sol)


# Grid search over L and tau
L_values = np.linspace(0.1, 3, 20)  # Different pendulum lengths
Tau_values = np.linspace(0.01, 0.21, 10)  # Different delay values

results = []
for L in L_values:
    for tau in Tau_values:
        _, unstable = run_simulation(L, tau)
        results.append((L, tau, unstable))

# Plot instability regions
L_vals, Tau_vals, Unstable_vals = zip(*results)
L_vals = np.array(L_vals)
Tau_vals = np.array(Tau_vals)
Unstable_vals = np.array(Unstable_vals)

plt.figure(figsize=(10, 6))
plt.scatter(L_vals[Unstable_vals], Tau_vals[Unstable_vals],
            color='red', label='Unstable')
plt.scatter(L_vals[~Unstable_vals], Tau_vals[~Unstable_vals],
            color='green', label='Stable')
plt.xlabel('Pendulum Length (L)')
plt.ylabel('Delay (tau)')
plt.title('Stability of Cart-Pendulum System')
plt.legend()
plt.grid()
plt.show()
