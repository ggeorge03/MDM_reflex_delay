import numpy as np


def cartpend(y, m, M, L, g, d, u):
    '''
    This function computes the equations of motion for an inverted pendulum balanced on someone's hand.

    Function parameters:
    y : Vector [x, x_dot, theta, theta_dot], where:
        - x: lateral position of hand.
        - x_dot: hand's velocity.
        - theta: pendulum angle (measured from the upright position).
        - theta_dot: angular velocity of the pendulum.
    m : Mass of the pendulum.
    M : Mass of the person's arm.
    L : Length of the pendulum.
    g : Acceleration due to gravity.
    d : Damping coefficient.
    u : Control input force applied.

    Returns:
    dy : Time derivatives [x_dot, x_ddot, theta_dot, theta_ddot].
    '''
    Sy = np.sin(y[2])
    Cy = np.cos(y[2])
    D = m * L * L * (M + m * (1 - Cy**2))  # Denominator of EoMs

    dy = np.zeros(4)
    dy[0] = y[1]
    dy[1] = (1/D) * (m**2 * L**2 * g * Cy * Sy + m * L**2 *
                     (y[1] * y[3] * Cy * Sy - d * y[1])) + m * L * L * (1/D) * u
    dy[2] = y[3]
    dy[3] = (1/D) * ((m + M) * m * L * Sy * (g + y[1] * y[3]) + m * L * Cy * d * y[1]
                     ) - m * L * Cy * (1/D) * u

    return dy
