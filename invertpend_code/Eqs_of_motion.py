import numpy as np


def cartpend(y, m, M, L, g, d, u):
    Sy = np.sin(y[2])
    Cy = np.cos(y[2])
    D = m * L * L * (M + m * (1 - Cy**2))  # checked

    dy = np.zeros(4)
    dy[0] = y[1]
    dy[1] = (1 / D) * (m**2 * L**2 * g * Cy * Sy + m * L**2 *
                       (y[1] * y[3] * Cy * Sy - d * y[1])) + m * L**2 * (1 / D) * u
    dy[2] = y[3]
    dy[3] = (1 / D) * ((m + M) * m * L * Sy * (g + y[1] * y[3]) + m * L * Cy *
                       (m * L * y[3]**2 * Sy - d * y[1])) - m * L * Cy * (1 / D) * u

    # dy[0] = y[1]
    # dy[1] = (1/D) * (m**2 * L**2 * g * Cy * Sy + m * L**2 *
    #                  (y[1] * y[3] * Cy * Sy - d * y[1])) + m * L * L * (1/D) * u
    # dy[2] = y[3]
    # dy[3] = (1/D) * ((m + M) * m * L * Sy * (g + y[1] * y[3]) + m * L * Cy * d * y[1]
    #                  ) - m * L * Cy * (1/D) * u

    return dy
