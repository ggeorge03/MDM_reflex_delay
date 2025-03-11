from matplotlib import animation
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class NonlinearMPC:
    def __init__(self, dynamics_func, C, Q, R, pred_hor, cont_hor, desired_state):
        """
        Initialize Nonlinear MPC controller
        
        Parameters:
        dynamics_func - Function that computes state derivatives given state and input
        C - Output matrix
        Q - State error penalty matrix
        R - Control effort penalty
        pred_hor - Prediction horizon
        cont_hor - Control horizon
        desired_state - Reference trajectory/target state
        """
        self.dynamics_func = dynamics_func  # Nonlinear dynamics function
        self.C = C
        self.Q = Q  # Cost function weight on accuracy
        self.R = R  # Cost function weight on effort
        self.f = pred_hor  # prediction horizon (steps ahead to predict)
        self.v = cont_hor  # control horizon 
        self.z = np.array(desired_state)  # reference trajectory
        
    def mpc_cost_function(self, u_sequence, dt):
        '''
        Cost function for MPC that evaluates state tracking error and control effort
        
        Parameters:
        u_sequence - Flattened control sequence to be optimized
        dt - Time step
        
        Returns:
        J - Scalar cost value ( equivalent to the total cost accumulated over the whole of the prediction horizon)
        '''
        u=u_sequence
        J = 0  # total cost
        x = self.x0.copy()  # current state
        theta_max = self.states_ub[2]
        x_max = self.states_ub[0]
        
        # Simulate system over prediction horizon. We will assume that the length of the prediciton horizon is equal in length to the control horizon.
        for k in range(self.f):
            # State error cost 
            state_error = x - self.z
            state_cost = state_error.T @ self.Q @ state_error
            
            # Control cost 
            control_cost = u[k] * self.R * u[k] # because u is just 1d we don't have to do matrix multiplication.
            
            # Special penalty for going out of bounds
            theta = x[2]  # Assuming theta is the 3rd state
            xpos = x[0]
            x_ratio = abs(xpos) / x_max
            theta_ratio = abs(theta) / theta_max

            if (theta_ratio > 0.7) | (x_ratio > 0.7):
                penalty = 100 * (theta_ratio + x_ratio)**2
            else:
                penalty = 0
            
            # Sum costs for this time step
            J += state_cost + control_cost + penalty
            
            # Propagate dynamics to next state using runge Kutta model
            x = self.runge_kutta(x, u[k], dt)
        
        return J
    
    def get_opt_control(self, x0, dt):
        """
        Compute optimal control sequence using nonlinear optimisation
        
        Parameters:
        x0 - Current state
        dt - Time step
        
        Returns:
        Optimal control sequence
        """
        bounds = [(self.input_lb, self.input_ub) for _ in range(self.f)] # bounds need to be passed in as every timestep
        # so just repeat the same values for every timestep as desired state here is independent of elapsed time.

        self.x0 = np.array(x0)
        initial_guess = np.zeros(self.f)
        
        # Use previous solution as initial guess if available
        if self.prev_solution is not None:
            # Shift previous solution and append a zero to the end 
            initial_guess = np.append(self.prev_solution[1:], 0)
            # this makes it much more computationally efficient and more likely to converge to a solution within the given time.
        
        result = minimize(
            self.mpc_cost_function, # optimise for this function.
            initial_guess,
            args=(dt),
            method="SLSQP", # Sequential least squares quadratic programming. Known to be effective for solving non-linear problems.
            bounds=bounds,
            options={
                'ftol': 1e-4, # minimisation algorithm will stop if there is less than a 0.0001 improvement in the cost func between iterations
                'maxiter': 300, # max number of iterations
                'disp': False # no output for minimisation process ( means same as verbose param in other methods)
            }
        )
        
        if result.success:
            self.prev_solution = result.x  # Store solution for warm start on next timestep
            return result.x
        else:
            raise ValueError("Optimization failed: " + result.message)
    
    def runge_kutta(self, x, u, dt):
        """
        4th order Runge-Kutta integration - using the dynamics model passed in.
        
        Parameters:
        x - Current state ( 4d vector )
        u - Control input ( 1d vector )
        dt - Time step legnth in seconds.
        
        Returns:
        Next state
        """
        k1 = self.dynamics_func(x, u)
        k2 = self.dynamics_func(x + 0.5 * k1 * dt, u)
        k3 = self.dynamics_func(x + 0.5 * k2 * dt, u)
        k4 = self.dynamics_func(x + k3 * dt, u)
        return x + (k1 + 2*k2 + 2*k3 + k4) * dt / 6
    
    def set_bounds(self, state_min, state_max, input_min=-15, input_max=15):
        """Set bounds for states and control inputs"""
        self.states_lb = state_min
        self.states_ub = state_max
        self.input_lb = input_min
        self.input_ub = input_max
    
    def sim_model(self, x0, n_iter=100, dt=0.1, delay=1):
        """
        Simulate closed-loop system with MPC controller
        
        Parameters:
        x0 - Initial state
        n_iter - Number of simulation steps
        dt - length of one time step
        delay - Control delay ( in time steps rather than seconds )
        
        Returns:
        states - Array of states over time
        control_inputs - Array of control inputs over time
        """
        states = [x0]
        control_inputs = []
        x = x0
        self.prev_solution = None  # Initialize solution for warm start
        
        for k in range(n_iter):
            print(f"timestep: {k}") # for keeping track of where you are in the process ( incase it takes ages to run )
            if k >= delay:
                u_opt = self.get_opt_control(states[k-delay], dt)
            else:
                u_opt = np.zeros(self.f)    
            u_taken = u_opt[0]
            x = self.runge_kutta(x, u_taken, dt)
            states.append(x)
            control_inputs.append(u_taken)
            
        return np.array(states), np.array(control_inputs)


def inverted_pendulum_dynamics(y, u):
    """
    Nonlinear dynamics for inverted pendulum on a cart
    
    Parameters:
    state - State vector [x, x_dot, theta, theta_dot]
    u - Control input (force applied to cart)
    
    Returns:
    State derivatives [x_dot, x_ddot, theta_dot, theta_ddot]
    """
    # System parameters
    M = 3.50    # mass of the cart (kg)
    m = 0.191  # mass of the pendulum (kg)
    g = -9.81   # gravity (m/s^2)
    l = L/2    # length of the pendulum (m)
    d=1
    
    
    # Compute nonlinear dynamics
    
    Sy = np.sin(y[2])
    Cy = np.cos(y[2])
    D = m * l * l * (M + m * (1 - Cy**2))  # Denominator of EoMs

    dy = np.zeros(4)
    dy[0] = y[1]
    dy[1] = (-m*m*l*l*g*Cy*Sy + M*l*l*(m*L*(y[2]**2)*Sy) + m*l*l*u)/ D
    dy[2] = y[3]
    dy[3] = ((m+M)*m*g*l*Sy -m*l*Cy*(m*L*(y[2]**2)*Sy)+ m*l*Cy*u)/D
    return dy


def initialise_controller():
    """Initialise the nonlinear MPC controller for inverted pendulum"""
    # Output matrix - identity for full state feedback
    C = np.eye(4)
    
    # Cost matrices
    Q = np.diag([10, 1, 100, 10])  # State error weight (high penalty on angle)
    R = 0.0001  # Control effort weight
    
    # Horizons
    pred_hor = 40  # Prediction horizon
    cont_hor = 40  # Control horizon. As stated in the non-linear mpc class, we will assume that the control and prediction horizon is equal.
    
    # Target state (upright pendulum position)
    desired_state = [0, 0, np.pi, 0]
    
    # Create controller
    controller = NonlinearMPC(
        dynamics_func=inverted_pendulum_dynamics,
        C=C,
        Q=Q,
        R=R,
        pred_hor=pred_hor,
        cont_hor=cont_hor,
        desired_state=desired_state
    )
    
    # Set bounds
    state_min = [-2, -5, -(np.pi +0.8), -8]
    state_max = [2, 5, np.pi+0.8, 8]
    controller.set_bounds(state_min, state_max, input_min=-10, input_max=10)
    
    return controller

if __name__ == "__main__":
    L = 4
    ITERATIONS = 500
    DT = 0.01
    TAU = 3 # delay ( measured in timesteps)
    # Initialise controller
    controller = initialise_controller()
    
    # Initial state (pendulum theta slight displacement)
    x0 = [0, 0, np.pi + 0.01, 0]  # x, x_dot, theta, theta_dot
    
    # Run simulation
    states, inputs = controller.sim_model(x0, n_iter=ITERATIONS, dt=DT, delay=TAU)
    x, x_dot, phi, phi_dot = states[:, 0], states[:, 1], states[:, 2], states[:, 3]
    plt.plot(phi,label="phi")
    plt.title("theta against time")
    plt.plot(inputs,label="inputs")
    plt.legend()
    plt.show()

    fig, ax = plt.subplots()

    cart_x = x
    pendulum_theta = phi
    # code for animation is mainly taken from George's code for animating LQR, with a few changes.

    # Compute pendulum end coordinates
    pendulum_x = cart_x + L * np.sin(pendulum_theta)
    pendulum_y = -L * np.cos(pendulum_theta)

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


    # Create the animation
    ani = animation.FuncAnimation(
        fig, update, frames=ITERATIONS, init_func=init, blit=True, interval=1/DT)
    plt.show()
