import numpy as np
from scipy.optimize import minimize
from anim import animPendulum
import matplotlib.pyplot as plt
class MPC:
    # this is pretty much only useable for an inverted pendulum model atm, but can be adapted for other models.

    # The higher the prediction horizon is, the more optimal solution that will be produced but it will take exponentially longer
    # also its diminishing returns as you increase it.

    def __init__(self,A,B,C,Q,R,pred_hor, cont_hor, desired_state):
        self.A = A # state transition matric
        self.B = B # input matrix
        self.C = C
        self.Q = Q # cost function weight on accuracy
        self.R = R # cost function weight on effort
        self.f = pred_hor # prediction horizon ( no. of steps ahead to go and predict )
        self.v = cont_hor # control horizon 
        self.z = np.array(desired_state) # reference trajectory - for this we can treat it as constant but could be adjusted to vary with time.


    def mpc_cost_function(self,u_sequence,dt):
        '''
        Cost function for MPC that heavily penalizes theta deviations.
        If it didn't penalise theta as much it thinks the most efficient way is to repeatedly swing it up and down.
        
        J: Scalar cost value
        '''
        # Reshape control sequence
        u = u_sequence.reshape(self.f, -1)
        
        J = 0 # total cost
        x = self.x0.copy() # current state
        theta_max = self.states_ub[2]
        x_max = self.states_ub[0]
        # Simulate system over prediction horizon
        for k in range(self.f):
            # State error cost 
            state_error = x - self.z
            state_cost = state_error.T @ self.Q @ state_error
            
            # Control cost 
            control_cost = u[k].T * self.R * u[k] # scalar multiplication becuase control input is scalar.
            
            # Special penalty for going out of bounds.
            theta = x[2]  # Assuming theta is the 3rd state
            xpos = x[0]
            x_ratio = abs(xpos) / x_max
            theta_ratio = abs(theta) / theta_max

            if (theta_ratio > 0.7 )|(x_ratio > 0.7):
                
                penalty = 100*(theta_ratio+x_ratio)**2
            else:
                penalty = 0
            
            # Sum costs for this time step
            J += state_cost + control_cost + penalty
            
            # Propagate dynamics to next state
            x = self.runge_kutta(x, u[k],dt)
        
        return J
    
    def get_opt_control(self, x0,dt):

        bounds = [(self.input_lb, self.input_ub) for _ in range(self.f)]

        self.x0 = np.array(x0)
        initial_guess = np.ones(self.f)*0.5
        result = minimize(
                    self.mpc_cost_function,
                    initial_guess,
                    args=(dt),
                    method="SLSQP",
                    bounds=bounds,
                    options={
                        'ftol': 1e-4,
                        'maxiter': 300, # increase number of iterations
                        'disp': False
                    }
                )
        if result.success:
            return result.x
        else:
            raise ValueError("Optimization failed: " + result.message)

    def dynamics_model(self, x, u):
        
        state_change = self.A @ x
        control_change = self.B * u
        
        return state_change + control_change.flatten()
    
        # Runge-Kutta 4th order integration to propagate to next state.
    def runge_kutta(self, x, u, dt):
        '''
        Apparantly Runge-kutta integration is more accurate than Euler
        Euler integration being x_n+1 = x_n + x_dot_n * dt
        '''
        k1 = self.dynamics_model(x, u)
        k2 = self.dynamics_model(x + 0.5 * k1 * dt, u)
        k3 = self.dynamics_model(x + 0.5 * k2 * dt, u)
        k4 = self.dynamics_model(x + k3 * dt, u)
        return x + (k1 + 2*k2 + 2*k3 + k4) * dt / 6
    
    def set_bounds(self, state_min, state_max, input_min=-10, input_max=10):

        self.states_lb = state_min
        self.states_ub = state_max
        self.input_lb = input_min
        self.input_ub = input_max
    
    def sim_model(self,x0,n_iter=100, dt=0.1,delay=1):
        states = [x0]
        control_inputs = []
        x = x0
        for k in range(n_iter):
            print(f"timestep: {k}")
            if k > delay:
                u_opt = self.get_opt_control(states[k-delay],dt)
            else: u_opt = [0]
            #u_opt = [0] - uncomment if you want to run with no control input.
            u_taken = np.array([u_opt[0]])
            x = self.runge_kutta(x, u_taken,dt)
            states.append(x)
            control_inputs.append(u_opt[0])
        return np.array(states), np.array(control_inputs)
        

def initialise_variables():
    M = 1.0   # mass of the cart (kg)
    m = 0.1 # mass of the pendulum (kg)
    g = 9.81  # gravity (m/s^2)
    l = 1.0   # length of the pendulum (m)
    def initialise_matricies():
        A = np.array([
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, g/l, 0]
    ])
    
        B = np.array([
            [0],
            [1/m],
            [0],
            [-1/(m*l)]
        ])
        C = np.eye(4)

        Q = np.diag([1,1,100,10])
        
        R = 0.0001
        return A,B,C,Q,R
    return initialise_matricies()
    


    
if __name__ == "__main__": # for testing..

    # linearise system around equilibrium point theta = 0.
    matricies = initialise_variables()
    mpc = MPC(*matricies,pred_hor=20, cont_hor= 0 ,desired_state=[0,0,0,0])
    mpc.set_bounds([-5,-5,-np.pi/2, -2], [5,5,np.pi/2, 2])
    x0 = np.array([0,0,0.01,0])
    states, inputs = mpc.sim_model(x0,n_iter=1000,dt=0.1,delay=1) # 1ms
    x, x_dot, theta, theta_dot = states[:, 0], states[:, 1], states[:, 2], states[:, 3]
    pend=animPendulum(theta, x, 1, 5, m=4)
    plt.plot(theta,label="theta")
    plt.title("theta against time")
    plt.plot(inputs,label="inputs")
    plt.legend()
    plt.show()
