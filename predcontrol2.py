# I am not leaving here until i come up with a solution that works for the mpc.
import numpy as np
from scipy.optimize import minimize
class MPC:


    def __init__(self,A,B,C,Q,R,pred_hor, cont_hor, desired_state):
        self.A = A # state transition matric
        self.B = B # input matrix
        self.C = C
        self.Q = Q # cost function weight on accuracy
        self.R = R # cost function weight on effort
        self.f = pred_hor # prediction horizon ( no. of steps ahead to go and predict )
        self.v = cont_hor # control horizon - shouldn't this be for the whole estimated itm
        self.z = np.array(desired_state) # reference trajectory - for this we can treat it as constant but could be adjusted to vary with time.

    def set_bounds(self, min, max):
        '''
        min and max should have the same dimensions as x
        ie if x = [x1,x2,x3..etc], xmin should be [x1_min, x2_min etc...]
        '''
        self.states_lb = min
        self.states_ub  = max

    def mpc_cost_function(self,u_sequence):
        """
        Cost function for MPC that heavily penalizes theta deviations.
        
        Returns:
            J: Scalar cost value
        """
        # Reshape control sequence
        u = u_sequence.reshape(self.f, -1)
        
        # Initialize total cost and current state
        J = 0
        x = self.x0.copy()
        theta_max = self.states_ub[2]
        # Simulate system over prediction horizon
        for k in range(self.f):
            # State error cost (quadratic)
            state_error = x - self.z
            state_cost = state_error.T @ self.Q @ state_error
            
            # Control cost (quadratic)
            control_cost = u[k].T * self.R * u[k]
            
            # Special theta penalty (exponential barrier function)
            theta = x[2]  # Assuming theta is the 3rd state
            theta_ratio = abs(theta) / theta_max
            # Exponential barrier that grows extremely fast as theta approaches limits
            if theta_ratio > 0.7:
                barrier_strength = 2000.0  # Higher means stronger constraint enforcement
                theta_penalty = 10000*theta_ratio
            else:
                theta_penalty = 0
            
            # Sum costs for this time step
            J += state_cost + control_cost + theta_penalty
            
            # Propagate dynamics to next state
            x = self.dynamics_model(x, u[k])
        
        return J
    
    def get_opt_control(self, x0):
        self.x0 = np.array(x0)
        initial_guess = np.ones(self.f)
        u = minimize(self.mpc_cost_function, initial_guess, method="SLSQP")
        if u.success:
            return u.x
        else:
            raise ValueError("Optimization failed: " + u.message)

    def dynamics_model(self, x, u):
        #print(f"{x=}, {u=}")
        state_change = self.A @ x
        control_change = self.B * u
        #print(f"{state_change.shape=}, {control_change.flatten().shape=}")
        return state_change + control_change.flatten()
    
        # Runge-Kutta 4th order integration to propagate to next state.
    def runge_kutta(self, x, u, dt):
        k1 = self.dynamics_model(x, u)
        k2 = self.dynamics_model(x + 0.5 * k1 * dt, u)
        k3 = self.dynamics_model(x + 0.5 * k2 * dt, u)
        k4 = self.dynamics_model(x + k3 * dt, u)
        return x + (k1 + 2*k2 + 2*k3 + k4) * dt / 6
    
    def sim_model(self,x0,n_iter=100, dt=0.1):
        states = [x0]
        control_inputs = []
        x = x0
        for _ in range(n_iter):
            u_opt = self.get_opt_control(x)
            u_opt = [0]
            u_taken = np.array([u_opt[0]])
            x = self.runge_kutta(x, u_taken,dt)
            states.append(x)
            control_inputs.append(u_opt[0])
        return np.array(states), np.array(control_inputs)
        

def initialise_variables():
    M = 1.0   # mass of the cart (kg)
    m_p = 0.1 # mass of the pendulum (kg)
    g = 9.81  # gravity (m/s^2)
    l = 5.0   # length of the pendulum (m)

    def initialise_matricies():
        A = [
            [0, 1, 0, 0],
            [0, 0, (m_p*g)/l, 0],
            [0, 0, 0, 1],
            [0, 0, (M+m_p)/M, 0]
            ]

        B = [
            [0],
            [1/M],
            [0],
            [-1 / M*l]
            ]
        C = np.eye(4)

        Q = np.diag([1,1,100,10])
        
        R = 0.0001
        return A,B,C,Q,R
    return initialise_matricies()
    


    
if __name__ == "__main__":

    # linearise system around equilibrium point theta = 0.
    matricies = initialise_variables()
    mpc = MPC(*matricies,pred_hor=20, cont_hor= 0 ,desired_state=[0,0,0,0])
    mpc.set_bounds([-5,-5,-np.pi/2, -2], [5,5,np.pi/2, 2])
    x0 = np.array([0,0,0.1,0])
    states, inputs = mpc.sim_model(x0)