clear all; close all; clc

% Parameters
m = 1;      % Pendulum mass
M = 5;      % Cart mass (finger mass)
L = 2;      % Pendulum length
g = -10;    % Gravity (negative for downward)
d = 1;      % Damping (simulates friction/resistance)

% Initial conditions (small perturbation from upright)
y0 = [0; 0; pi + 0.05; 0]; % [cart position, cart velocity, pendulum angle, angular velocity]

% Time span
tspan = 0:0.001:10;

% PID Controller Parameters (approximating human response)
Kp = 10000;   % Proportional gain (higher for stronger response)
Ki = 5;     % Integral gain (helps with steady-state errors)
Kd = 20;    % Derivative gain (smooths corrections)

% Noise and reaction delay
reaction_delay = 0.01;   % Delay in seconds
noise_amplitude = .01;    % Reduced noise for better stability
max_force = 10;         % Human cannot apply infinite force

% Integrate system using ODE45
[t, y] = ode45(@(t, y) cartpend_human(t, y, m, M, L, g, d, Kp, Ki, Kd, reaction_delay, noise_amplitude, max_force), tspan, y0);

% Animation
for k = 1:100:length(t)
    draw_cartpend(y(k,:), m, M, L);
end

function dydt = cartpend_human(t, y, m, M, L, g, d, Kp, Ki, Kd, reaction_delay, noise_amplitude, max_force)
    % Extract state variables
    x = y(1);       % Cart position
    x_dot = y(2);   % Cart velocity
    theta = y(3);   % Pendulum angle
    theta_dot = y(4); % Angular velocity

    % Compute error (goal is to keep theta at pi)
    error = pi - theta;
    persistent integral_error previous_error previous_time;
    
    if isempty(integral_error)
        integral_error = 0;
        previous_error = error;
        previous_time = t;
    end
    
    % Compute time step (avoid division by zero)
    dt = t - previous_time;
    if dt <= 0, dt = 1e-3; end
    
    % Compute integral and derivative terms
    integral_error = integral_error + error * dt;  
    error_derivative = (error - previous_error) / dt;
    
    % Simulate delayed reaction
    if t > reaction_delay
        u = Kp * error + Ki * integral_error + Kd * error_derivative;  % PID control
    else
        u = 0;  % No control at the start
    end

    % Add human-like noise
    u = u + noise_amplitude * randn; % Small jitter in control

    % Simulate limited hand speed (force saturation)
    u = max(-max_force, min(max_force, u));

    % Equations of motion (from inverted pendulum dynamics)
    Sy = sin(theta);
    Cy = cos(theta);
    D = M + m * (1 - Cy^2);

    dxdt = x_dot;
    dx_dotdt = (1/D) * (m * L * theta_dot^2 * Sy - d * x_dot + u);
    dthetadt = theta_dot;
    dtheta_dotdt = (1/(L * D)) * (-m * L * theta_dot^2 * Sy * Cy + (M + m) * g * Sy - d * x_dot * Cy + u * Cy);

    dydt = [dxdt; dx_dotdt; dthetadt; dtheta_dotdt];

    % Update previous values for next step
    previous_error = error;
    previous_time = t;
end
