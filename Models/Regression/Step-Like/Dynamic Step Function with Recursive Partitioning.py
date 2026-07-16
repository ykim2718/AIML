"""
y, 2024.12.19
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# Custom step function
def step_function(x, breakpoints, values):
    y = np.zeros_like(x)
    for i, (bp, value) in enumerate(zip(breakpoints, values)):
        if i == 0:
            y[x < bp] = value
        else:
            y[(x >= breakpoints[i - 1]) & (x < bp)] = value
    y[x >= breakpoints[-1]] = values[-1]
    return y


# Objective function to minimize (mean squared error)
def objective(params, x, y):
    num_steps = (len(params) + 1) // 2
    breakpoints = params[:num_steps - 1]
    values = params[num_steps - 1:]
    y_pred = step_function(x, breakpoints, values)
    return np.mean((y - y_pred) ** 2)


# Recursive partitioning
def recursive_partitioning(x, y, min_step=2, max_step=10, tol=1e-3):
    best_score = float('inf')
    best_params = None
    best_num_steps = min_step

    for num_steps in range(min_step, max_step + 1):
        initial_breakpoints = np.linspace(x.min(), x.max(), num_steps + 1)[1:-1]
        initial_values = np.linspace(y.min(), y.max(), num_steps)
        initial_params = np.hstack([initial_breakpoints, initial_values])

        # Optimization
        result = minimize(objective, initial_params, args=(x, y), method='L-BFGS-B',
                          bounds=[(x.min(), x.max())] * (num_steps - 1) + [(y.min(), y.max())] * num_steps)
        current_score = result.fun

        # Check if the improvement is significant
        if best_score - current_score < tol:
            break

        best_score = current_score
        best_params = result.x
        best_num_steps = num_steps

    return best_params, best_num_steps


# Example data
x = np.linspace(0, 10, 100)
# y = np.piecewise(x, [x < 3, (x >= 3) & (x < 6), x >= 6], [1, 2, 3]) + np.random.normal(0, 0.2, x.shape)
y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.2, x.shape)

# Perform recursive partitioning
optimal_params, optimal_num_steps = recursive_partitioning(x, y)

# Get the optimal breakpoints and values
num_steps = (len(optimal_params) + 1) // 2
optimal_breakpoints = optimal_params[:num_steps - 1]
optimal_values = optimal_params[num_steps - 1:]

# Predict using the optimized parameters
y_pred = step_function(x, optimal_breakpoints, optimal_values)

plt.plot(x, y, label='Data')
plt.plot(x, y_pred, label='Optimized flat step function', linestyle='--')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
