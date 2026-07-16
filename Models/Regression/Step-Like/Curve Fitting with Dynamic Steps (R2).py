"""
y, 2024.12.19

[copilot] Let's create a Python code that fits a curve using a variable span and generates a fitting line that
looks like dynamic steps with a certain tolerance from the data. We'll optimize the R² value without smoothing
the data and without introducing look-ahead bias.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Generate example data (noisy sine wave)
np.random.seed(0)
x = np.linspace(0, 10, 500)
y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.2, x.shape)


# Function to create dynamic steps
def create_dynamic_steps(x, y, initial_span, tolerance):
    step_x = [x[0]]
    step_y = [y[0]]
    current_step_value = y[0]
    span = initial_span

    i = 0
    while i < len(x):
        start = i
        end = min(i + span, len(x))
        segment_mean = np.mean(y[start:end])

        if abs(segment_mean - current_step_value) > tolerance:  # TODO 2024.12.20, compare percent
            current_step_value = segment_mean
            step_x.append(x[start])
            step_y.append(current_step_value)

        step_x.append(x[end - 1])
        step_y.append(current_step_value)

        i = end

        # Adjust span dynamically
        if i < len(x):
            next_segment_mean = np.mean(y[end:min(end + span, len(x))])
            if abs(next_segment_mean - current_step_value) > tolerance:
                span = max(1, span // 2)  # Reduce span if the change is significant
            else:
                span = min(len(x), span * 2)  # Increase span if the change is not significant

    return np.array(step_x), np.array(step_y)


# Define the initial span and tolerance
initial_span = 1
tolerance = 0.2

# Create dynamic steps
step_x, step_y = create_dynamic_steps(x, y, initial_span, tolerance)

# Calculate the fitted values based on step function
y_pred = np.zeros_like(y)
for i in range(len(step_x) - 1):
    mask = (step_x[i] <= x) & (x < step_x[i + 1])
    y_pred[mask] = step_y[i]
y_pred[x >= step_x[-1]] = step_y[-1]

# Calculate R² value
r2 = r2_score(y, y_pred)

# Plot the data and the dynamic steps
plt.figure(figsize=(10, 5))
plt.plot(x, y, label='Noisy Data')
plt.step(step_x, step_y, label='Dynamic Steps', color='red', linewidth=2, where='post')
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Curve Fitting with Dynamic Steps (R²: {r2:.2f})')
plt.legend()
plt.show()

print(f'R² value: {r2:.2f}')
