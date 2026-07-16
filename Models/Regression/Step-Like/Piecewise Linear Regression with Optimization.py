"""
y, 2024.12.19
"""

import numpy as np
import matplotlib.pyplot as plt
import pwlf

# Example data
x = np.linspace(0, 10, 100)
y = np.piecewise(x, [x < 3, (x >= 3) & (x < 6), x >= 6], [1, 2, 3]) + np.random.normal(0, 0.2, x.shape)

# Create piecewise linear fit object
my_pwlf = pwlf.PiecewiseLinFit(x, y)

# Find the optimal number of breakpoints
breaks = my_pwlf.fit(10)  # Here 3 is the initial number of segments

# Predict the fitted values
y_hat = my_pwlf.predict(x)

# Plot the results
plt.plot(x, y, label='Data')
plt.plot(x, y_hat, label='Piecewise fit', linestyle='--')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
