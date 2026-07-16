"""
y, 2016.4.24, Scipy - linregress.py
"""

from scipy import stats
import numpy as np
import pandas as pd

x = np.random.random(10)
y = np.random.random(10)
x = pd.DataFrame([1, 2, 3, 4])
y = 2 * x + 3
x = x[0].values.tolist()
y = y[0].values.tolist()
print(x, y)
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

print(slope, intercept, r_value ** 2)
