"""
y, sklearn.metrics.mean_squared_error.py, 2018.10.7
http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
"""

import numpy as np
from sklearn.metrics import mean_squared_error

print('-'*8)
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([[6, 5, 4], [3, 2, 1]])
print(a, a.shape)
print(b, b.shape)
r2 = mean_squared_error(a, b, multioutput='raw_values')
print(r2)
r2 = mean_squared_error(a, b, multioutput='uniform_average')
print(r2)
r2 = mean_squared_error(a.flatten(), b.flatten(), multioutput='uniform_average')
print(r2)

