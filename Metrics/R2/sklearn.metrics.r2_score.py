"""
y, sklearn.metrics.r2_score.py, 2018.10.6
http://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score
"""

import numpy as np
from sklearn.metrics import r2_score

print('-'*8)
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([[6, 5, 4], [3, 2, 1]])
print(a, a.shape)
print(b, b.shape)
r2 = r2_score(a, b, multioutput='raw_values')
print(r2)
r2 = r2_score(a, b, multioutput='uniform_average')
print(r2)
r2 = r2_score(a, b, multioutput='variance_weighted')
print(r2)

print('-'*8)
a = a.transpose()
b = b.transpose()
print(a, a.shape)
print(b, b.shape)
r2 = r2_score(a, b, multioutput='raw_values')
print(r2)

