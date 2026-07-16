"""
y, numpy - pearson spearman coeff.py 2018.10.6
https://docs.scipy.org/doc/numpy/reference/generated/numpy.corrcoef.html
https://stackoverflow.com/questions/3425439/why-does-corrcoef-return-a-matrix/3425548#3425548
"""

import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([[6, 5, 4], [3, 2, 1]])
r = np.corrcoef(a, b, rowvar=False)
print(r)
print(r[1, 0])
a = a.flatten()
b = b.flatten()
r = np.corrcoef(a, b, rowvar=False)
print(r)
print(r[1, 0])
