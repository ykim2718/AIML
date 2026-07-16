"""
y, scipy.stats - person, rank corr coeff.py, 2018.10.6
https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.spearmanr.html
"""

import numpy as np
import scipy.stats as ss

a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([[6, 5, 4], [3, 2, 1]])
a = a.flatten()
b = b.flatten()
r = ss.spearmanr(a, b)
print('spearman r =', r, r[0])
r = ss.pearsonr(a, b)
print('pearson r =', r, r[0])
