"""
y, 2017.2.1, scipy - chi-squared test.py

Goodness of fit
---------------
https://en.wikipedia.org/wiki/Goodness_of_fit

Pearson's chi-squared test
--------------------------
https://en.wikipedia.org/wiki/Pearson's_chi-squared_test

Chi-squared distribution
------------------------
https://en.wikipedia.org/wiki/Chi-squared_distribution

"""

import numpy as np
import scipy.stats as ss

ax = np.array([-22, -16, -1])
ay = np.array([4130, 4450, 4400])
bx = np.array([-36, -21, -1])
by = np.array([4450, 4400, 4420])

a_slope, a_intercept, a_r, *_ = ss.linregress(ax, ay)
b_slope, b_intercept, b_r, *_ = ss.linregress(bx, by)

az = ax * a_slope + a_intercept
bz = bx * b_slope + b_intercept

a_c2, *_ = ss.chisquare(ay, az)
b_c2, *_ = ss.chisquare(by, bz)

print(a_slope, a_intercept, a_r ** 2, ':', a_c2)
print(b_slope, b_intercept, b_r ** 2, ':', b_c2)

print(ss.chi2.isf(0.8, 3))




