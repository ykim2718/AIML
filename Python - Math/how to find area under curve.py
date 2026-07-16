"""
y, how to find area under curve.py, 2018.3.15
"""

import numpy as np
from scipy.integrate import trapz, simps

y = [0, 1, 0, -1, 0, 1]
x = list(range(len(y)))
print((trapz(y, x)))
print((simps(y, x)))