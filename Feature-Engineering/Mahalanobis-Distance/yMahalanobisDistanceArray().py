# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 19:54:45 2014
@author: Y
"""

import numpy as np
import matplotlib.pyplot as plt
from yFunctions import MahalanobisDistanceArray

N = 256
xSeed = np.linspace(0, 10, N)
xData = xSeed + np.random.normal(0, 5, N)
yData = xSeed + np.random.normal(0, 5, N)
xyMD = np.array(MahalanobisDistanceArray(xData, yData))

fig, ax = plt.subplots()
ax.set_title("MahalanobisDistance().py")
ax.scatter(xData, yData, c='b', marker='+', s=20)

MDCut = 1.5
xyMD[xyMD < MDCut] = np.nan
a = np.vstack((xData, yData, xyMD))
a = np.ma.compress_cols(np.ma.fix_invalid(a))
if N < 64: print(a)
ax.scatter(a[0], a[1], c='r', marker='o', s=25)

# ax.set_aspect('equal')
plt.show()
