"""
http://stackoverflow.com/questions/21420792/exponential-curve-fitting-in-scipy
"""

from pylab import *
from scipy.optimize import curve_fit

x = np.array([399.75, 989.25, 1578.75, 2168.25, 2757.75, 3347.25, 3936.75, 4526.25, 5115.75, 5705.25])
y = np.array([109, 62, 39, 13, 10, 4, 2, 0, 1, 2])


def func(x, a, b, c, d):
    return a * np.exp(-c * (x - b)) + d


popt, pcov = curve_fit(func, x, y, [100, 400, 0.001, 0])
print(popt)
print(pcov)

plot(x, y, marker='.', c='0.25')
x = linspace(400, 6000, 10000)
plot(x, func(x, *popt))
show()
