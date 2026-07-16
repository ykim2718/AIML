"""
y, smooth data by moving average box.py, 2018.7.18
https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-in-the-right-way
"""

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,2*np.pi,100)
y = np.sin(x) + np.random.random(100) * 0.8

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

plt.plot(x, y,'o')
plt.plot(x, smooth(y, 3), 'r-', lw=2)
plt.plot(x, smooth(y, 19), 'g-', lw=2)
plt.show()