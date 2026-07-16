"""
http://glowingpython.blogspot.kr/2012/07/distribution-fitting-with-scipy.html
"""

from scipy.stats import norm
from numpy import linspace
from pylab import plot, show, hist, figure, title

# picking 150 of from a normal distrubution
# with mean 0 and standard deviation 1
samp = norm.rvs(loc=0, scale=1, size=150)

param = norm.fit(samp)  # distribution fitting

# now, param[0] and param[1] are the mean and
# the standard deviation of the fitted distribution
x = linspace(-5, 5, 100)
# fitted distribution
pdf_fitted = norm.pdf(x, loc=param[0], scale=param[1])
# original distribution
pdf = norm.pdf(x)

title('Normal distribution')
plot(x, pdf_fitted, 'r-', x, pdf, 'b-')
hist(samp, normed=1, alpha=.3)
show()
