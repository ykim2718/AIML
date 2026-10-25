"""
y, 2016.7.2, Scipy - genextreme vs gumbel_r fit.py

scipy.stats.genextreme: genextreme.pdf(x, c) = exp(-exp(-x))*exp(-x), for c==0
scipy.stats.gumbel_r: gumbel_r.pdf(x) = exp(-(x + exp(-x)))
scipy.stats.gumbel_r: gumbel_r.pdf(x) = exp(-(x + exp(-x)))
"""

from scipy.stats import genextreme, gumbel_r, gumbel_l
import numpy as np

# Python, Genexetreme (Shape 0, Loc 3, Scale 2)
shape, loc, scale = 0, 3, 2
with np.errstate(invalid='ignore'):
    rvs = genextreme.rvs(shape, loc=loc, scale=scale, size=100)
    print('genextreme', genextreme.fit(rvs))
    print('gumbel_r', gumbel_r.fit(rvs))

# Minitab, LEV (Loc 3, Scale 2)
# LEV Fit --> Loc 2.014, Scale 1.374, N 10, AD 0.312, P-Value >0.250
# SEV Fit --> Loc 3.884, Scale 2.121, N 10, AD 0.647, P-Value 0.079
lev = [4.60384, 3.19784, 0.95884, 7.08785, 1.51730, 1.85600, 1.43297, 3.05610, 0.76324, 4.19548]
with np.errstate(invalid='ignore'):
    print('genextreme', genextreme.fit(lev, 0),
          'weird !!!')  # genextreme (-0.39560261036696143, 1.75109886706597, 1.1089469496993998)

print('gumbel_r', gumbel_r.fit(lev))  # gumbel_r (2.0144895625409269, 1.3738196095187054)
print('gumbel_l', gumbel_l.fit(lev))  # gumbel_l (3.8839250805920256, 2.121265251416967)
