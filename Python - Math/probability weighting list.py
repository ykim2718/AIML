"""
probability weighting list.py
how to make probability weighting list from user weight function
"""

import numpy as np


def get_probability_weighting_list(size, function):
    a = np.arange(size)
    f = np.vectorize(function)
    r = f(a)
    w = r / r.sum()
    return w.tolist()


def sample_weighting_function(x):
    y = -1.2 * x + 10
    return y if y > 0 else 0
    # return 1.0


def demo_get_probability_weighting_list(sample_size):
    ss = abs(sample_size)
    x = np.array([1] * ss)
    y = np.random.choice(50, x.size)

    print('x=', x)
    print('y=', y, y.mean())

    w = get_probability_weighting_list(x.size, sample_weighting_function)
    print('w=', w, sum(w))
    r = y * w
    print('r=', r, r.mean(), r.sum())
    print('Note: y.mean() = r.sum() if y donot have zero')


if __name__ == '__main__':
    demo_get_probability_weighting_list(10)
