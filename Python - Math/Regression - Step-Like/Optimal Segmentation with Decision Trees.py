"""
y, 2024.12.19
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

# Example data
x = np.linspace(0, 10, 100).reshape(-1, 1)
y = np.piecewise(x.ravel(), [x.ravel() < 3, (x.ravel() >= 3) & (x.ravel() < 6), x.ravel() >= 6], [1, 2, 3]) + np.random.normal(0, 0.2, x.shape[0])

# Optimize the number of leaves using cross-validation
best_score = -np.inf
best_leaf_nodes = None
for leaf_nodes in range(2, 10):
    tree = DecisionTreeRegressor(max_leaf_nodes=leaf_nodes)
    scores = cross_val_score(tree, x, y, cv=5)
    if scores.mean() > best_score:
        best_score = scores.mean()
        best_leaf_nodes = leaf_nodes

# Fit the best model
best_tree = DecisionTreeRegressor(max_leaf_nodes=best_leaf_nodes)
best_tree.fit(x, y)
y_pred = best_tree.predict(x)

plt.plot(x, y, label='Data')
plt.plot(x, y_pred, label='Best decision tree fit', linestyle='--')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
