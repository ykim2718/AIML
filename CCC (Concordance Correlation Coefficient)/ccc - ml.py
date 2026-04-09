import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble      import RandomForestRegressor
from sklearn.metrics        import mean_squared_error, r2_score
from sklearn.datasets       import make_regression

def ccc_score(y_true, y_pred):
    """sklearn 호환 scorer로 사용 가능"""
    mu_t, mu_p = np.mean(y_true), np.mean(y_pred)
    var_t = np.var(y_true)
    var_p = np.var(y_pred)
    cov   = np.mean((y_true-mu_t)*(y_pred-mu_p))
    return 2*cov / (var_t + var_p + (mu_t-mu_p)**2)

from sklearn.metrics import make_scorer

# sklearn의 cross_val_score에서 CCC를 scorer로 등록
ccc_scorer = make_scorer(ccc_score, greater_is_better=True)

# 데이터 생성
X, y = make_regression(n_samples=500, n_features=20,
                        noise=15, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)

# 교차 검증으로 예측
y_pred = cross_val_predict(model, X, y, cv=5)

# 종합 평가 리포트
metrics = {
    'CCC':     round(ccc_score(y, y_pred), 4),
    'R²':      round(r2_score(y, y_pred),   4),
    'RMSE':    round(mean_squared_error(y, y_pred)**0.5, 2),
    'Bias':    round(np.mean(y_pred - y), 4),
}

# CCC를 GridSearchCV objective로 사용
from sklearn.model_selection import GridSearchCV

param_grid = {'n_estimators': [50, 100, 200],
              'max_depth':    [None, 5, 10]}

grid = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    scoring=ccc_scorer,   # ← CCC로 최적화!
    cv=3,
    n_jobs=-1
)
grid.fit(X, y)
print(f"Best CCC: {grid.best_score_:.4f}")
print(f"Best params: {grid.best_params_}")