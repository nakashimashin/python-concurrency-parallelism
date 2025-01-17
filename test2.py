from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import optuna
from optuna.integration import OptunaSearchCV


## グリッドサーチ
# iris = load_iris()

# param_grid = dict(
#     C=[0.01, 0.01, 0.1, 1, 10, 100],
#     gamma=[0.01, 0.01, 0.1, 1, 10, 100]
# )
# print("Parameter grid:\n{}".format(param_grid))

# grid_search = GridSearchCV(SVC(), param_grid, cv=5)

# X_train, X_test, y_train, y_test = train_test_split(
#     iris.data, iris.target, random_state=0)

# grid_search.fit(X_train, y_train)

# print("test set score:{:.2f}".format(grid_search.score(X_test, y_test)))
# print("best parameters:{}".format(grid_search.best_params_))
# print("Best cross-validation score :{:.2f}".format(grid_search.best_score_))

## ランダムサーチ

# iris = load_iris()

# param_grid = dict(
#     C=[0.01, 0.01, 0.1, 1, 10, 100],
#     gamma=[0.01, 0.01, 0.1, 1, 10, 100]
# )
# print("Parameter grid:\n{}".format(param_grid))

# Randomized_search = RandomizedSearchCV(SVC(), param_grid, n_iter=15, cv=5)

# X_train, X_test, y_train, y_test = train_test_split(
#     iris.data, iris.target, random_state=0)

# Randomized_search.fit(X_train, y_train)
# print("test set score:{:.2f}".format(Randomized_search.score(X_test, y_test)))
# print("Best parameters:{}".format(Randomized_search.best_params_))
# print("Best cross-validation score :{:.2f}".format(Randomized_search.best_score_))

## Optuna ハイパーパラメータチューニング ベイズ最適化

iris = load_iris()

param_grid = dict(
    C=optuna.distributions.FloatDistribution(0.01, 100),
    gamma=optuna.distributions.FloatDistribution(0.01, 100)
)

print("Parameter grid:\n{}".format(param_grid))

Optuna_search = OptunaSearchCV(SVC(), param_grid, n_trials=100, cv=5)

X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, random_state=0)

Optuna_search.fit(X_train, y_train)
print("test set score:{:.2f}".format(Optuna_search.score(X_test, y_test)))
print("Best parameters:{}".format(Optuna_search.best_params_))
print("Best cross-validation score :{:.2f}".format(Optuna_search.best_score_))