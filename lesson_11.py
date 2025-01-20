import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import numpy as np

# seed = 42

# X, y = load_breast_cancer(return_X_y=True)
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
# kfold = model_selection.KFold(n_splits=5)
# scores = {}

# # ロジスティック回帰
# lr_clf = LogisticRegression(solver='lbfgs', max_iter=10000)

# # 決定木
# dtc_clf = DecisionTreeClassifier(random_state=seed)

# # サポートベクターマシン(SVM)
# svm_clf = SVC(probability=True, random_state=seed)

# # アンサンブル学習
# estimators = [('lr', lr_clf), ('dtc', dtc_clf), ('svm', svm_clf)]

# vote_clf=VotingClassifier(estimators=estimators, voting='hard')
# vote_clf.fit(X_train, y_train)

# results = model_selection.cross_val_score(vote_clf, X_train, y_train, cv = kfold)
# scores[('Voting', 'train_score')] = results.mean()
# scores[('Voting', 'test_score')] = vote_clf.score(X_test, y_test)
# print(pd.Series(scores).unstack())

## 2.2.1 バギング

# seed = 42

# X, y = load_breast_cancer(return_X_y=True)
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
# kfold = model_selection.KFold(n_splits=5)
# scores = {}

# # ランダムフォレストで学習
# rfc_clf = RandomForestClassifier(n_estimators=100, random_state=seed)
# rfc_clf.fit(X_train, y_train)

# # 交差検証法でモデル評価
# results = model_selection.cross_val_score(rfc_clf, X_train, y_train, cv = kfold)
# scores[('Random_Forest', 'train_score')] = results.mean()
# scores[('Random_Forest', 'test_score')] = rfc_clf.score(X_test, y_test)
# print(pd.Series(scores).unstack())

## ブースティング

# seed = 42

# X, y = load_breast_cancer(return_X_y=True)
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
# kfold = model_selection.KFold(n_splits=5)
# scores = {}

# lgbm_clf = lgb.LGBMClassifier(boosting_type='gbdt', max_depth=1, random_state=0)
# lgbm_clf.fit(X_train, y_train)

# # 交差検証法でモデル評価
# results = model_selection.cross_val_score(lgbm_clf, X_train, y_train, cv=kfold)
# scores[('lightgbm', 'train_score')] = results.mean()
# scores[('lightgbm', 'test_score')] = lgbm_clf.score(X_test, y_test)
# print(pd.Series(scores).unstack())

## 2.2.3 スタッキング

seed = 42

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=seed)
kfold = model_selection.KFold(n_splits=5)
scores = {}

# 第一段階
# ロジスティック回帰
lr_clf = LogisticRegression(solver='lbfgs', max_iter=10000)
lr_clf.fit(X_train, y_train)
lr_pred = lr_clf.predict(X_valid)
lr_test_pred = lr_clf.predict(X_test)

# 決定木
dtc_clf = DecisionTreeClassifier(random_state=seed)
dtc_clf.fit(X_train, y_train)
dtc_pred = dtc_clf.predict(X_valid)
dtc_test_pred = dtc_clf.predict(X_test)

# サポートベクターマシン(SVM)
svm_clf = SVC(probability=True, random_state=seed)
svm_clf.fit(X_train, y_train)
svm_pred = svm_clf.predict(X_valid)
svm_test_pred = svm_clf.predict(X_test)

# 第1段階の予測値を積み重ねる
stack_pred = np.column_stack((lr_pred, dtc_pred, svm_pred))

# 第2段階モデルの学習
meta_model = LogisticRegression(solver='lbfgs', max_iter=10000)
meta_model.fit(stack_pred, y_valid)

# 各モデルの検証データを積み重ねる
stack_test_pred = np.column_stack((lr_test_pred, dtc_test_pred, svm_test_pred))

# 交差検証法でモデル評価
results = model_selection.cross_val_score(meta_model, stack_test_pred, y_test, cv=kfold)
scores[('Weight_Average_voting', 'train_score')] = results.mean()
scores[('Weight_Average_voting', 'test_score')] = meta_model.score(stack_test_pred, y_test)
print(pd.Series(scores).unstack())