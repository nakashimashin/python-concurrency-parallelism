import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier

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

seed = 42

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
kfold = model_selection.KFold(n_splits=5)
scores = {}

# ランダムフォレストで学習
rfc_clf = RandomForestClassifier(n_estimators=100, random_state=seed)
rfc_clf.fit(X_train, y_train)

# 交差検証法でモデル評価
results = model_selection.cross_val_score(rfc_clf, X_train, y_train, cv = kfold)
scores[('Random_Forest', 'train_score')] = results.mean()
scores[('Random_Forest', 'test_score')] = rfc_clf.score(X_test, y_test)
print(pd.Series(scores).unstack())