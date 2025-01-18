import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

seed = 42

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
kfold = model_selection.KFold(n_splits=5)
scores = {}

# ロジスティック回帰
lr_clf = LogisticRegression(solver='lbfgs', max_iter=10000)

# 決定木
dtc_clf = DecisionTreeClassifier(random_state=seed)

# サポートベクターマシン(SVM)
svm_clf = SVC(probability=True, random_state=seed)

# アンサンブル学習
estimators = [('lr', lr_clf), ('dtc', dtc_clf), ('svm', svm_clf)]

vote_clf=VotingClassifier(estimators=estimators, voting='hard')
vote_clf.fit(X_train, y_train)