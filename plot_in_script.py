from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from lcurvetools import lcurves_by_MLP_estimator
from sklearn.neural_network import MLPClassifier

# import matplotlib.pyplot as plt

X, y = make_classification(n_samples=100, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=1
)

clf = MLPClassifier(random_state=1, max_iter=300, early_stopping=True)
clf.fit(X_train, y_train)

lcurves_by_MLP_estimator(clf)
# plt.show()
