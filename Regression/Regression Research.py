from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import preprocessing
from pandas import read_csv, DataFrame
from sklearn.metrics import roc_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import numpy as np

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 11)


# instantiate the model and set the number of neighbors to consider to 3
reg_knn = KNeighborsRegressor(n_neighbors=3)
# fit the model using the training data and training targets
reg_knn.fit(X_train, y_train)

print("Test set predictions:\n", reg_knn.predict(X_test))
print("Test set R^2: {:.2f}".format(reg_knn.score(X_test, y_test)))

from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor(max_depth=10).fit(X_train, y_train)

print("Test set predictions:\n", tree.predict(X_test))
print("Test set R^2: {:.2f}".format(tree.score(X_test, y_test)))

check=X_test.copy()

check['Y_hat'] = tree.predict(X_test)
check['Y_test'] = y_test

check['diff'] = check['Y_test'] - check['Y_hat']

dist = np.linalg.norm(a-b)