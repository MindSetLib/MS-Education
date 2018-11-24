# -*- coding: utf-8 -*-
# For ML for managers course
#-----------------------------------------------------------
# General
"""
Created on Thu Oct 11 18:31:24 2018
`
@author: Shikhaliev Frank for ML for managers course
"""

#---------------------------------------------------------
# Проверка версий ключевых библиотек
# Подготовительный этап

import sys
print("Python version:", sys.version)

import pandas as pd
print("pandas version:", pd.__version__)

import matplotlib
print("matplotlib version:", matplotlib.__version__)

import numpy as np
print("NumPy version:", np.__version__)

import scipy as sp
print("SciPy version:", sp.__version__)

import IPython
print("IPython version:", IPython.__version__)

import sklearn
print("scikit-learn version:", sklearn.__version__)

#---------------------------------------------------------------

# Simple cycle
for i in [1,2,3]:
    print(i)

# Define functions
def muliplyby2(x):
    return x*2

def apply_to_one(f):
    return f(1)

# Array allocation
my_list=[1,2,3,4]
my_tuple=('a','b','c','d')

test_zip=list(zip(my_list,my_tuple))
print(test_zip)

print (my_list)
print (my_tuple)

from collections import Counter
c=Counter([0,1,2,3,1,0,1])
#подсчет количества уникальных элементов в массиве
print(c)

#Count words in document
word_counts=Counter(document)

#--------------------------------------------------------
# ZIP
s = 'abc'
t = (15, 11, 30)
u = (-7, 23, -46)
list(zip(s,t,u))


#---------------------------------------------------------
#Get name of local data
import os
cwd = os.getcwd()
os.chdir("/../../base")

# Корректировка переменных path

import os
os.environ["PATH"] += os.pathsep + 'D:/Program Files (x86)/Graphviz2.38/bin/'

# Read data
import csv
# First
data = []
with open("C:\\gt.csv") as f:
    for line in f:
        data.append([x for x in line.split()])

#Second
file = open("\\gt.csv", "r")
data = [map(float, line.split(",")) for line in file]


#Third
with open("C:\\gt.csv") as f:
    data = [map(float, row) for row in csv.reader(f, delimiter=',')]

#Fourth - most popular
f = open(u'C:\\gt.csv')
data = pd.read_csv(f, sep=',')

#write data

fileW=open('MMW.csv','w')
DFW.to_csv(fileW)


#--------------------------------------------------------------
#Numpy
#--- numpy array stat
from scipy import stats
stats.describe(data_np)

#Удаление нулов из списка
data_K1 = [x.upper() for x in data['K1'] if not pd.isnull(x)]

#--------------------------------------------------------------
#Visualisation

#Matplotlib connect - work in notebook
from matplotlib import pyplot as plt
%matplotlib inline

# обзорная матрица

pd.plotting.scatter_matrix(dataframe, c=y_train, figsize=(15, 15),
                           marker='o', hist_kwds={'bins': 20}, s=60,
                           alpha=.8, stratify=0.5) #, cmap=mglearn.cm3



#---------------------------------------------------------------
#Normalization

mean = float_data[:].mean(axis=0)
float_data -= mean
std = float_data[:].std(axis=0)
float_data /= std

# Compute the minimum value per feature on the training set
min_on_training = X_train.min(axis=0)
# Compute the range of each feature (max - min) on the training set
range_on_training = (X_train - min_on_training).max(axis=0)

# вариант нормализации на тесте
X_train_scaled = (X_train - min_on_training) / range_on_training
print("Minimum for each feature\n", X_train_scaled.min(axis=0))
print("Maximum for each feature\n", X_train_scaled.max(axis=0))

# Повторяем для теста
X_test_scaled = (X_test - min_on_training) / range_on_training

#---------------------------------------------------------------
# Categorical features encoding


#----------------------------------------------------------------
# Разделение выборок
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)

# ---------------------------------------------------------------
# Возможные классификаторы

#байесовский классификатор

from sklearn.naive_bayes import GaussianNB
nbc = GaussianNB()
nbc.fit(X_train, y_train)

# ---------------
# knn
from sklearn.neighbors import KNeighborsClassifier
clf_knn = KNeighborsClassifier(n_neighbors=3)

training_accuracy = []
test_accuracy = []
# try n_neighbors from 1 to 10
neighbors_settings = range(1, 11)

for n_neighbors in neighbors_settings:
    # build the model
    clf_knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf_knn.fit(X_train, y_train)
    # record training set accuracy
    training_accuracy.append(clf_knn.score(X_train, y_train))
    # record generalization accuracy
    test_accuracy.append(clf_knn.score(X_test, y_test))

plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()

from sklearn.linear_model import LogisticRegression

#----------------
# logistic regression

logreg = LogisticRegression().fit(X_train, y_train)
print("Training set score: {:.3f}".format(logreg.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg.score(X_test, y_test)))

# варианты регуляризации
# чем меньше C, тем больше регуляризация

logreg100 = LogisticRegression(C=100).fit(X_train, y_train)
print("Training set score: {:.3f}".format(logreg100.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg100.score(X_test, y_test)))

logreg001 = LogisticRegression(C=0.01).fit(X_train, y_train)
print("Training set score: {:.3f}".format(logreg001.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg001.score(X_test, y_test)))

#----------------

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))

tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(X_train, y_train)

print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))


from sklearn.tree import export_graphviz
export_graphviz(tree, out_file="tree.dot", class_names=features_names,
                feature_names=cancer.feature_names, impurity=False, filled=True)

#----------------
# визуализация дерева
import graphviz

with open("tree.dot") as f:
    dot_graph = f.read()
display(graphviz.Source(dot_graph))

#----------------
#Random forest

from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=5, random_state=2)
forest.fit(X_train, y_train)

#----------------
# Gradient boosting
from sklearn.ensemble import GradientBoostingClassifier

gbrt = GradientBoostingClassifier(learning_rate=0.01,random_state=0)
gbrt.fit(X_train, y_train)

print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test)))
print(gbrt.predict_proba(X_test[:16]))
#----------------
# SVM - важна предобработка
from sklearn.svm import SVC

svc = SVC()
svc.fit(X_train, y_train)

print("Accuracy on training set: {:.2f}".format(svc.score(X_train, y_train)))
print("Accuracy on test set: {:.2f}".format(svc.score(X_test, y_test)))

#----------------
# MLPClassifier - многослойный перцептрон от sklearn - важна предобработка
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[10])
mlp.fit(X_train, y_train)

# -------------------------------------------------------------
# возможные регрессоры
from sklearn.neighbors import KNeighborsRegressor

# instantiate the model and set the number of neighbors to consider to 3
reg_knn = KNeighborsRegressor(n_neighbors=3)
# fit the model using the training data and training targets
reg_knn.fit(X_train, y_train)

print("Test set predictions:\n", reg_knn.predict(X_test))
print("Test set R^2: {:.2f}".format(reg_knn.score(X_test, y_test)))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
# create 1,000 data points, evenly spaced between -3 and 3
line = np.linspace(-3, 3, 1000).reshape(-1, 1)
for n_neighbors, ax in zip([1, 3, 9], axes):
    # make predictions using 1, 3, or 9 neighbors
    reg_knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    reg_knn.fit(X_train, y_train)
    ax.plot(line, reg_knn.predict(line))
    ax.plot(X_train, y_train, '^', markersize=8) # , c=mglearn.cm2(0)
    ax.plot(X_test, y_test, 'v', markersize=8) #  , c=mglearn.cm2(1)

    ax.set_title(
        "{} neighbor(s)\n train score: {:.2f} test score: {:.2f}".format(
            n_neighbors, reg_knn.score(X_train, y_train),
            reg_knn.score(X_test, y_test)))
    ax.set_xlabel("Feature")
    ax.set_ylabel("Target")
axes[0].legend(["Model predictions", "Training data/target",
                "Test data/target"], loc="best")

#
from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor(max_depth=3).fit(X_train, y_train)

# --------------------------------------------------------------
# Вычисление и вывод базовых метрик
# Print base metrics

from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

def ModelDemo_MSLib(model):
    print(model)
    print("precision:",metrics.precision_score(y_test, model.predict(X_test)))
    print("recall:",metrics.recall_score(y_test, model.predict(X_test)))
    print("roc_auc:",roc_auc_score(y_test, model.predict(X_test)))
    print("gini:",2*roc_auc_score(y_test, model.predict(X_test))-1)
    print ("accuracy:",accuracy_score(y_test, model.predict(X_test)))


#----------------------------------------------------------------
#Simple NN Keras
import keras
keras.__version__

from keras import backend as K
K.set_image_dim_ordering('tf')

from keras import layers
from keras import models

#Convolutional layers
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.summary()

# Denselayers
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()


#mnist example

from keras.datasets import mnist
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=64)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_acc)


#-------------------------------------------------------------
# connect to audio device

import speech_recognition as sr

r = sr.Recognizer()

with sr.Microphone() as source:
    print('Question')
    audio = r.listen(source)

question = r.recognize_google(audio, language="ru-RU")
print(question)