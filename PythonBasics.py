# -*- coding: utf-8 -*-
# For ML for managers course
#-----------------------------------------------------------
# General
"""
Created on Thu Oct 11 18:31:24 2018
`
@author: Shikhaliev Frank for ML for managers course
"""

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

#---------------------------------------------------------
#Get name of local data
import os
cwd = os.getcwd()
os.chdir("/../../base")


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



#---------------------------------------------------------------
#Normalization

mean = float_data[:].mean(axis=0)
float_data -= mean
std = float_data[:].std(axis=0)
float_data /= std


#Print base metrics
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