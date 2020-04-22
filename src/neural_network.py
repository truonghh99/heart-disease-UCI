import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report    
from sklearn.neural_network import MLPClassifier
from preprocessing import preprocessing

class neural_network:
    x_train = preprocessing.x_train
    y_train = preprocessing.y_train
    x_test = preprocessing.x_test
    y_test = preprocessing.y_test

    clf = MLPClassifier()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    accuracy = metrics.accuracy_score(y_test, y_pred)
