import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score 
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report    
from sklearn import tree
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import svm
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from matplotlib import pyplot as plt
import seaborn as sns
from preprocessing import preprocessing

class svm_model:
    x_train = preprocessing.x_train
    y_train = preprocessing.y_train
    x_test = preprocessing.x_test
    y_test = preprocessing.y_test

    clf = svm.SVC(kernel = "linear")
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    accuracy = metrics.accuracy_score(y_test, y_pred)
    confusion_matrix = confusion_matrix(y_test, y_pred)