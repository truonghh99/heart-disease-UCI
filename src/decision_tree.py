import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report    
from sklearn import tree
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from preprocessing import preprocessing

class decision_tree:
    x_train = preprocessing.x_train
    y_train = preprocessing.y_train
    x_test = preprocessing.x_test
    y_test = preprocessing.y_test
      
    # Decision tree with entropy 
    clf= DecisionTreeClassifier( 
                criterion = "entropy", splitter = "best",
                max_depth = 5, min_samples_leaf = 10) 
    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    accuracy = metrics.accuracy_score(y_test, y_pred)

    # visualize decision tree
    feature_cols = [2,3,8,11,15,18,31,37,39,40,43,50,51]
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data,  
                    filled=True, rounded=True,
                    special_characters=True, feature_names = feature_cols, class_names=['0','1'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    graph.write_png('heart_diseases.png')
    Image(graph.create_png())
