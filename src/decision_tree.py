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
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from matplotlib import pyplot as plt
import seaborn as sns

# import all data
data = np.genfromtxt('../modified-data/cleveland.data')
data2 = np.genfromtxt('../modified-data/hungarian.data')
data3 = np.genfromtxt('../modified-data/switzerland.data')
data4 = np.genfromtxt('../modified-data/long-beach-va.data')
data = np.concatenate((data,data2,data3,data4), axis = 0)
df = pd.DataFrame(data)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(df.mean(), inplace=True)

# assign 66% of each type to training dataset
num_type = [0,0,0,0,0]
for index, row in df.iterrows():
	num_type[int(row[57])] += 1

num_test = [int(i * 0.66) for i in num_type]

training, testing = [],[]

for index, row in df.iterrows():
	num_test[int(row[57])] -= 1
	if (num_test[int(row[57])] >= 0):
		training.append(row)
	else:
		testing.append(row)
 
training = pd.DataFrame(training)
testing = pd.DataFrame(testing)

#split dataset in features and target variable

feature_cols = [2,3,8,11,15,18,31,37,39,40,43,50]
x_train = training[feature_cols]
y_train = training[57] #target
x_test = testing[feature_cols]
y_test = testing[57] #target

'''
#split dataset in features and target variable

x_train = training
x_test = testing

x_train = training.loc[:, 0:56]
y_train = training[57] #target
x_test = testing.loc[:, 0:56]
y_test = testing[57] #target
'''
def train_using_gini(X_train, X_test, y_train): 
  
    # Creating the classifier object 
    clf_gini = DecisionTreeClassifier(criterion = "gini", splitter = "best",
            max_depth=5, min_samples_leaf=10) 
  
    # Performing training 
    clf_gini.fit(X_train, y_train) 
    return clf_gini 
      
# Function to perform training with entropy. 
def tarin_using_entropy(X_train, X_test, y_train): 
  
    # Decision tree with entropy 
    clf_entropy = DecisionTreeClassifier( 
            criterion = "entropy", splitter = "best",
            max_depth = 5, min_samples_leaf = 10) 
  
    # Performing training 
    clf_entropy.fit(X_train, y_train) 
    return clf_entropy 

clf_gini = train_using_gini(x_train, x_test, y_train) 
clf_entropy = tarin_using_entropy(x_train, x_test, y_train)

# classification object

clf_gini = clf_gini.fit(x_train, y_train)
clf_entropy = clf_entropy.fit(x_train, y_train)

y_pred_gini = clf_gini.predict(x_test)
y_pred_entropy = clf_entropy.predict(x_test)

print("Accuracy using gini:", metrics.accuracy_score(y_test, y_pred_gini))
print("Accuracy using entropy:", metrics.accuracy_score(y_test, y_pred_entropy))

"""
# visualize decision tree
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1','2','3','4'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('heart_diseases.png')
Image(graph.create_png())
"""

def plot_multiclass_roc(clf, X_test, y_test, n_classes, figsize):
    y_score = clf.predict_proba(X_test)

    # structures
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # calculate dummies once
    y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # roc for each class
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Heart Disease Decision Tree Model performance')
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f) for label %i' % (roc_auc[i], i))
    ax.legend(loc="best")
    ax.grid(alpha=.4)
    sns.despine()
    plt.show()

plot_multiclass_roc(clf_gini, x_test, y_test, n_classes=5, figsize=(16, 10))
