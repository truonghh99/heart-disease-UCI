import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report    
from sklearn import tree
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
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

#simplify predicted value (num) from [0,4] to either 0 (absence) or 1 (presence)
for index, row in df.iterrows():
	if row[57] > 0:
		row[57] = 1

# assign 66% of each type to training dataset
num_type = [0,0]
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

feature_cols = [2,3,8,11,15,18,31,37,39,40,43,50,51]
x_train = training[feature_cols]
y_train = training[57] #target
x_test = testing[feature_cols]
y_test = testing[57] #target

# classification object
clf_1 = DecisionTreeClassifier( 
            criterion = "gini", splitter = "best",
            max_depth = 4, min_samples_leaf = 20, class_weight = {0:2, 1:1})

clf_1 = clf_1.fit(x_train, y_train)
y_pred_decision_tree = clf_1.predict(x_test)
print("Accuracy using decision tree:", metrics.accuracy_score(y_test, y_pred_decision_tree))
print("Confusion matrix on testing set: ")
print(confusion_matrix(y_test, y_pred_decision_tree))
print("Confusion matrix on training set: ")
print(confusion_matrix(y_train, clf_1.predict(x_train)))


clf_2 = svm.SVC(kernel='poly', probability = True, class_weight = {0:1.5, 1:1})
clf_2.fit(x_train, y_train)
y_pred_svm = clf_2.predict(x_test)

print("Accuracy using svm:", metrics.accuracy_score(y_test, y_pred_svm))
print("Confusion matrix on testing set: ")
print(confusion_matrix(y_test, y_pred_svm))
print("Confusion matrix on training set: ")
print(confusion_matrix(y_train, clf_2.predict(x_train)))


clf_3 = MLPClassifier()
clf_3.fit(x_train, y_train)
y_pred_nn = clf_3.predict(x_test)

print("Accuracy using neural network:", metrics.accuracy_score(y_test, y_pred_nn))
print("Confusion matrix on testing set: ")
print(confusion_matrix(y_test, y_pred_nn))
print("Confusion matrix on training set: ")
print(confusion_matrix(y_train, clf_3.predict(x_train)))


# Compare confusion matrices
dt_true_positive, svm_true_positive = [], []
dt_false_positive, svm_false_positive = [], []
dt_true_negative, svm_true_negative = [], []
dt_false_negative, svm_false_negative = [], []

index = 0
for i in y_test:
	if (i == 1):
		if (y_pred_decision_tree[index] == 1):
			dt_true_positive.append(index)
		else:
			dt_false_negative.append(index)
		if (y_pred_svm[index] == 1):
			svm_true_positive.append(index)
		else:
			svm_false_negative.append(index)
	else:
		if (y_pred_decision_tree[index] == 0):
			dt_true_negative.append(index)
		else:
			dt_false_positive.append(index)
		if (y_pred_svm[index] == 0):
			svm_true_negative.append(index)
		else:
			svm_false_positive.append(index)
	index += 1

def intersection(lst1, lst2): 
    return list(set(lst1) & set(lst2))

col = [dt_true_positive, dt_false_positive, dt_true_negative, dt_false_negative]
row = [svm_true_positive, svm_false_positive, svm_true_negative, svm_false_negative]
compare_matrix = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

for i  in range (4):
	for j in range (4):
		compare_matrix[i][j] = len(intersection(row[i], col[j]))

compare_matrix = pd.DataFrame(compare_matrix, 
								index = ['svm_true_positive', 'svm_false_positive', 'svm_true_negative', 'svm_false_negative'],
								columns = ['dt_true_positive', 'dt_false_positive', 'dt_true_negative', 'dt_false_negative'])
print('Compare matrix: ')
print(compare_matrix)

# plot roc curve
ax = plt.gca()
dt_disp = plot_roc_curve(clf_1, x_test, y_test)
svm_disp = plot_roc_curve(clf_2, x_test, y_test, ax=ax)
nn_disp = plot_roc_curve(clf_3, x_test, y_test, ax=ax)
dt_disp.plot(ax=ax, alpha=0.8)
plt.show()
