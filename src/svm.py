import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score 
from sklearn import svm
from sklearn.metrics import confusion_matrix


# import all data
data = np.genfromtxt('../modified-data/cleveland.data')
data2 = np.genfromtxt('../modified-data/hungarian.data')
data3 = np.genfromtxt('../modified-data/switzerland.data')
data4 = np.genfromtxt('../modified-data/long-beach-va.data')
data = np.concatenate((data,data2,data3,data4), axis = 0)
df = pd.DataFrame(data)


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

clf = svm.SVC()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Confusion matrix: ")
print(confusion_matrix(y_test, y_pred))