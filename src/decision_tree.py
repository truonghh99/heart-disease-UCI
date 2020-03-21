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

# classification object
clf = DecisionTreeClassifier()
# Train Decision Tree Classifer
clf = clf.fit(x_train,y_train)
# predict the response for test dataset
y_pred = clf.predict(x_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

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
