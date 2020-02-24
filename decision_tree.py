import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report     

# Create dataframe
data = np.genfromtxt('modified-data/cleveland.data')
data2 = np.genfromtxt('modified-data/hungarian.data')
data3 = np.genfromtxt('modified-data/switzerland.data')
data4 = np.genfromtxt('modified-data/long-beach-va.data')
data = np.concatenate((data,data2,data3,data4), axis = 0)
needed = [3,4,9,10,12,16,19,32,38,40,41,44,51,58]
index, deleted = 0, 0
for col in range(1,77):
	if index == 14 or col != needed[index]:
		data = np.delete(data, col - deleted - 1, 1)
		deleted += 1
	else:
		index += 1
columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
df = pd.DataFrame(data=data, columns=columns)

print(df)
# Define target attribute (#13)
X = df.values[:, 0:12]
Y = df.values[:, 13]

# Create training set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 100)

# Build decision tree
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100, max_depth=5, min_samples_leaf=5) 
clf_gini.fit(X_train, y_train)

print("Result using gini index: ")
y_pred_gini = clf_gini.predict(X_test)
print(y_pred_gini)