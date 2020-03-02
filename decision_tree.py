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
df = pd.DataFrame(data)

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

print(training)
print(testing)