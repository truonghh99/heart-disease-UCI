import numpy as np
import pandas as pd

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
#print(df)
print("Mean values: ")
print(np.mean(df, 0))