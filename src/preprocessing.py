import pandas as pd
import numpy as np

class preprocessing:
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