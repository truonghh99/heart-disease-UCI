import numpy as np
import pandas as pd

data = np.genfromtxt('modified-data/cleveland.data')
data2 = np.genfromtxt('modified-data/hungarian.data')
data3 = np.genfromtxt('modified-data/switzerland.data')
data4 = np.genfromtxt('modified-data/long-beach-va.data')
data = np.concatenate((data,data2,data3,data4), axis = 0)

# Select 14 needed attributes
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

# Print descriptions
descriptions = {
	"age - age in years" : "RATIO",
	"sex - sex (1 = male; 0 = female)" : "NOMINAL",
	"cp - chest pain type" : "NOMINAL",
	"trestbps - resting blood pressure" : "RATIO",
	"chol - serum cholestoral in mg/dl" : "RATIO",
	"fbs - if fasting blood sugar > 120 mg/dl" : "NOMINAL",
	"restecg - resting electrocardiographic results" : "ORDINAL",
	"thalach - maximum heart rate achieved" : "RATIO",
	"exang - exercise induced angina" : "NOMINAL",
	"oldpeak - ST depression induced by exercise relative to rest" : "RATIO",
	"slope - the slope of the peak exercise ST segment" : "ORDINAL",
	"ca - number of major vessels (0-3) colored by flourosopy" : "RATIO",
	"thal - 3 = normal; 6 = fixed defect; 7 = reversable defect" : "NOMINAL",
	"num - diagnosis of heart disease (angiographic disease status)" : "NOMINAL"
}
for name, description in descriptions.items():
    print(name," : ",description)

# Select numerical attributes for calculation
df.select_dtypes(include=["float", 'int'])

# Calculate statiscal characteristics
print("Mean values: ")
print(np.mean(df, 0))
print("Median values: ")
print(np.median(df, 0))
print("Standard deviation: ")
print(np.std(df, 0))