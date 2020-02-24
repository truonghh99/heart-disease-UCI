import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

data = np.genfromtxt('modified-data/cleveland.data')
data2 = np.genfromtxt('modified-data/hungarian.data')
data3 = np.genfromtxt('modified-data/switzerland.data')
data4 = np.genfromtxt('modified-data/long-beach-va.data')
data = np.concatenate((data,data2,data3,data4), axis = 0)

print(data[12][12])
# Print description
description = [
	[1,"id","patient identification number","NOMINAL"],
	[2,"ccf","social security number","NOMINAL"],
	[3,"age","age in years", "RATIO"],
	[4,"sex","sex (1 = male; 0 = female)", "NOMINAL"],
	[5,"painloc","chest pain location (1 = substernal; 0 = otherwise)","NOMINAL"],
	[6,"painexer","(1 = provoked by exertion; 0 = otherwise)","NOMINAL"],
	[7,"relrest","(1 = relieved after rest; 0 = otherwise)","NOMINAL"],
	[8,"pncaden","(sum of 5, 6, and 7)","INTERVAL"],
	[9,"cp","chest pain type","NOMINAL"],
	[10,"trestbps", "resting blood pressure (in mm Hg on admission to the hospital)", "RATIO"],
	[11,"htn", "htn", "UNDEFINED"],
	[12,"chol", "serum cholestoral in mg/dl", "RATIO"],
	[13,"smoke", "is or is not a smoker", "RATIO"],
	[14,"cigs", "cigarettes per day", "RATIO"],
	[15,"years", "number of years as a smoker", "RATIO"],
	[16,"fbs", "(fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)", "NOMINAL"],
	[17,"dm", "(1 = history of diabetes; 0 = no such history)", "NOMINAL"],
	[18,"famhist", "family history of coronary artery disease (1 = yes; 0 = no)", "NOMINAL"],
	[19,"restecg", "resting electrocardiographic results", "RATIO"],
	[20, "ekgmo", "month of exercise ECG reading", "RATIO"],
	[21, "ekgday", "day of exercise ECG reading","RATIO"],
	[22,"ekgyr", "year of exercise ECG reading", "RATIO"],
	[23,"dig", "(digitalis used furing exercise ECG: 1 = yes; 0 = no)", "NOMINAL"],
	[24,"prop", "(Beta blocker used during exercise ECG: 1 = yes; 0 = no)", "NOMINAL"],
	[25,"nitr", "(nitrates used during exercise ECG: 1 = yes; 0 = no)", "NOMINAL"],
	[26,"pro", "(calcium channel blocker used during exercise ECG: 1 = yes; 0 = no)", "NOMINAL"],
	[27,"diuretic", "(diuretic used used during exercise ECG: 1 = yes; 0 = no)", "NOMINAL"],
	[28,"proto", "exercise protocol", "NOMINAL"],
	[29,"thaldur", "duration of exercise test in minutes", "RATIO"],
	[30,"thaltime", "time when ST measure depression was noted", "RATIO"],
	[31,"met", "mets achieved", "RATIO"],
	[32,"thalach", "maximum heart rate achieved", "RATIO"],
	[33,"thalrest", "resting heart rate", "RATIO"],
	[34,"tpeakbps", "peak exercise blood pressure (first of 2 parts)", "RATIO"],
	[35,"tpeakbpd", "peak exercise blood pressure (second of 2 parts)","RATIO"],
	[36,"dummy", "not used", "UNDEFINED"],
	[37,"trestbpd", "resting blood pressure", "RATIO"],
	[38,"exang", "exercise induced angina (1 = yes; 0 = no)", "NOMINAL"],
	[39,"xhypo", "(1 = yes; 0 = no)", "NOMINAL"],
	[40,"oldpeak", "ST depression induced by exercise relative to rest", "RATIO"],
	[41,"slope", "the slope of the peak exercise ST segment", "ORDINAL"],
	[42,"rldv5", "height at rest", "RATIO"],
	[43,"rldv5e", "height at peak exercise", "RATIO"],
	[44,"ca", "number of major vessels (0-3) colored by flourosopy", "RATIO"],
	[45,"restckm", "irrelevant", "UNDEFINED"],
	[46,"exerckm", "irrelevant", "UNDEFINED"],
	[47,"restef", "rest raidonuclid (sp?) ejection fraction", "RATIO"],
	[48,"restwm", "rest wall (sp?) motion abnormality", "ORDINAL"],
	[49,"exeref", "exercise radinalid (sp?) ejection fraction", "RATIO"],
	[50,"exerwm", "exercise wall (sp?) motion", "RATIO"],
	[51,"thal","3 = normal; 6 = fixed defect; 7 = reversable defect","NOMINAL"],
	[52,"thalsev", "not used", "UNDEFINED"],
	[53,"thalpul", "not used", "UNDEFINED"],
	[54,"earlobe", "not used", "UNDEFINED"],
	[55,"cmo", "month of cardiac cath (sp?)", "RATIO"],
	[56,"cday", "day of cardiac cath (sp?)", "RATIO"],
	[57,"cyr", "year of cardiac cath (sp?)", "RATIO"],
	[58,"num", "diagnosis of heart disease (angiographic disease status)", "NOMINAL"],
	[59,"lmt", "vessels", "RATIO"],
	[60,"ladprox", "vessels", "RATIO"],
	[61,"laddist", "vessels", "RATIO"],
	[62,"diag", "vessels", "RATIO"],
	[63,"cxmain", "vessels", "RATIO"],
	[64,"ramus", "vessels", "RATIO"],
	[65,"om1", "vessels", "RATIO"],
	[66,"om2", "vessels", "RATIO"],
	[67,"rcaprox", "vessels", "RATIO"],
	[68,"rcadist", "vessels", "RATIO"],
	[69,"lvx1", "not used", "UNDEFINED"],
	[70,"lvx2", "not used", "UNDEFINED"],
	[71,"lvx3", "not used", "UNDEFINED"],
	[72,"lvx4", "not used", "UNDEFINED"],
	[73,"lvf", "not used", "UNDEFINED"],
	[74,"cathef", "not used", "UNDEFINED"],
	[75,"junk", "not used", "UNDEFINED"],
	[76,"name", "not used", "UNDEFINED"]]

for row in description:
    print(row[0]," : ",row[1]," : ",row[2]," : ",row[3])

# create dataframe
columns = []
for row in description:
	columns.append(row[1])
df = pd.DataFrame(data=data, columns=columns)
df_index = pd.DataFrame(data=data)

#simplify predicted value (num) from [0,4] to either 0 (absence) or 1 (presence)
for index, row in df.iterrows():
	if row[57] > 0:
		row[57] = 1

# Calculate statiscal characteristics
print("Mean values: ")
print(np.mean(df.select_dtypes(include=["float", 'int'])))
print("Median values: ")
print(np.median(df.select_dtypes(include=["float", 'int'])))
print("Standard deviation: ")
print(np.std(df.select_dtypes(include=["float", 'int'])))

# Plot 2 dimensional figures
toDraw = [2,9,11,13,18,43]
for i in toDraw:
	ax = plt.axes(projection='3d')
	ax.set_xlabel(description[i][1])
	ax.set_ylabel(description[57][1])
	ax.set_zlabel("count")
	x = df_index[i];
	y = df_index[57];
	hist, xedges, yedges = np.histogram2d(x, y, bins=10, range=[[df_index[i].min(), df_index[i].max()], [df_index[57].min(), df_index[57].max()]])
	xpos, ypos = np.meshgrid(xedges[:-1] + 0.1, yedges[:-1] + 0.1, indexing="ij")
	xpos = xpos.ravel()
	ypos = ypos.ravel()
	zpos = 0
	dx = (df_index[i].max() - df_index[i].min())/10 * np.ones_like(zpos)
	dy = (df_index[57].max() - df_index[57].min())/10 * np.ones_like(zpos)
	dz = hist.ravel()
	ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
	plt.show()

# Plot 3 dimensional figures
toDraw = [[2,9,57],[11,13,57],[18,43,57]]
for i in toDraw:
	ax = plt.axes(projection='3d')
	ax.set_xlabel(description[i[0]][1])
	ax.set_ylabel(description[i[1]][1])
	ax.set_zlabel(description[i[2]][1])
	ax.scatter3D(df_index[i[0]], df_index[i[1]], df_index[i[2]], alpha = 0.5)
	plt.show()