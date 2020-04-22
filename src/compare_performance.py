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
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from matplotlib import pyplot as plt
import seaborn as sns
from neural_network import neural_network
from decision_tree import decision_tree
from svm import svm_model

class performance_evaluation:
	# accuracy
	print("Accuracy using decision_tree:", decision_tree.accuracy)
	print("Accuracy using svm:", svm_model.accuracy)
	print("Accuracy using neural network:", neural_network.accuracy)

	# confusion matrix
	print("Confusion matrix using decision_tree:")
	print(decision_tree.confusion_matrix)
	print("Confusion matrix using svm:")
	print(svm_model.confusion_matrix)
	print("Confusion matrix using neural network:")
	print(neural_network.confusion_matrix)

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
