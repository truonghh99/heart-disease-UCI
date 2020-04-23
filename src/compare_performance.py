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
from preprocessing import preprocessing
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

	x_train = preprocessing.x_train
	y_train = preprocessing.y_train
	x_test = preprocessing.x_test
	y_test = preprocessing.y_test

	# compare confusion matrices
	def compare_confusion_matrices(a, b, a_name, b_name):
		print("Compare confusion matrices between (a) - ", a_name," - and (b) - ", b_name)

		x_train = preprocessing.x_train
		y_train = preprocessing.y_train
		x_test = preprocessing.x_test
		y_test = preprocessing.y_test

		a_true_positive, b_true_positive = [], []
		a_false_positive, b_false_positive = [], []
		a_true_negative, b_true_negative = [], []
		a_false_negative, b_false_negative = [], []

		index = 0
		for i in y_test:
			if (i == 1):
				if (a.y_pred[index] == 1):
					a_true_positive.append(index)
				else:
					a_false_negative.append(index)
				if (b.y_pred[index] == 1):
					b_true_positive.append(index)
				else:
					b_false_negative.append(index)
			else:
				if (a.y_pred[index] == 0):
					a_true_negative.append(index)
				else:
					a_false_positive.append(index)
				if (b.y_pred[index] == 0):
					b_true_negative.append(index)
				else:
					b_false_positive.append(index)
			index += 1

		def intersection(lst1, lst2): 
			return list(set(lst1) & set(lst2))

		col = [a_true_positive, a_false_positive, a_true_negative, a_false_negative]
		row = [b_true_positive, b_false_positive, b_true_negative, b_false_negative]
		compare_matrix = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

		for i  in range (4):
			for j in range (4):
				compare_matrix[i][j] = len(intersection(row[i], col[j]))

		compare_matrix = pd.DataFrame(compare_matrix, 
										index = ['b_true_positive', 'b_false_positive', 'b_true_negative', 'b_false_negative'],
										columns = ['a_true_positive', 'a_false_positive', 'a_true_negative', 'a_false_negative'])
		print(compare_matrix)

	compare_confusion_matrices(decision_tree, svm_model, "decision_tree", "svm")
	compare_confusion_matrices(decision_tree, neural_network, "decision_tree", "neural network")
	compare_confusion_matrices(neural_network, svm_model, "neural network", "svm")
	# plot roc curve
	ax = plt.gca()
	dt_disp = plot_roc_curve(decision_tree.clf, x_test, y_test, ax=ax)
	svm_disp = plot_roc_curve(svm_model.clf, x_test, y_test, ax=ax)
	nn_disp = plot_roc_curve(neural_network.clf, x_test, y_test, ax=ax)
	plt.show()
