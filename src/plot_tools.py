from sklearn.metrics import plot_roc_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from matplotlib import pyplot as plt
import seaborn as sns

class plot_tools:
	def plot_multiclass_roc(clf, X_test, y_test, n_classes, figsize):
	    y_score = clf.predict_proba(X_test)

	    # structures
	    fpr = dict()
	    tpr = dict()
	    roc_auc = dict()

	    # calculate dummies once
	    y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
	    for i in range(n_classes):
	        fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_score[:, i])
	        roc_auc[i] = auc(fpr[i], tpr[i])

	    # roc for each class
	    fig, ax = plt.subplots(figsize=figsize)
	    ax.plot([0, 1], [0, 1], 'k--')
	    ax.set_xlim([0.0, 1.0])
	    ax.set_ylim([0.0, 1.05])
	    ax.set_xlabel('False Positive Rate')
	    ax.set_ylabel('True Positive Rate')
	    ax.set_title('Heart Disease Decision Tree Model performance')
	    for i in range(n_classes):
	        ax.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f) for label %i' % (roc_auc[i], i))
	    ax.legend(loc="best")
	    ax.grid(alpha=.4)
	    sns.despine()
	    plt.show()