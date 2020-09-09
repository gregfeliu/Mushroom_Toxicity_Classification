# Functions used in mushroom_classification_project.ipynb
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, chi2
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression


def get_chi_values(y_dum_train, X_dum_train):
    """ 
    Gets Chi values in order to determine if the values are dependent or independent of the edible and poisonous classes.
    """
    probability = 0.95
    all_column_chi = {}
    for item in X_dum_train.columns:
        cont_table = pd.crosstab(index = y_dum_train, columns = X_dum_train[item])
        stat, p, dof, expected = chi2_contingency(cont_table)
        alpha = 1.0 - probability
        if p <= alpha:
            all_column_chi[item] = [f"probability = {round(p, 3)}", 'Dependent (reject H0)']
        else:
            all_column_chi[item] = [f"probability = {round(p, 3)}", 'Independent (fail to reject H0)']
    return all_column_chi

def only_improving_features(X_dummy_data, y_dummy_data):
    """
    Runs and prints the all features of a logistic regression.
    """
    X_dum_train, X_dum_test, y_dum_train, y_dum_test = train_test_split(X_dummy_data, y_dummy_data, random_state=1)
    logreg = LogisticRegression(C=10, max_iter=50, random_state = 5, penalty='l1', solver='liblinear')
    num_of_improving_roc_features = 0
    for i in range(0, len(X_dum_train.columns)):
        column = X_dum_train.columns[i]
        logreg.fit(X_dum_train[[column]], y_dum_train)
        dum_predict_log = logreg.predict(X_dum_test[[column]])
        fpr, tpr, thresholds = roc_curve(y_dum_test, dum_predict_log)
        if len(fpr) > 2:
            print_roc_curve(y_dum_test, dum_predict_log)
            num_of_improving_roc_features += 1
    return num_of_improving_roc_features
        
def print_roc_curve(y_test, predictions):
    fpr, tpr, thresholds = roc_curve(y_test, predictions)
    plt.plot(fpr, tpr)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title('ROC curve for Mushroom Classification')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.grid(True)
