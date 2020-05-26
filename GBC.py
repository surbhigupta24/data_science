# -*- coding: utf-8 -*-
"""
Created on Tue May 26 13:05:02 2020

@author: SURBHI
"""

###############################################################################
#                          KF ON Gradient Boosting Classifier                 #
###############################################################################

from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
import statistics
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef

rf = []
f1 = []
auc = []
mcc = []
pr = []
rc = []
if __name__ == '__main__':
    start = timeit.default_timer()
    kf = KFold(n_splits=10, shuffle = True, random_state = 42)

    predicted_y = []
    expected_y = [] 
    X = pd.DataFrame(X)

    for train_index, test_index in kf.split(X, y):

        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        classifier1 = GradientBoostingClassifier(max_depth= None, n_estimators=200, learning_rate=1, random_state=30)
        start = timeit.default_timer()
        # Fit classifier
        classifier1.fit(X_train, y_train)
        predicted_y.extend(classifier1.predict(X_test))

        expected_y.extend(y_test)
        rf.append(metrics.accuracy_score(expected_y, predicted_y))
        pr.append(metrics.precision_score(expected_y, predicted_y))
        rc.append(metrics.recall_score(expected_y, predicted_y))
        f1.append(f1_score( expected_y,  predicted_y))
        auc.append(roc_auc_score(expected_y,  predicted_y))
        mcc.append(matthews_corrcoef( expected_y,  predicted_y))
        
    stop = timeit.default_timer()    
    print('Time: ', stop - start)              
    print("Accuracy: " + statistics.mean(rf).__str__())
    print("Precision: " + statistics.mean(pr).__str__())
    print("Recall: " + statistics.mean(rc).__str__())
    print("AUC: " + statistics.mean(auc).__str__())
    print("F1_Score: " + statistics.mean(f1).__str__())
    print("MCC: " + statistics.mean(mcc).__str__())
    
    