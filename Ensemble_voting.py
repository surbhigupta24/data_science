# -*- coding: utf-8 -*-
"""
Created on Tue May 26 13:08:53 2020

@author: SURBHI
"""

###############################################################################
#                             6. Ensemble Classifier                          #
###############################################################################
    
rf = []
f1 = []
auc = []
mcc = []
pr = []
rc = []

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn import metrics
import statistics

# Initializing Gradient Boosting classifier
classifier1 = GradientBoostingClassifier(max_depth= None, n_estimators=200, learning_rate=1, random_state=30)

# Initializing Multi-layer perceptron classifier
classifier2 = MLPClassifier(activation = "relu", alpha = 0.1, hidden_layer_sizes = (10,10,10),
                            learning_rate = "constant", max_iter = 2000, random_state = 1000)

# Initialing KNN classifier
classifier3 = KNeighborsClassifier() 

# Initializing Random Forest classifier
classifier4 = RandomForestClassifier(n_estimators = 200, criterion = 'entropy', random_state = 30)

if __name__ == '__main__':
#    start = timeit.default_timer()
    kf = KFold(n_splits=10, shuffle = True, random_state = 42)

    predicted_y = []
    expected_y = [] 
    X = pd.DataFrame(X)

    for train_index, test_index in kf.split(X, y):

        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        classifier =VotingClassifier(
                estimators=[('lr', classifier1), ('rf', classifier2), ('svc', classifier3), ('sv', classifier4)],
                voting='soft')
        start = timeit.default_timer()
        # Fit classifier
        classifier.fit(X_train, y_train)
        predicted_y.extend(classifier.predict(X_test))
        expected_y.extend(y_test)
        stop = timeit.default_timer()  
        rf.append(metrics.accuracy_score(expected_y, predicted_y))
        pr.append(metrics.precision_score(expected_y, predicted_y))
        rc.append(metrics.recall_score(expected_y, predicted_y))
        f1.append(f1_score( expected_y,  predicted_y))
        auc.append(roc_auc_score(expected_y,  predicted_y))
        mcc.append(matthews_corrcoef( expected_y,  predicted_y))
        
    
    print('Time: ', stop - start)              
    print("Accuracy: " + statistics.mean(rf).__str__())
    print("Precision: " + statistics.mean(pr).__str__())
    print("Recall: " + statistics.mean(rc).__str__())
    print("AUC: " + statistics.mean(auc).__str__())
    print("F1_Score: " + statistics.mean(f1).__str__())
    print("MCC: " + statistics.mean(mcc).__str__())
            

