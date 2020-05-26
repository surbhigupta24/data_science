# -*- coding: utf-8 -*-
"""
Created on Tue May 26 13:09:51 2020

@author: SURBHI
"""

            
###############################################################################
#                             6. Stacking Classifier                          #
###############################################################################

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
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

sclf = StackingCVClassifier(classifiers = [classifier1, classifier2, classifier3, classifier4],
                            shuffle = False, use_probas = True, cv = 10, meta_classifier = SVC(probability = True))

start = timeit.default_timer()
sclf.fit(X_train, y_train)
y_pred = sclf.predict(X_test)
stop = timeit.default_timer()  

print('Time: ', stop - start)     
acc=metrics.accuracy_score(y_test, y_pred)
print(acc)
pr=metrics.precision_score(y_test, y_pred)
print(pr)
rc=metrics.recall_score(y_test, y_pred)
print(rc)
f1=f1_score(y_test, y_pred)
print(f1)
auc=roc_auc_score(y_test, y_pred)
print(auc)
mcc=matthews_corrcoef(y_test, y_pred)
print(mcc)