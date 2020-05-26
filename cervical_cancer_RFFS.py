# -*- coding: utf-8 -*-
"""
Created on Tue May 26 13:20:45 2020

@author: SURBHI
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

sel = SelectFromModel(RandomForestClassifier(n_estimators = 200), max_features = 10,threshold=-np.inf)
sel.fit(X, y)
sel.get_support()
selected_feat= pd.DataFrame(X).columns[(sel.get_support())]
length = len(selected_feat)
print(selected_feat)
#pd.Series(sel.estimator_,sel.feature_importances_, sel.ravel()).hist()
cols = selected_feat
X = data1.iloc[:,cols]
