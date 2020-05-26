# -*- coding: utf-8 -*-
"""
Created on Tue May 26 13:15:34 2020

@author: SURBHI
"""

#########################Extra_trees#########################

from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_)

feat_importances = pd.Series(model.feature_importances_, index=data1.iloc[:,:30].columns)
feat_importances.nlargest(15).plot(kind='barh')
plt.show()
imp = feat_importances.nlargest(15)
data1.columns

#Selected_Columns for most of the targets
cols = ['Age', 'Hormonal Contraceptives (years)', 'First sexual intercourse','Hormonal Contraceptives', 'Num of pregnancies','Number of sexual partners',
        'Smokes (years)','Smokes (packs/year)', 'IUD', 'IUD (years)','Smokes','STDs:HIV','STDs (number)','Dx:HPV','STDs: Number of diagnosis']
X = data1[cols]