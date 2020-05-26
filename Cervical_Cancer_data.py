# -*- coding: utf-8 -*-
"""
Created on Tue May 26 12:38:10 2020

@author: SURBHI
"""

""" ***************************************************************************
# * File Description:                                                         *
# * An example of how to stack classifiers using a synthetic dataset.         *
# *                                                                           *
# * The contents of this script are:                                          *
# * 1. Importing Libraries                                                    *
# * 2. Missing Value Imputation                                               *
# * 3. Selecting Target Column                                                *
# * AUTHORS(S): SURBHI <sur7312@gmail.com>                                    *
# * --------------------------------------------------------------------------*
# * ************************************************************************"""

###############################################################################
#                          1. Importing Libraries                             #
###############################################################################
# For reading, visualizing, and preprocessing data
import numpy as np
import pandas as pd
import seaborn as sns
import itertools
import timeit
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn import model_selection
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.metrics import f1_score,precision_score,recall_score
# Classifiers
from sklearn.svm import NuSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingCVClassifier 

# Used to ignore warnings generated from StackingCVClassifier
import warnings
warnings.simplefilter('ignore')

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from mlxtend.classifier import StackingCVClassifier 
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn import metrics
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score
from xgboost.sklearn import XGBClassifier  
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
sns.set()
import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

# UCI Data
data = pd.read_csv('cervical.csv')

des = data.describe()
inf = data.info()

null = data.isna().sum()

null = {}
c = 0
n = 0
for i in data.columns:
    for j in data[i]:
        if j == '?':
            c = c+1
    null[i] = c
    c = 0
    
a = {key: val for key, val in null.items() if val > 0}

for i in data.columns:
    for j in data.index: 
        if data[i][j] == '?':
            data[i][j] = np.nan
        else:
            pass
        
#Counting null values       
data.isna().sum()

#Deleting rows with majority null
data = data.drop(columns = ['STDs: Time since first diagnosis', 'STDs: Time since last diagnosis'])
data = data.apply(pd.to_numeric, errors='coerce')
d = {}
for i in data:
    d[i] = data[i].nunique()
       

###############################################################################
#                     2.   Missing Values Imputation                            #
###############################################################################
    
from missingpy import KNNImputer 
cols = list(data)

data1 = pd.DataFrame(KNNImputer().fit_transform(data))
data1.columns = cols

data1.isna().sum()        

data1 = data1.dropna()
data1.isna().sum()

###############################################################################
#             3.  Select one target column by uncomenting it                   #
###############################################################################

#Biopsy
data1 = data1.drop(columns = ['Schiller', 'Citology','Hinselmann'])

#Citology
#data1 = data1.drop(columns = ['Biopsy', 'Schiller','Hinselmann'])

##Hinsellmann
#data1 = data1.drop(columns = ['Biopsy', 'Citology','Schiller'])

##Schiller
#data1 = data1.drop(columns = ['Biopsy', 'Citology','Hinselmann'])


###############################################################################
#                                 3.  Get Data                                #      
###############################################################################

X = data1.iloc[:, :-1].values
y = data1.iloc[:, -1].values
