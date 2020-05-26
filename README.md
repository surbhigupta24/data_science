# data_science
cancer prediction model
Computational Prediction of Cervical Cancer Diagnosis with Class Imbalance Handling and Feature Reduction Techniques

Installation: To run the scripts, you need to have installed:

Spyder(Python) 
Python 3.7
Python packages panda
pip install panda
Python packages panda
pip install numpy

You need to have root privileges, an internet connection, and at least 1 GB of free space on your hard disk.
Our scripts were originally developed on a Dell -15JPO9P computer with an Intel Core i7-8550U CPU 1.80GHz processor, with 8 GB of Random-Access Memory (RAM).

Dataset preparation
Download the Cervical cancer (Risk Factors) Data Set file  at the following URL: 
https://archive.ics.uci.edu/ml/datasets/Cervical+cancer+%28Risk+Factors%29

To get data and handle missing values:
Execute Cervical_Cancer_data.py

To balance and scale data and split into train and test set:
Execute Cervical_Cancer_preprocess.py

Feature selection
To run the Python code for random forest feature selection:
Execute cervical_cancer_RFFS.py

To run the Python code for Extremely randomized trees feature selection:
Execute cervical_cancer_RFFS.py

Classification ALgorithms

To run the Multi-layer Perceptron Classifier:
Execute MLP.py 

To run the Random Forest Classifier:
Execute RF.py 

To run the KNN Classifier:
Execute KNN.py 

To run the Gradient Boosting Classifier:
Execute GBC.py 

To run the Majority Voting Ensemble Classifier:
Execute Ensemble_voting.py 

To run the Weighted Voting Ensemble Classifier:
Execute Weighted_Voting.py 

To run the proposed Stacking Classifier:
Execute Stacking.py 

Reference
More information about this project can be found on this paper:
Surbhi Gupta and Manoj K. Gupta "Computational Prediction of Cervical Cancer Diagnosis with Class Imbalance Handling and Feature Reduction Techniques".

The Cervical cancer (Risk Factors) dataset is publically available on the website of the University of California Irvine Machine Learning Repository, under its copyright license.

Contacts
This sofware was developed by Surbhi Gupta at the School of Computer Science & Engineering, Shri Mata Vaishno Devi University, Sub-Post Office,  Network Centre, Katra, Jammu and Kashmir 182320, India . 
For questions or help, please write to sur7312@gmail.com 