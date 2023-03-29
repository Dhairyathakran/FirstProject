# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 14:03:36 2023

@author: dhair
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix , classification_report

bank = pd.read_csv('/Users/dhair/OneDrive/Desktop/Bank_Customer_retirement.csv')
print(bank.keys())
print(bank.head(10))

#visualizing the data
sns.pairplot(bank , hue = 'Retire' , vars = ['Age' , '401K Savings'])
plt.show()

sns.countplot(bank['Retire'] , label = 'Retirement')
plt.show()

bank = bank.drop(['Customer ID'] , axis = 1 )
print(bank)

X = bank.drop(['Retire'] , axis = 1)

Y = bank['Retire']

#spliting Data 

X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size = 0.25 , random_state = 5)

#Now Traing the svm model


classifier = SVC()
classifier.fit(X_train , Y_train)

Y_predict = classifier.predict(X_test)
print(Y_predict)

cm = confusion_matrix(Y_test , Y_predict)
sns.heatmap ( cm , annot = True)

classification = classification_report(Y_test, Y_predict)
print (classification)


