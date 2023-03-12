# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 00:23:57 2023

@author: dhair
"""

"Import Libraries First"

import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import dataProcessing as dp



def applyLogisticRegression(X_train,X_test,y_train,y_test):
    classifier = LogisticRegression(random_state=0)
    classifier.fit(X_train, y_train)
    return classifier
    

def applyConfusionMatrix(classifier, X_test, y_test):    
    """---Model Testing TEst data and confusion matrix ---"""
    y_predict = classifier.predict(X_test)
    print(y_predict)
    
    cm = confusion_matrix(y_test, y_predict)
    print(cm)
    sns.heatmap(cm, annot= True, fmt= 'd')
    return y_predict


def fullAnalysis(actual, predicted):
    #Classification Report 
    print(classification_report(actual, predicted))