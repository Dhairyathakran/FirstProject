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
import logisticRegr as lg


def usage():
    print("Error : Please pass the valid arguments")
    print("python executable <input Argument>")
    print("Exiting the program")
    return



def main(cmdArgs):
    # Parse input arguments
    argsCount = len(cmdArgs)
    if argsCount < 2:
        usage()
        return
        
    dataFile = cmdArgs[1]
 
    #prepare data
    X_train,X_test,y_train,y_test = dp.dataLoadAndSplit(dataFile)
    
    #apply logistic regression
    print("apply logistic regression")
    logClassifier = lg.applyLogisticRegression(X_train, X_test, y_train, y_test)
    
    #check for accuracy using confussion matrix
    print("confusion matrix")
    predictedResults = lg.applyConfusionMatrix(logClassifier, X_test, y_test)
    
    print("\n")
    print("**********************************************************")
    print("         ***** Results Analysis Sumary  *****                     ")
    print("**********************************************************")
    lg.fullAnalysis(y_test, predictedResults)

    #display
    print("display results")



if __name__ == "__main__":
    print("implemetation of LogisticRegression")
    main(sys.argv)
    

    






















































