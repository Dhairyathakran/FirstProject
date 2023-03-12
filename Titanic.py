# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 00:23:57 2023

@author: dhair
"""

"Import Libraries First"

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix



dataFile='/Users/dhair/Downloads/ML+Classification+Package/ML Classification Package/3. Logisitic Regression/Train_Titanic.csv'

""" --Fill Average value it into the missing Place--"""
def Fill_age(data):
    age=data[0]
    sex=data[1]
    if pd.isnull(age):
        if sex == 'male':
            return 29
        else:
            return 25
    else:
            return age
        
def dataLoadAndSplit(dataFile):
    print("Titanic project!!")
    
    "Import DataSet Second"
    
    training_set = pd.read_csv(dataFile)
    print(training_set.head(10))
    print(training_set.tail(5))
    
    """Explore / Visualize Data Set Total,Survived,No_Survived
    Passenger"""
    
    survived = training_set[training_set['Survived']==1]
    no_survived = training_set[training_set['Survived']==0]
    
    print ('Total Num of Passenger = ',len(training_set))
    print('Num of survived passenger= ',len(survived))
    print('Num of passenger who did not survived=',len(no_survived))
    
    "Calculate The Percentage of Passengers who are Survived or Not"
    
    print ('% of Survived = ', 1.*len(survived)/len(training_set)*100)
    
    print ('% of No Survived = ', 1.*len(no_survived)/len(training_set)*100)
    
    
    """"below command line of 'Full' we are calculate the total num of columns 
    and count the number of quanties inside the Every column """
    
    full = pd.concat([training_set])
    SURV=891
    full.info()
    
    """Wirte a plt for plot size ,subplot (RCF)"""
    
    plt.figure(figsize=[6,6])
    plt.subplot(211)
    sns.countplot(x= 'Pclass',data= training_set)
    plt.subplot(212)
    
    "hue is basically creating histogram based on vaule of some specific variable "
    
    sns.countplot(x= 'Pclass',hue='Survived',data= training_set)
    
    """Wirte a code for finding the howmany siblings are--"""
    plt.figure(figsize=[6,12])
    plt.subplot(211)
    sns.countplot(x= 'SibSp',data= training_set)
    plt.subplot(212)
    sns.countplot(x= 'SibSp',hue='Survived',data= training_set)
    "Finding the parents "
    plt.figure(figsize=[6,12])
    plt.subplot(211)
    sns.countplot(x= 'Parch',data= training_set)
    plt.subplot(212)
    sns.countplot(x= 'Parch',hue='Survived',data= training_set)
    
    "Write a code for Embarked"
    plt.figure(figsize=[6,12])
    plt.subplot(211)
    sns.countplot(x= 'Embarked',data= training_set)
    plt.subplot(212)
    sns.countplot(x= 'Embarked',hue='Survived',data= training_set)
    "---Lets same for Male/Female---"
    plt.figure(figsize=[6,12])
    plt.subplot(211)
    sns.countplot(x= 'Sex',data= training_set)
    plt.subplot(212)
    sns.countplot(x= 'Sex',hue='Survived',data= training_set)
    "--Lets do for AGE--"
    plt.figure(figsize=(40,20))
    sns.countplot(x='Age',hue='Survived',data = training_set)
    training_set['Age'].hist(bins= 30)
    
    
    #data = [[4,3,2,1],[1,2,3,4]]
    #sns.heatmap(data,cmap='Reds')
    
    
    """---Data for training / Data cleaning---"""
    
    training_set.drop(['Name','Cabin','Ticket','Embarked'],axis=1,inplace=True)
    print(training_set)
    sns.heatmap(training_set.isnull(), yticklabels=False,cbar=True,cmap='Greens')
    
    #Lets write the a function calcute the Mean/Average of Age column
    plt.figure(figsize=(15,10))
    sns.boxplot(x='Sex',y='Age',data=training_set)
    

    training_set['Age'] = training_set[['Age','Sex']].apply(Fill_age,axis=1)
    sns.heatmap(training_set.isnull(), yticklabels=False,cbar=True,cmap='Greens')
    
    
    training_set['Age'].hist(bins=20)
    
    training_set.drop(['PassengerId'],axis=1,inplace=True)
    print(training_set)
    
    """--Replace the sex column with Dummies--"""
    male= pd.get_dummies(training_set['Sex'],drop_first=True)
    print (male)
    
    training_set.drop(['Sex'],axis=1,inplace=True)
    
    """Add Male column in training set with concatenate """
    training_set= pd.concat([training_set, male], axis=1)
    
    """--Now we create X and Y -- """
    
    X= training_set.drop('Survived',axis=1).values
    print (X)
    
    y= training_set['Survived'].values
    print(y)
    
    """---Model Training Train the data here"""
    #X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
    return train_test_split(X,y,test_size=0.2,random_state=0)



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
    


if __name__ == "__main__":
    print("I am in main")
    
    #prepare data
    X_train,X_test,y_train,y_test = dataLoadAndSplit(dataFile)
    
    #apply logistic regression
    print("apply logistic regression")
    logClassifier = applyLogisticRegression(X_train, X_test, y_train, y_test)
    
    #check for accuracy using confussion matrix
    print("confusion matrix")
    predictedResults = applyConfusionMatrix(logClassifier, X_test, y_test)
    
    print("\n")
    print("**********************************************************")
    print("         ***** Results Analysis Sumary  *****                     ")
    print("**********************************************************")
    fullAnalysis(y_test, predictedResults)
    
    #display
    print("display results")
    






















































