#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 16:44:46 2019

@author: opasina
"""

#import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
#import seaborn as sns

from pylab import rcParams

# Loading the CSV with pandas
data = pd.read_csv('Telco-Customer-Churn.csv')

# Data to plot
sizes = data['Churn'].value_counts(sort = True)
colors = ["grey","purple"] 
rcParams['figure.figsize'] = 5,5
# Plot
plt.pie(sizes,  labels=["Yes","No"], colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=270,)
plt.title('Percentage of Churn in Dataset')
plt.show()


data.drop(['customerID'], axis=1, inplace=True)

data['TotalCharges'] = data['TotalCharges'].apply(lambda x: float(x) if x != " " else " ")

data['MonthlyCharges'] = data['MonthlyCharges'].apply(lambda x: float(x) if x != " " else " ")

data.drop(columns =['TotalCharges','MonthlyCharges'])

## Remove empty columns
data = data[data['TotalCharges'] != ' ']
data = data[data['MonthlyCharges'] != ' ']

def sublst(row):
    housing_map = {'Yes': 1, 'No': 0}
    row = row.map(housing_map)
       
    return row

colNames = ['Partner','Dependents','PhoneService','Churn','PaperlessBilling']

data[colNames] = data[colNames].apply(lambda x: sublst(x) )

dummy_array = ["gender","MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaymentMethod"]

dummyDf = pd.DataFrame()

dummyDf = pd.get_dummies(data=data, columns=dummy_array)
#for dummy in dummy_array:
#    df_col = pd.get_dummies(data[dummy])
#    dummyDf = pd.concat([dummyDf,df_col],sort=False)
    

#data["Churn"] = data["Churn"].astype(int)
Y = dummyDf["Churn"].values
X = dummyDf.drop(labels = ["Churn"],axis = 1)
# Create Train & Test Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=101)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
result = model.fit(X_train, y_train)

from sklearn import metrics
prediction_test = model.predict(X_test)
# Print the prediction accuracy
print (metrics.accuracy_score(y_test, prediction_test))

# To get the weights of all the variables
weights = pd.Series(model.coef_[0],index=X.columns.values)
weights.sort_values(ascending = False)
