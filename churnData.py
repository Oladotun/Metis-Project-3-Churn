#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 16:44:46 2019

@author: opasina
"""

import numpy as np
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

#data.drop(columns =['TotalCharges','MonthlyCharges'])


## Remove empty columns
data = data[data['TotalCharges'] != ' ']
data = data[data['MonthlyCharges'] != ' ']




def sublst(row):
    housing_map = {'Yes': 1, 'No': 0}
    
    row = row.map(housing_map)
       
    return row

churn = data['Churn'].to_frame().apply(lambda x: sublst(x) )
#churn.to_csv("ChurnValues.txt", index=False)

churn.hist(bins=20, color='steelblue', edgecolor='black', linewidth=1.0,
           xlabelsize=10, ylabelsize=10, grid=False)    
plt.tight_layout(rect=(0, 0, 1.0, 1.0))  

colNames = ['Partner','Dependents','PhoneService','Churn','PaperlessBilling']

data[colNames] = data[colNames].apply(lambda x: sublst(x) )

dummy_array = ["gender","MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaymentMethod"]

dummyDf = pd.DataFrame()

dummyDf = pd.get_dummies(data=data, columns=dummy_array)


import seaborn as sns
f, ax = plt.subplots(figsize=(10, 6))
corr = data.corr()
hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f',
                 linewidths=.05)
f.subplots_adjust(top=0.93)
t= f.suptitle('Churn Attributes Correlation Heatmap', fontsize=14)
#for dummy in dummy_array:
#    df_col = pd.get_dummies(data[dummy])
#    dummyDf = pd.concat([dummyDf,df_col],sort=False)

#data.to_csv("TelcoChurnFinalDataFrame.csv", index=False)

cols = ['TotalCharges', 'MonthlyCharges']
subset_df = data[cols]

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()

scaled_df = ss.fit_transform(subset_df)
scaled_df = pd.DataFrame(scaled_df, columns=cols)
final_df = scaled_df.copy()
final_df['Churn'] = data['Churn'].apply(lambda x: "Yes" if x == 1 else "No")
final_df.head()

# plot parallel coordinates
from pandas.plotting import parallel_coordinates
plt.figure()
parallel_coordinates(final_df, 'Churn', color=('#FFE888', '#FF9999'))
plt.show()
    

#data["Churn"] = data["Churn"].astype(int)
Y = dummyDf["Churn"].values
X = dummyDf.drop(labels = ["Churn"],axis = 1)

dummyDf.to_csv("allDummyData.csv",index=True)
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
weightSorted = weights.sort_values(ascending = False)

#for column in X.columns.values:
#    dummyDf[column]
    
dataCount = dummyDf.drop(labels = ["TotalCharges", "MonthlyCharges","tenure"],axis = 1) 

#for column in dataCount.columns:
#    dataCount[column] == 1 and dataCount["Churn"] == 1

#snrDf = pd.DataFrame()
#snrDf["columnName"] = ""
#snrDf.set_index('columnName')

snrDfArray = []
i = 0
for column in dataCount.columns:
    if column != "Churn":
        res = dataCount.groupby([column, "Churn"]).size()
#        snrDf.insert(i, "Churn_0", res[2])
#        snrDf.insert(i, "Churn_1", res[3])
#        snrDf.insert(i, "columnName", column)
        snrDfArray.append({"feature":column,"Churn_0":res[2],"Churn_1":res[3]}) 
        i += 1
snrDf = pd.DataFrame(snrDfArray)
snrDf.to_csv("churnUniqueCounts.csv",index=True)
#        snrDf.insert(1, column, res[3])
        

#dcSC = dataCount.groupby(["SeniorCitizen", "Churn"]).size()

plt.figure(figsize=(18,4))

weightSortedIndex = weightSorted.index.values
weightSortedValues = weightSorted.values

weights = weights.reset_index()

weights.to_csv("IndexCoefficent.csv", index=True)

firstLastFiveIndex = [' Month To Month\nContract','Internet Service\n Fiber Optic','Paperless\nBilling','Uses Technology \nSupport', 'Two Year\nContract ', 'Internet Service Type\nDSL']
#np.concatenate((weightSortedIndex[:3], weightSortedIndex[-3:]), axis=None)
firstLastFiveValues = np.concatenate((weightSortedValues[:3], weightSortedValues[-3:]), axis=None)
plt.bar(firstLastFiveIndex,firstLastFiveValues )
#plt.xticks(rotation=180)

#labels = ['Contract\n Month To Month','Internet Service\n Fiber Optic','Paperless\nBilling','Tech Support', 'Contract Two Year', 'Internet Service\nDSL']
#plt.plot(firstLastFiveIndex,firstLastFiveValues, 'r')
#plt.xticks(firstLastFiveValues, labels, rotation='vertical')
#plt.title('Parameters')
#plt.ylabel('Logistic Regression Coefficients');
#plt.show()
