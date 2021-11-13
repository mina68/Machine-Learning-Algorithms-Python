# -*- coding: utf-8 -*-
"""
Created on Thu May 16 02:24:42 2019

@author: Mina
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('Churn_Modelling.csv')

df = df.drop(['RowNumber',
              'CustomerId',
              'Surname',
              'Geography',
              'Gender'
              ], axis =1)

columns = 	['CreditScore',
            'Age',
            'Tenure',
            'Balance',
            'NumOfProducts',
            'HasCrCard',
            'IsActiveMember',
            'EstimatedSalary',
            'Exited'
        ]


labels = df['Exited'].values
features = df[list(columns)].values

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.30)

clf = RandomForestClassifier(n_estimators=4)
clf = clf.fit(X_train, y_train)

accuracy = clf.score(X_train, y_train)
print (accuracy*100)

accuracy = clf.score(X_test, y_test)
print (accuracy*100)




