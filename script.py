#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 23:18:33 2018

@author: slytherin
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
data = pd.read_csv("turnover.csv")
data.salary = data.salary.astype('category')
data.salary = data.salary.cat.reorder_categories(['low', 'medium', 'high'])
data.salary = data.salary.cat.codes
departments = pd.get_dummies(data.department)
departments=departments.drop('accounting',axis=1)
data=data.drop('department',axis=1)
data=data.join(departments)
n_employees = len(data)
#print(data.churn.value_counts())
#print(data.churn.value_counts()/n_employees*100)
target=data.churn
features=data.drop('churn',axis=1)
target_train, target_test, features_train, features_test = train_test_split(target,features,test_size=0.25,random_state=42)
model = DecisionTreeClassifier(random_state=42,max_depth=7,class_weight='balanced')
model.fit(features_train,target_train)
print(model.score(features_train,target_train)*100)
print(model.score(features_test,target_test)*100)
export_graphviz(model,"tree.dot")

#precision metrics
prediction = model.predict(features_test)
print(precision_score(target_test, prediction))
 
#recall metrics
prediction = model.predict(features_test)
print(recall_score(target_test,prediction))

#roc auc score
prediction = model.predict(features_test)
print(roc_auc_score(target_test, prediction))

