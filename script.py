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
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

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
model = DecisionTreeClassifier(random_state=42,max_depth=5,class_weight='balanced')
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

#Hyperparameter tuning
print(cross_val_score(model,features,target,cv=10))
depth = [i for i in range(5,21)]
samples = [i for i in range(50,500,50)]
parameters = dict(max_depth=depth,min_samples_leaf=samples)
param_search = GridSearchCV(model, parameters)
param_search.fit(features_train,target_train)
print(param_search.best_params_)

#important features
feature_importances = model.feature_importances_
feature_list = list(features)
relative_importances = pd.DataFrame(index=feature_list, data=feature_importances, columns=["importance"])
relative_importances.sort_values(by='importance', ascending=False)
print(relative_importances.head())
selected_features = relative_importances[relative_importances.importance>0.01]
selected_list = selected_features.index
features_train_selected = features_train[selected_list]
features_test_selected = features_test[selected_list]

#best model
model_best = DecisionTreeClassifier(max_depth=8,min_samples_leaf=150,class_weight='balanced',random_state=42)
model_best.fit(features_train_selected,target_train)
prediction_best = model_best.predict(features_test_selected)
print(model_best.score(features_test_selected,target_test)*100)
print(recall_score(prediction_best,target_test)*100)
print(roc_auc_score(prediction_best,target_test)*100)
