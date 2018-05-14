#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 23:18:33 2018

@author: slytherin
"""

import pandas as pd
data = pd.read_csv("turnover.csv")
data.salary = data.salary.astype('category')
data.salary = data.salary.cat.reorder_categories(['low', 'medium', 'high'])
data.salary = data.salary.cat.codes
departments = pd.get_dummies(data.department)
departments=departments.drop('accounting',axis=1)
data=data.drop('department',axis=1)
data=data.join(departments)
n_employees = len(data)
print(data.churn.value_counts())
print(data.churn.value_counts()/n_employees*100)