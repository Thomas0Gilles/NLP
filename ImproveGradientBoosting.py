#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 14:37:17 2017

@author: marc-jourdan
"""

#%% Import
import numpy as np
import pandas as pd 

from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer


train = pd.read_csv('../train.csv')
test = pd.read_csv('../test.csv')

#%% Gini
def gini(actual, pred, cmpcol = 0, sortcol = 1):
    all = np.asarray(np.c_[ actual, pred, np.arange(actual.shape[0]) ], dtype=np.float)
    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
    totalLosses = all[:,0].sum()
    giniSum = all[:,0].cumsum().sum() / totalLosses
 
    giniSum -= (actual.shape[0] + 1) / 2.
    return giniSum / actual.shape[0]
    
def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)

gini_score = make_scorer(gini_normalized,greater_is_better=True)
gini_loss  = make_scorer(gini_normalized, greater_is_better=False)

#%% Scission des variables catégoriques
id_test = test['id'].values
target_train = train['target'].values

train = train.drop(['target','id'], axis = 1)
test = test.drop(['id'], axis = 1)

col_to_drop = train.columns[train.columns.str.startswith('ps_calc_')]
train = train.drop(col_to_drop, axis=1)  
test = test.drop(col_to_drop, axis=1)  

train = train.replace(-1, np.nan)
test = test.replace(-1, np.nan)

cat_features = [a for a in train.columns if a.endswith('cat')]

for column in cat_features:
	temp = pd.get_dummies(pd.Series(train[column]))
	train = pd.concat([train,temp],axis=1)
	train = train.drop([column],axis=1)
    
for column in cat_features:
	temp = pd.get_dummies(pd.Series(test[column]))
	test = pd.concat([test,temp],axis=1)
	test = test.drop([column],axis=1)

print(train.values.shape, test.values.shape)


#%% Etude mémoire et réduction

#--- memory consumed by train dataframe ---
mem = train.memory_usage(index=True).sum()
print("Memory consumed by training set  :   {} MB" .format(mem/ 1024**2))
print('\n')
#--- memory consumed by test dataframe ---
mem = test.memory_usage(index=True).sum()
print("Memory consumed by test set      :   {} MB" .format(mem/ 1024**2))

#Changement de tous les types entiers (donc aussi variable binaire)
def change_datatype(df):
    float_cols = list(df.select_dtypes(include=['int']).columns)
    for col in float_cols:
        if ((np.max(df[col]) <= 127) and(np.min(df[col] >= -128))):
            df[col] = df[col].astype(np.int8)
        elif ((np.max(df[col]) <= 32767) and(np.min(df[col] >= -32768))):
            df[col] = df[col].astype(np.int16)
        elif ((np.max(df[col]) <= 2147483647) and(np.min(df[col] >= -2147483648))):
            df[col] = df[col].astype(np.int32)
        else:
            df[col] = df[col].astype(np.int64)

change_datatype(train)
change_datatype(test) 

#--- Converting columns from 'float64' to 'float32' ---
def change_datatype_float(df):
    float_cols = list(df.select_dtypes(include=['float']).columns)
    for col in float_cols:
        df[col] = df[col].astype(np.float32)
        
change_datatype_float(train)
change_datatype_float(test)

#--- memory consumed by train dataframe ---
mem = train.memory_usage(index=True).sum()
print("Memory consumed by training set  :   {} MB" .format(mem/ 1024**2))
print('\n') 
#--- memory consumed by test dataframe ---
mem = test.memory_usage(index=True).sum()
print("Memory consumed by test set      :   {} MB" .format(mem/ 1024**2))

#%% Split

print("Splitting")

X_train, X_test, y_train, y_test = train_test_split(train, target_train, test_size=0.2, random_state=42)

#%% First fit
pipeline = Pipeline([
    ('classifier', GradientBoostingClassifier(random_state = 42))
])

print("Fitting")

pipeline.fit(X_train, y_train)

preds = pipeline.predict(X_test)
probs = pipeline.predict_proba(X_test)
Gini_test = gini_normalized(y_test,probs)
print("First try, Gini normalized : ", Gini_test)
print("Accuracy : ", np.mean(preds==y_test))

#%% Refitting

pipeline.get_params().keys()


hyperparameters = { 'classifier__learning_rate': [0.01, 0.1],
                    'classifier__n_estimators': [20, 30, 70, 100],
                    'classifier__max_depth': [2, 4],
                    'classifier__min_samples_leaf': [2, 5],
                    'classifier__verbose':[1]
                  }
clf = GridSearchCV(pipeline, hyperparameters, cv = 5, scoring = gini_score)
 
# Fit and tune model
clf.fit(X_train, y_train)

#refitting on entire training data using best settings
clf.refit

preds = clf.predict(X_test)
probs = clf.predict_proba(X_test)

Gini_test = gini_normalized(y_test,probs)
print("Refit, Gini normalized = ", Gini_test)
print("Accuracy : ", np.mean(preds==y_test))

#%% Predict submission

predictions = clf.predict_proba(test)

#%% Write submission

sub = pd.DataFrame()
sub['id'] = id_test
sub['target'] = predictions

sub.to_csv('ImproveGradBoost.csv', index = False)
#%%
