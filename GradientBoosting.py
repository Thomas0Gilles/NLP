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


df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

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

#%% Scission des variables catégoriques

#Supposition: de 0 à p pour variable catégorique (p+1 different feature). NB: -1 est une valeur qui n'apparait jamais
#Première version où je delete la variable -1 dans les catégoriels: elle ne prendra juste pas de valeurs (sum!=1)
variables = df_train.columns.values
p = df_train.shape[1]
variables_cat = []
features_cat = 0
for i in range(2,p):
    name = variables[i]
    if(name.find("cat")!=-1):
        variables_cat.append(name)
        features_cat += max(df_train[name])+1  
        print("Variable ", name," a ", max(df_train[name])+1, " features")

print("Scission")

cat_scinde_train = np.zeros((df_train.shape[0],features_cat))
cat_scinde_test = np.zeros((df_test.shape[0],features_cat))
count = 0
for ind in variables_cat:
    print("new")
    value = np.unique(df_train[ind])
    for v in value:
        if(v!=-1):
            cat_scinde_train[np.where(df_train[ind]==v),count] = 1
            cat_scinde_test[np.where(df_test[ind]==v),count] = 1
            count += 1

df_train = df_train.drop(variables_cat, axis=1)
df_test = df_test.drop(variables_cat, axis=1)

print("Concatenation")

df_tr = pd.DataFrame(data = cat_scinde_train,index=df_train.index.values,dtype=np.int8) #nom de colonne?
df_te = pd.DataFrame(data = cat_scinde_test,index=df_test.index.values,dtype=np.int8)

frame_train = [df_train, df_tr]
frame_test = [df_test, df_te]

df_train = pd.concat(frame_train, axis=1)
df_test = pd.concat(frame_test, axis=1)

#%% Etude mémoire et réduction

#--- memory consumed by train dataframe ---
mem = df_train.memory_usage(index=True).sum()
print("Memory consumed by training set  :   {} MB" .format(mem/ 1024**2))
print('\n')
#--- memory consumed by test dataframe ---
mem = df_test.memory_usage(index=True).sum()
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

change_datatype(df_train)
change_datatype(df_test) 

#--- Converting columns from 'float64' to 'float32' ---
def change_datatype_float(df):
    float_cols = list(df.select_dtypes(include=['float']).columns)
    for col in float_cols:
        df[col] = df[col].astype(np.float32)
        
change_datatype_float(df_train)
change_datatype_float(df_test)

#--- memory consumed by train dataframe ---
mem = df_train.memory_usage(index=True).sum()
print("Memory consumed by training set  :   {} MB" .format(mem/ 1024**2))
print('\n') 
#--- memory consumed by test dataframe ---
mem = df_test.memory_usage(index=True).sum()
print("Memory consumed by test set      :   {} MB" .format(mem/ 1024**2))

#%% Split

print("Splitting")

features= [c for c in df_train.columns.values if c  not in ['id', 'target']]
X_train, X_test, y_train, y_test = train_test_split(df_train[features], df_train['target'], test_size=0.2, random_state=42)

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


hyperparameters = { #'features__text__tfidf__max_df': [0.9, 0.95],
                    #'features__text__tfidf__ngram_range': [(1,1), (1,2)],
                    #'classifier__learning_rate': [0.1, 0.2],
                    'classifier__n_estimators': [20, 30, 50],
                    'classifier__max_depth': [2, 4],
                    'classifier__min_samples_leaf': [2, 4]
                  }
clf = GridSearchCV(pipeline, hyperparameters, cv = 3)
 
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

test_features= [c for c in df_test.columns.values if c  not in ['id']]
predictions = clf.predict_proba(df_test[test_features])   #On a des proba avec cet algo

#%% Write submission

preds = pd.DataFrame(data = predictions, columns = clf.best_estimator_.named_steps['classifier'].classes_)

result = pd.concat([df_test[['id']], preds], axis=1)
result = result.drop(0, axis=1)
result.columns = ['id', 'target']
result.head()

result.to_csv('GradientBoosting_split.csv', index=False)
#%%