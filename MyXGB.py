# Based on https://www.kaggle.com/the1owl/regressing-during-insomnia-0-21496

from multiprocessing import Pool, cpu_count
import gc; gc.enable()
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn import *
import sklearn

#%%
def change_datatype(df):
    int_cols = list(df.select_dtypes(include=['int']).columns)
    for col in int_cols:
        if ((np.max(df[col]) <= 127) and(np.min(df[col] >= -128))):
            df[col] = df[col].astype(np.int8)
        elif ((np.max(df[col]) <= 32767) and(np.min(df[col] >= -32768))):
            df[col] = df[col].astype(np.int16)
        elif ((np.max(df[col]) <= 2147483647) and(np.min(df[col] >= -2147483648))):
            df[col] = df[col].astype(np.int32)
        else:
            df[col] = df[col].astype(np.int64)

def change_datatype_float(df):
    float_cols = list(df.select_dtypes(include=['float']).columns)
    for col in float_cols:
        df[col] = df[col].astype(np.float32)

#%% Loading 
print("Loading 1 ...")
train = pd.read_csv('../data/train.csv')
train = pd.concat((train, pd.read_csv('../data/train_v2.csv')), axis=0, ignore_index=True).reset_index(drop=True)
test = pd.read_csv('../data/sample_submission_v2.csv')

#%% Merge user_FE
print("Loading 3 ...")
userFE = pd.read_csv('../data/user_FE.csv')
#userFE = userFE.drop(['mnso.1'], axis=1)
change_datatype_float(userFE)
change_datatype(userFE)

#from sklearn import preprocessing 
for f in userFE.columns: 
    if userFE[f].dtype=='object': 
        print("type object pour ", f)
        if f!='msno':
            userFE.drop([f],axis=1)
#        lbl = preprocessing.LabelEncoder() 
#        lbl.fit(list(train[f].values)) 
#        train[f] = lbl.transform(list(train[f].values))

train = pd.merge(train, userFE, how='left', on='msno')
test = pd.merge(test, userFE, how='left', on='msno')
del userFE

#%% Merge trans_mem
print("Loading 2 ...")
transmem = pd.read_csv('../data/trans_mem.csv')
change_datatype_float(transmem)
change_datatype(transmem)

#from sklearn import preprocessing 
for f in transmem.columns: 
    if transmem[f].dtype=='object': 
        print("type object pour ", f)
        if f!='msno':
            transmem.drop([f],axis=1)
#        lbl = preprocessing.LabelEncoder() 
#        lbl.fit(list(train[f].values)) 
#        train[f] = lbl.transform(list(train[f].values))

train = pd.merge(train, transmem, how='left', on='msno')
test = pd.merge(test, transmem, how='left', on='msno')
del transmem

#%% Replace na by 0, extract the columns used for prediction
train = train.fillna(0)
test = test.fillna(0)

cols = [c for c in train.columns if c not in ['is_churn','msno']]

print("Number of features: ",len(cols))

#%%
def xgb_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'log_loss', metrics.log_loss(labels, preds)

fold = 10
for i in range(fold):
    params = {
        'eta': 0.002, #use 0.002, was 0.02 before
        'max_depth': 7,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'seed': i,
        'silent': False
    }
    x1, x2, y1, y2 = model_selection.train_test_split(train[cols], train['is_churn'], test_size=0.2, random_state=i)
    watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
    model = xgb.train(params, xgb.DMatrix(x1, y1), 1500,  watchlist, feval=xgb_score, maximize=False, verbose_eval=50, early_stopping_rounds=50) #use 1500, was 150 before
    if i != 0:
        pred += model.predict(xgb.DMatrix(test[cols]), ntree_limit=model.best_ntree_limit)
    else:
        pred = model.predict(xgb.DMatrix(test[cols]), ntree_limit=model.best_ntree_limit)
pred /= fold
test['is_churn'] = pred.clip(0.+1e-15, 1-1e-15)
test[['msno','is_churn']].to_csv('MyXgb_unscaled.csv', index=False)
