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

#%% Merge trans_mem
print("Loading 2 ...")
transmem = pd.read_csv('../data/trans_mem_unscaled_categorical.csv', dtype={'Unnamed: 0':np.int32,'payment_plan_days':np.float32,'plan_list_price':np.float32,
                                                              'actual_amount_paid':np.float32,'is_auto_renew': np.int8, 'is_cancel': np.float32,
                                                              'trans_count':np.float32,'discount':np.float32,'is_discount':np.int8,'amt_per_day': np.float32,
                                                              'membership_duration':np.float32,'bd':np.float32,'registration_duration': np.float32,'reg_mem_duration':np.float32,
                                                              'autorenew_&_not_cancel':np.int8,'notAutorenew_&_cancel': np.int8,'long_time_user':np.float32})
change_datatype_float(transmem)
change_datatype(transmem)

for f in transmem.columns: 
    if transmem[f].dtype=='object': 
        print("type object pour ", f)
        if f!='msno':
            transmem.drop([f],axis=1)

train = pd.merge(train, transmem, how='left', on='msno')
test = pd.merge(test, transmem, how='left', on='msno')
del transmem

#%% Merge user_FE
print("Loading 3 ...")
userFE = pd.read_csv('../data/user_FE.csv')
change_datatype_float(userFE)
change_datatype(userFE)

for f in userFE.columns: 
    if userFE[f].dtype=='object': 
        print("type object pour ", f)
        if f!='msno':
            userFE.drop([f],axis=1)
print(userFE.dtypes)
userFE = userFE.drop(['msno.1'],axis=1)

train = pd.merge(train, userFE, how='left', on='msno')
test = pd.merge(test, userFE, how='left', on='msno')
del userFE



#%% Replace na by 0, extract the columns used for prediction
result = pd.DataFrame()
result['msno'] = test['msno']
labels = train['msno']

train = train.drop(['msno','is_churn'],axis=1)
test = test.drop(['msno','is_churn'],axis=1)

train = train.fillna(0)
test = test.fillna(0)
print(train.dtypes)
print(test.dtypes)

print("Number of features: ",train.shape[1])

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
    x1, x2, y1, y2 = model_selection.train_test_split(train, labels, test_size=0.2, random_state=i)
    watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
    model = xgb.train(params, xgb.DMatrix(x1, y1), 1500,  watchlist, feval=xgb_score, maximize=False, verbose_eval=50, early_stopping_rounds=50) #use 1500, was 150 before
    if i != 0:
        pred += model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit)
    else:
        pred = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit)
pred /= fold


result['is_churn'] = pred
result.to_csv('MyXgb_mini_plus.csv', index=False)
