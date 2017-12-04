# Based on https://www.kaggle.com/the1owl/regressing-during-insomnia-0-21496

from multiprocessing import Pool, cpu_count
import gc; gc.enable()
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn import *
import sklearn

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

#%% Import train and test
train = pd.read_csv('../data/train.csv')
train = pd.concat((train, pd.read_csv('../data/train_v2.csv')), axis=0, ignore_index=True).reset_index(drop=True)
test = pd.read_csv('../data/sample_submission_v2.csv')

#%% Count the number of transactions
transactions = pd.read_csv('../data/transactions.csv', usecols=['msno'])
transactions = pd.concat((transactions, pd.read_csv('../data/transactions_v2.csv', usecols=['msno'])), axis=0, ignore_index=True).reset_index(drop=True)
transactions = pd.DataFrame(transactions['msno'].value_counts().reset_index())
transactions.columns = ['msno','trans_count']

#%% Merging transactions
train = pd.merge(train, transactions, how='left', on='msno')
test = pd.merge(test, transactions, how='left', on='msno')
transactions = []
print('transaction merge...')

#%% Count the number of logs
#user_logs = pd.read_csv('../data/user_logs_v2.csv', usecols=['msno'])
user_logs = pd.read_csv('../data/user_logs.csv', usecols=['msno'])
user_logs = pd.concat((user_logs, pd.read_csv('../data/user_logs_v2.csv', usecols=['msno'])), axis=0, ignore_index=True).reset_index(drop=True)
user_logs = pd.DataFrame(user_logs['msno'].value_counts().reset_index())
user_logs.columns = ['msno','logs_count']

#%% Merging logs
train = pd.merge(train, user_logs, how='left', on='msno')
test = pd.merge(test, user_logs, how='left', on='msno')
user_logs = []
print('user logs merge...')

#%% Merging members
members = pd.read_csv('../data/members_v3.csv')
train = pd.merge(train, members, how='left', on='msno')
test = pd.merge(test, members, how='left', on='msno')
members = []
print('members merge...')

#%% Categorize gender
gender = {'male':1, 'female':2}
train['gender'] = train['gender'].map(gender)
test['gender'] = test['gender'].map(gender)

train = train.fillna(0)
test = test.fillna(0)

#%% Keep the latest transaction (keep=first and ascending=False) & Merging
transactions = pd.read_csv('../data/transactions.csv')
transactions = pd.concat((transactions, pd.read_csv('../data/transactions_v2.csv')), axis=0, ignore_index=True).reset_index(drop=True)
transactions = transactions.sort_values(by=['transaction_date'], ascending=[False]).reset_index(drop=True)
transactions = transactions.drop_duplicates(subset=['msno'], keep='first')

train = pd.merge(train, transactions, how='left', on='msno')
test = pd.merge(test, transactions, how='left', on='msno')
transactions=[]

#%% Reading the logs of the users by chunk, merge them with train and test
def transform_df(df):
    df = pd.DataFrame(df)
    df = df.sort_values(by=['date'], ascending=[False])
    df = df.reset_index(drop=True)
    df = df.drop_duplicates(subset=['msno'], keep='first')
    return df

def transform_df2(df):
    df = df.sort_values(by=['date'], ascending=[False])
    df = df.reset_index(drop=True)
    df = df.drop_duplicates(subset=['msno'], keep='first')
    return df

df_iter = pd.read_csv('../data/user_logs.csv', low_memory=False, iterator=True, chunksize=10000000)
last_user_logs = []
i = 0 #~400 Million Records - starting at the end but remove locally if needed
for df in df_iter:
    if i>35:
        if len(df)>0:
            print(df.shape)
            p = Pool(cpu_count())
            df = p.map(transform_df, np.array_split(df, cpu_count()))
            df = pd.concat(df, axis=0, ignore_index=True).reset_index(drop=True)
            df = transform_df2(df)
            p.close(); p.join()
            last_user_logs.append(df)
            print('...', df.shape)
            df = []
    i+=1
last_user_logs.append(transform_df(pd.read_csv('../data/user_logs_v2.csv')))
last_user_logs = pd.concat(last_user_logs, axis=0, ignore_index=True).reset_index(drop=True)
last_user_logs = transform_df2(last_user_logs)

train = pd.merge(train, last_user_logs, how='left', on='msno')
test = pd.merge(test, last_user_logs, how='left', on='msno')
last_user_logs=[]

#%% Replace na by 0, extract the columns used for prediction
train = train.fillna(0)
test = test.fillna(0)


id_test = test['msno'].values
target_train = train['is_churn'].values

train = train.drop(['msno','is_churn'], axis = 1)
test = test.drop(['msno'], axis = 1)

#%%
def xgb_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'log_loss', metrics.log_loss(labels, preds)

params = []
params1 = {
        'eta': 0.002, #use 0.002, was 0.02 before
        'max_depth': 7,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'seed': 1,
        'silent': False
    }
params.append(params1)
params2 = {
        'eta': 0.005, #use 0.002, was 0.02 before
        'max_depth': 15,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'subsample': 0.95,
        'seed': 2,
        'silent': False
    }
params.append(params2)
params3 = {
        'eta': 0.01, #use 0.002, was 0.02 before
        'max_depth': 20,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'subsample': 0.80,
        'seed': 3,
        'silent': False
    }
params.append(params3)

#%% Classifier
Nfold = 5
folds = list(StratifiedKFold(n_splits=Nfold, shuffle=True, random_state=2016).split(train, target_train))

S_train = np.zeros((train.shape[0], len(params)))
S_test = np.zeros((test.shape[0], len(params)))
for i, param in enumerate(params):
    
    S_test_i = np.zeros((test.shape[0], Nfold))

    for j, (train_idx, test_idx) in enumerate(folds):
        X_train = train[train_idx]
        y_train = target_train[train_idx]
        X_holdout = train[test_idx]
        y_holdout = target_train[test_idx]
        
        watchlist = [(xgb.DMatrix(X_train, y_train), 'train'), (xgb.DMatrix(X_holdout, y_holdout), 'valid')]
        print ("Fit %d model, fold %d" % (i+1, j+1))
        model = xgb.train(param, xgb.DMatrix(X_train, y_train), 1500,  watchlist, feval=xgb_score, maximize=False, verbose_eval=10, early_stopping_rounds=50)
        
        S_train[test_idx, i] = model.predict(xgb.DMatrix(X_holdout), ntree_limit=model.best_ntree_limit)
        S_test_i[:, j] = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit)
        
    S_test[:, i] = S_test_i.mean(axis=1)

#%% Stacker
log_model = LogisticRegression()

log_model.fit(S_train, target_train)
res = log_model.predict_proba(S_test)[:,1]

sub = pd.DataFrame()
sub['msno'] = id_test
sub['is_churn'] = res

sub.to_csv('xgb_ensemble_0.csv', index=False)
