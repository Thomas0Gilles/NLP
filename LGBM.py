# Inspired by https://www.kaggle.com/talysacc/lgbm-starter-lb-0-23434

from multiprocessing import Pool, cpu_count
import gc; gc.enable()
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import ShuffleSplit
import gc

#%% Read and merge train and member
df_train = pd.read_csv('../data/train.csv',dtype={'is_churn' : bool,'msno' : str})
df_train = pd.concat((df_train, pd.read_csv('../data/train_v2.csv',dtype={'is_churn' : bool,'msno' : str})), axis=0, ignore_index=True).reset_index(drop=True)
df_test = pd.read_csv('../data/sample_submission_v2.csv',dtype = {'msno' : str})

df_members = pd.read_csv('../data/members_v3.csv',dtype={'registered_via' : np.uint8})

gender = {'male':1, 'female':2}
df_members['gender'] = df_members['gender'].map(gender)

df_members = df_members.fillna(0)

df_train = pd.merge(left = df_train,right = df_members,how = 'left',on=['msno'])
df_test = pd.merge(left = df_test,right = df_members,how = 'left',on = ['msno'])

del df_members

#%% read, merge transaction and aggregate results
df_transactions = pd.read_csv('../data/transactions.csv',dtype = {'payment_method' : 'category',
                                                                  'payment_plan_days' : np.uint8,
                                                                  'plan_list_price' : np.uint8,
                                                                  'actual_amount_paid': np.uint8,
                                                                  'is_auto_renew' : np.bool}) #,is_cancel' : np.bool

df_transactions = pd.concat((df_transactions, pd.read_csv('../data/transactions_v2.csv', dtype = {'payment_method' : 'category',
                                                                  'payment_plan_days' : np.uint8,
                                                                  'plan_list_price' : np.uint8,
                                                                  'actual_amount_paid': np.uint8,  #,'is_cancel' : np.bool
                                                                  'is_auto_renew' : np.bool})), axis=0, ignore_index=True).reset_index(drop=True)

df_transactions = pd.merge(left = df_train,right = df_transactions,how='left',on='msno')
df_transactions_test = pd.merge(left = df_test,right = df_transactions,how='left',on='msno')

grouped  = df_transactions.copy().groupby('msno')
grouped_test  = df_transactions_test.copy().groupby('msno')

df_stats = grouped.agg({'msno' : {'total_order' : 'count'},
                         'plan_list_price' : {'plan_net_worth' : 'sum'},
                         'actual_amount_paid' : {'mean_payment_each_transaction' : 'mean',
                                                  'total_actual_payment' : 'sum'},
                         'is_cancel' : {'cancel_times' : lambda x : sum(x==1)}})
df_stats_test = grouped_test.agg({'msno' : {'total_order' : 'count'},
                         'plan_list_price' : {'plan_net_worth' : 'sum'},
                         'actual_amount_paid' : {'mean_payment_each_transaction' : 'mean',
                                                  'total_actual_payment' : 'sum'},
                         'is_cancel' : {'cancel_times' : lambda x : sum(x==1)}})

df_stats.columns = df_stats.columns.droplevel(0)
df_stats.reset_index(inplace=True)
df_train = pd.merge(left = df_train,right = df_stats,how='left',on='msno')

df_stats_test.columns = df_stats_test.columns.droplevel(0)
df_stats_test.reset_index(inplace=True)
df_test = pd.merge(left = df_test,right = df_stats_test,how='left',on='msno')

del df_transactions_test,df_stats_test
del df_transactions,df_stats

# The bd column have clear outlier, we clip it and reduce the type
df_train['bd'].clip(0,100)
df_train['bd'].fillna(0,inplace=True)
df_train['bd'].astype(np.uint8,inplace=True)

df_test['bd'].clip(0,100)
df_test['bd'].fillna(0,inplace=True)
df_test['bd'].astype(np.uint8,inplace=True)

#%% Count the number of logs
user_logs = pd.read_csv('../data/user_logs_v2.csv', usecols=['msno'])
#user_logs = pd.read_csv('../data/user_logs.csv', usecols=['msno'])
#user_logs = pd.concat((user_logs, pd.read_csv('../data/user_logs_v2.csv', usecols=['msno'])), axis=0, ignore_index=True).reset_index(drop=True)
user_logs = pd.DataFrame(user_logs['msno'].value_counts().reset_index())
user_logs.columns = ['msno','logs_count']

#%% Merging logs
df_train = pd.merge(df_train, user_logs, how='left', on='msno')
df_test = pd.merge(df_test, user_logs, how='left', on='msno')
del user_logs
print('user logs merge...')

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

#df_iter = pd.read_csv('../data/user_logs.csv', low_memory=False, iterator=True, chunksize=10000000)
last_user_logs = []
#i = 0 #~400 Million Records - starting at the end but remove locally if needed
#for df in df_iter:
#    if i>35:
#        if len(df)>0:
#            print(df.shape)
#            p = Pool(cpu_count())
#            df = p.map(transform_df, np.array_split(df, cpu_count()))
#            df = pd.concat(df, axis=0, ignore_index=True).reset_index(drop=True)
#            df = transform_df2(df)
#            p.close(); p.join()
#            last_user_logs.append(df)
#            print('...', df.shape)
#            df = []
#    i+=1
last_user_logs.append(transform_df(pd.read_csv('../data/user_logs_v2.csv')))
last_user_logs = pd.concat(last_user_logs, axis=0, ignore_index=True).reset_index(drop=True)
last_user_logs = transform_df2(last_user_logs)

df_train = pd.merge(df_train, last_user_logs, how='left', on='msno')
df_test = pd.merge(df_test, last_user_logs, how='left', on='msno')
del last_user_logs

#%% Replace na by 0, extract the columns used for prediction
df_train = df_train.fillna(0)
df_test = df_test.fillna(0)

#%% Training phase
bst = None

for train_indices,val_indices in ShuffleSplit(n_splits=1,test_size = 0.1,train_size=0.4).split(df_train):
    train_data = lgb.Dataset(df_train.drop(['msno','is_churn'],axis=1).loc[train_indices,:],label=df_train.loc[train_indices,'is_churn'])
    val_data = lgb.Dataset(df_train.drop(['msno','is_churn'],axis=1).loc[val_indices,:],label=df_train.loc[val_indices,'is_churn'])

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting': 'gbdt',
        'learning_rate': 0.1 , #small learn rate, large number of iterations
        'verbose': 0,
        'num_leaves': 108,
        'bagging_fraction': 0.95,
        'bagging_freq': 1,
        'bagging_seed': 1,
        'feature_fraction': 0.9,
        'feature_fraction_seed': 1,
        'max_bin': 128,
        'max_depth': 10,
        'num_rounds': 200,
        }

    bst = lgb.train(params, train_data, 100, valid_sets=[val_data])


predictions = bst.predict(df_test.drop(['msno','is_churn'],axis=1))
df_test['is_churn'] = predictions.clip(0.+1e-15, 1-1e-15)
df_test[['msno','is_churn']].to_csv('LGBM_0.csv',index=False)
