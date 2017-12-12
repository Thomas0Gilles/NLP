import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
import gc; gc.enable()

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

#%%
print('Import')

df_iter = pd.read_csv('../data/user_logs.csv', low_memory=False, iterator=True, chunksize=10000000)
last_user_logs = []
i = 0 #~400 Million Records - starting at the end but remove locally if needed
for df in df_iter:
    print("New chunk !")
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
print('Concat')
last_user_logs.append(transform_df(pd.read_csv('../data/user_logs_v2.csv')))
last_user_logs = pd.concat(last_user_logs, axis=0, ignore_index=True).reset_index(drop=True)
last_user_logs = transform_df2(last_user_logs)
print("Before selection: ",last_user_logs.shape)

#%%
print('Conversion')
date_cols = ['date']
for col in date_cols:
    last_user_logs[col] = pd.to_datetime(last_user_logs[col], format='%Y%m%d')

#%%
print('Selection')
Fe2017 = pd.to_datetime(20170102, format='%Y%m%d')
last_user_logs = last_user_logs[last_user_logs['date'] >= Fe2017]

#Je sais pas si c'ets vraiment biend e drop la date mais je ne voi pas comment un NN l'exploite. Donc je drop toutes les dates
last_user_logs = last_user_logs.drop(['date'], axis=1)
print("After selection: ",last_user_logs.shape)

#%%
print('Aggregation')
last_user_logs = last_user_logs.groupby(last_user_logs.msno).agg({'num_25': ['sum','std','median','mean','nunique'], 'num_50':['sum','std','median','mean'],'num_75':['sum','std','median','mean'],'num_985':['sum','std','median','mean'], 'num_100':['sum','std','median','mean'],'num_unq':['sum','std','median','mean'], 'total_secs':['sum','std','median','mean']})

#%%
print('Scale')
for c in last_user_logs.columns:
    moy = np.nanmean(last_user_logs[c])
    last_user_logs[c] = (last_user_logs[c] - moy)/np.sqrt(np.nansum(np.square(last_user_logs[c] - moy)))

last_user_logs['msno'] = last_user_logs.index.values
print("At the end: ",last_user_logs.shape)

#%%
print("Write ...")
last_user_logs.to_csv('../data/user_FE_scaled.csv')


