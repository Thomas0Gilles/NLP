import pandas as pd
import numpy as np

#%%
user_logs = pd.read_csv('user_logs_v2.csv')
#%%
print(user_logs.shape)


#df_iter = pd.read_csv('../data/user_logs.csv', low_memory=False, iterator=True, chunksize=10000000)
#last_user_logs = []
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
#last_user_logs.append(transform_df(pd.read_csv('../data/user_logs_v2.csv')))
#last_user_logs = pd.concat(last_user_logs, axis=0, ignore_index=True).reset_index(drop=True)
#last_user_logs = transform_df2(last_user_logs)
