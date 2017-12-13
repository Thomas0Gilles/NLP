import pandas as pd
import numpy as np

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

print('Loading transmem ...')
transmem = pd.read_csv('../data/trans_mem.csv')
print(transmem.columns.values)

change_datatype_float(transmem)
change_datatype(transmem)

print(transmem.dtypes)
del transmem


print("Loading user_FE ...")
userFE = pd.read_csv('../data/user_FE.csv')
print(userFE.columns.values)

change_datatype_float(userFE)
change_datatype(userFE)

print(userFE.dtypes)