from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd

# fix random seed for reproducibility
seed = 2015
np.random.seed(seed)
#%% Loading 
train = pd.read_csv('../data/train.csv')
train = pd.concat((train, pd.read_csv('../data/train_v2.csv')), axis=0, ignore_index=True).reset_index(drop=True)
test = pd.read_csv('../data/sample_submission_v2.csv')

#%%
transmem = pd.read_csv('../data/trans_mem.csv', usecols=['msno'])

train = pd.merge(train, transmem, how='left', on='msno')
test = pd.merge(test, transmem, how='left', on='msno')

#%%
userFE = pd.read_csv('../data/user_FE.csv', usecols=['msno'])

train = pd.merge(train, userFE, how='left', on='msno')
test = pd.merge(test, userFE, how='left', on='msno')


#%%
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, epochs=150, batch_size=10,  verbose=2)
# calculate predictions
predictions = model.predict(X)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)
