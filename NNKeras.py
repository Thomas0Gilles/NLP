from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import json

# fix random seed for reproducibility
seed = 2015
np.random.seed(seed)

#%% Loading 
print("Loading 1 ...")
train = pd.read_csv('../data/train.csv')
train = pd.concat((train, pd.read_csv('../data/train_v2.csv')), axis=0, ignore_index=True).reset_index(drop=True)
test = pd.read_csv('../data/sample_submission_v2.csv')

#%% Merge trans_mem
print("Loading 2 ...")
transmem = pd.read_csv('../data/trans_mem.csv', usecols=['msno'])

train = pd.merge(train, transmem, how='left', on='msno')
test = pd.merge(test, transmem, how='left', on='msno')
del transmem

#%% Merge user_FE
print("Loading 3 ...")
userFE = pd.read_csv('../data/user_FE.csv', usecols=['msno'])

train = pd.merge(train, userFE, how='left', on='msno')
test = pd.merge(test, userFE, how='left', on='msno')
del userFE

#%% Create data & label
y = train['is_churn']
X = train.drop(['is_churn','msno'], axis=1)
del train

result = pd.DataFrame()
result['msno'] = test['msno']
test = test.drop(['msno','is_churn'], axis=1)

N_feature = X.shape[1]
print("Number of features: ",N_feature)
# We could use CV to improve the result
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2015)

#%%
# Function to create model, required for KerasClassifier
def create_model(optimizer='rmsprop', init='glorot_uniform'):
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=N_feature, kernel_initializer=init, activation='relu'))
	model.add(Dense(8, kernel_initializer=init, activation='relu'))
	model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	return model

model = KerasClassifier(build_fn=create_model, verbose=0)

#%% grid search epochs, batch size and optimizer
optimizers = ['rmsprop', 'adam']
init = ['glorot_uniform', 'normal', 'uniform']
epochs = [50, 100, 150]
batches = [5, 10, 20]
param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, init=init)

clf = GridSearchCV(estimator=model, param_grid=param_grid)

#%% Fitting
clf.fit(X, y)
clf.refit
del X
del y

#%% Export best param
bestParam = clf.best_params_

dfg=open("bestParams1.txt",'w')
json.dump(bestParam,dfg)
dfg.close()

print(bestParam)

#%%
pred = model.predict(test)

#%% Write results

result['is_churn'] = pred
result.to_csv('NN_FE_0.csv', index=False)

