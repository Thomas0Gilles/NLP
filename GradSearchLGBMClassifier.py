import numpy as np 
import pandas as pd 
import json

from sklearn.pipeline import Pipeline

from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV


#%%
train = pd.read_csv('../train.csv')
test = pd.read_csv('../test.csv')

#%%

id_test = test['id'].values
target_train = train['target'].values

train = train.drop(['target','id'], axis = 1)
test = test.drop(['id'], axis = 1)

col_to_drop = train.columns[train.columns.str.startswith('ps_calc_')]
train = train.drop(col_to_drop, axis=1)  
test = test.drop(col_to_drop, axis=1)  

train = train.replace(-1, np.nan)
test = test.replace(-1, np.nan)

cat_features = [a for a in train.columns if a.endswith('cat')]

for column in cat_features:
	temp = pd.get_dummies(pd.Series(train[column]))
	train = pd.concat([train,temp],axis=1)
	train = train.drop([column],axis=1)
    
for column in cat_features:
	temp = pd.get_dummies(pd.Series(test[column]))
	test = pd.concat([test,temp],axis=1)
	test = test.drop([column],axis=1)

print(train.values.shape, test.values.shape)

#%%

# LightGBM params
lgb_params = {}
lgb_params['learning_rate'] = 0.02
lgb_params['num_iterations'] = 10
lgb_params['subsample'] = 0.8
lgb_params['subsample_freq'] = 1
lgb_params['colsample_bytree'] = 0.8   


#%%

lgb_model = LGBMClassifier(**lgb_params)

pipeline = Pipeline([
    ('classifier', lgb_model)
])

#%% First fit

print("Fitting")

pipeline.get_params().keys()

#hyperparameters = { 'classifier__learning_rate': [0.02, 0.2],
#                    'classifier__num_iterations': [650,1100],
#                    'classifier__subsample': [0.7],
#                    'classifier__subsample_freq': [1,10],
#                    'classifier__colsample_bytree': [0.7,0.9],
#                    'classifier__silent': [False],
#                    'classifier__seed': [200],
#                    'classifier__num_leaves': [16,31],
#                    'classifier__max_depth': [-1, 4],
#                    'classifier__max_bin': [10, 255]
#                  }

hyperparameters = { 'classifier__learning_rate': [0.01, 0.1],
                    'classifier__num_iterations': [800,1300],
                    'classifier__subsample': [0.95],
                    'classifier__subsample_freq': [1,3],
                    'classifier__colsample_bytree': [0.80,0.95],
                    'classifier__silent': [False],
                    'classifier__seed': [500],
                    'classifier__num_leaves': [8,31],
                    'classifier__max_depth': [-1, 2],
                    'classifier__max_bin': [150, 255]
                  }

#hyperparameters = { 'classifier__learning_rate': [0.5, 0.05],
#                    'classifier__num_iterations': [100,800],
#                    'classifier__subsample': [0.8],
#                    'classifier__subsample_freq': [1,7],
#                    'classifier__colsample_bytree': [0.75,0.85],
#                    'classifier__silent': [False],
#                    'classifier__seed': [1000],
#                    'classifier__num_leaves': [20,31],
#                    'classifier__max_depth': [-1, 8],
#                    'classifier__max_bin': [75, 255]
#                  }

#NB: When CV is an integer, it computes cv with stratifiedkfold, hence we don't need to split it before
clf = GridSearchCV(pipeline, hyperparameters, cv = 6, scoring = 'roc_auc')

 
# Fit and tune model
clf.fit(train, target_train)

print("Refiting")

#refitting on entire training data using best settings
clf.refit

bestParam = clf.best_params_

dfg=open("bestParams2.txt",'w')
json.dump(bestParam,dfg)
dfg.close()

print(bestParam)



#%%
y_pred = clf.predict_proba(test)[:,1] 


sub_1 = pd.DataFrame()
sub_1['id'] = id_test
sub_1['target'] = y_pred

#%%

sub_1.to_csv('GridSearchCVLGBM2.csv', index = False)

