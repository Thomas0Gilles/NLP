import numpy as np 
import pandas as pd 
from sklearn.pipeline import Pipeline

from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from sklearn.metrics import make_scorer

#%%
train = pd.read_csv('../train.csv')
test = pd.read_csv('../test.csv')

#%% Gini
def gini(actual, pred, cmpcol = 0, sortcol = 1):
    all = np.asarray(np.c_[ actual, pred, np.arange(actual.shape[0]) ], dtype=np.float)
    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
    totalLosses = all[:,0].sum()
    giniSum = all[:,0].cumsum().sum() / totalLosses
 
    giniSum -= (actual.shape[0] + 1) / 2.
    return giniSum / actual.shape[0]
    
def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)

gini_score = make_scorer(gini_normalized,greater_is_better=True)
gini_loss  = make_scorer(gini_normalized, greater_is_better=False)

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

X_train, X_test, y_train, y_test = train_test_split(train, target_train, test_size=0.2, random_state=42)

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

hyperparameters = { 'classifier__learning_rate': [0.02, 0.2],
                    'classifier__num_iterations': [100,650,1100],
                    'classifier__subsample': [0.7,0.9],
                    'classifier__subsample_freq': [1,10],
                    'classifier__colsample_bytree': [0.7,0.9],
                    'classifier__silent': [False]
                  }
clf = GridSearchCV(pipeline, hyperparameters, cv = 5, scoring = gini_score)

 
# Fit and tune model
clf.fit(X_train, y_train)

#refitting on entire training data using best settings
clf.refit

probs = clf.predict_proba(X_test)[:,1] 

Gini_test = gini_normalized(y_test,probs)
print("Refit, Gini normalized = ", Gini_test)


#%%
y_pred = clf.predict_proba(test)[:,1] 

print(id_test.shape)
print(y_pred.shape)

sub_1 = pd.DataFrame()
sub_1['id'] = id_test
sub_1['target'] = y_pred

#%%

sub_1.to_csv('GradSearchLGBM.csv', index = False)

