import numpy as np 
import pandas as pd 
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

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
transmem = pd.read_csv('../data/trans_mem.csv', dtype={'Unnamed: 0':np.int32,'payment_plan_days':np.float32,'plan_list_price':np.float32,
                                                              'actual_amount_paid':np.float32,'is_auto_renew': np.int8, 'is_cancel': np.float32,
                                                              'trans_count':np.float32,'discount':np.float32,'is_discount':np.int8,'amt_per_day': np.float32,
                                                              'membership_duration':np.float32,'bd':np.float32,'registration_duration': np.float32,'reg_mem_duration':np.float32,
                                                              'autorenew_&_not_cancel':np.int8,'notAutorenew_&_cancel': np.int8,'long_time_user':np.float32,'2':np.int8})


for f in transmem.columns: 
    if transmem[f].dtype=='object': 
        print("type object pour ", f)
        if f!='msno':
            transmem.drop([f],axis=1)
    
train = pd.merge(train, transmem, how='left', on='msno')
test = pd.merge(test, transmem, how='left', on='msno')
del transmem

#%% Merge user_FE
#print("Loading 3 ...")
#userFE = pd.read_csv('../data/user_FE.csv')
#
#for f in userFE.columns: 
#    if userFE[f].dtype=='object': 
#        print("type object pour ", f)
#        if f!='msno':
#            userFE.drop([f],axis=1)
#
#train = pd.merge(train, userFE, how='left', on='msno')
#test = pd.merge(test, userFE, how='left', on='msno')
#del userFE

#%% Create data & label
y = train['is_churn']
X = train.drop(['is_churn','msno'], axis=1)
del train

result = pd.DataFrame()
result['msno'] = test['msno']
test = test.drop(['msno','is_churn'], axis=1)
test = test[X.columns]

N_feature = X.shape[1]
print("Number of features: ",N_feature)
#%%
change_datatype(X)
change_datatype_float(X)

change_datatype(test)
change_datatype_float(test)

#X = X.fillna(0)
#test = test.fillna(0)

print("train set size: ",X.shape)
print("test set size: ",test.shape)

#%%
class Ensemble(object):
    def __init__(self, n_splits, stacker, base_models):
        self.n_splits = n_splits
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        folds = list(StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=2016).split(X, y))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):

            S_test_i = np.zeros((T.shape[0], self.n_splits))

            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]

                print ("Fit %s fold %d" % (str(clf).split('(')[0], j+1))
                clf.fit(X_train, y_train)
                y_pred = clf.predict_proba(X_holdout)[:,1]                

                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict_proba(T)[:,1]
            S_test[:, i] = S_test_i.mean(axis=1)

        results = cross_val_score(self.stacker, S_train, y, cv=3, scoring='roc_auc')
        print("Stacker score: %.5f" % (results.mean()))

        self.stacker.fit(S_train, y)
        res = self.stacker.predict_proba(S_test)[:,1]
        return res



#%%

# LightGBM params
lgb_params = {}
lgb_params['learning_rate'] = 0.02
lgb_params['n_estimators'] = 650
lgb_params['max_bin'] = 10
lgb_params['subsample'] = 0.8
lgb_params['subsample_freq'] = 10
lgb_params['colsample_bytree'] = 0.8   
lgb_params['min_child_samples'] = 500
lgb_params['feature_fraction'] = 0.9
lgb_params['num_iterations']=900
lgb_params['bagging_freq'] = 1
lgb_params['seed'] = 200

lgb_params2 = {}
lgb_params2['n_estimators'] = 1090
lgb_params2['learning_rate'] = 0.02
lgb_params2['colsample_bytree'] = 0.3   
lgb_params2['subsample'] = 0.7
lgb_params2['subsample_freq'] = 2
lgb_params2['num_leaves'] = 16
lgb_params2['num_iterations']=900
lgb_params2['feature_fraction'] = 0.9
lgb_params2['bagging_freq'] = 1
lgb_params2['seed'] = 200


lgb_params3 = {}
lgb_params3['n_estimators'] = 1100
lgb_params3['max_depth'] = 4
lgb_params3['num_iterations']=900
lgb_params3['learning_rate'] = 0.02
lgb_params3['feature_fraction'] = 0.9
lgb_params3['bagging_freq'] = 1
lgb_params3['seed'] = 200

#%%

lgb_model = LGBMClassifier(**lgb_params)

lgb_model2 = LGBMClassifier(**lgb_params2)

lgb_model3 = LGBMClassifier(**lgb_params3)

#%%
log_model = LogisticRegression()
       
stack = Ensemble(n_splits=6,
        stacker = log_model,
        base_models = (lgb_model, lgb_model2, lgb_model3))        
        
y_pred = stack.fit_predict(X, y, test)

result['is_churn'] = y_pred


result.to_csv('LGBM_mini.csv', index=False)
