from sklearn import svm
import numpy as np
import pandas as pd
import pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

test = pd.read_csv('../test.csv',sep=',')
train = pd.read_csv('../train.csv',sep=',')

variables = train.columns.values
p = train.shape[1]
variables_cat = []
variables_bin = []
variables_other = []
number_features = 0
for i in range(2,p):
    name = variables[i]
    if(name.find("cat")!=-1):
        variables_cat.append(name)
        number_features += max(train[name])+1  #Supposition: de 0 à p pour variable catégorique (p+1 different feature)
        print("Variable ", name," a ", max(train[name])+1, " features")
    elif(name.find("bin")!=-1):
        variables_bin.append(name)
        number_features += 1
    else:
        variables_other.append(name)
        number_features += 1

Mean = np.zeros(len(variables_other))
Std = np.zeros(len(variables_other))
for i in range(len(variables_other)):
    Mean[i] = np.mean(train[variables_other[i]])
    Std[i] = np.std(train[variables_other[i]])
    train[variables_other[i]] = (train[variables_other[i]]-Mean[i])/Std[i]
    test[variables_other[i]] = (test[variables_other[i]]-Mean[i])/Std[i]  #We scale the test_set with the same Mean and Std
    test[variables_other[i]] = (test[variables_other[i]]-Mean[i])/Std[i]

y = train.iloc[:,1].values
X_submit_id = test.iloc[:,0].values
X = np.zeros((train.shape[0],number_features))
X_submit = np.zeros((test.shape[0],number_features))

for p in range(len(variables_other)):
    X[:,p] = train[variables_other[p]].values
    X_submit[:,p] = test[variables_other[p]].values
count = len(variables_other)
for p in range(len(variables_bin)):
    X[:,p+count] = train[variables_bin[p]].values
    X_submit[:,p+count] = test[variables_bin[p]].values
count += len(variables_bin)
for p in range(len(variables_cat)):
    size = max(train[variables_cat[p]])+1
    z = train[variables_cat[p]].values
    z_submit = test[variables_cat[p]].values
    for i in range(p):
        X[np.where(z==i),count+i] = 1
        X_submit[np.where(z_submit==i),count+i] = 1
    count+=size

print(X.shape)
print(X[:,X.shape[1]-1])
print(X[:,X.shape[1]-2])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = None)


def gini(actual, pred, cmpcol = 0, sortcol = 1):
    all = np.asarray(np.c_[ actual, pred, np.arange(actual.shape[0]) ], dtype=np.float)
    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
    totalLosses = all[:,0].sum()
    giniSum = all[:,0].cumsum().sum() / totalLosses
 
    giniSum -= (actual.shape[0] + 1) / 2.
    return giniSum / actual.shape[0]
    
def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)

def myTree(D,T=100):
    w = np.ones(X_train.shape[0]) / X_train.shape[0]
    y_pred = np.zeros(X_train.shape[0])
    y_pred_test = np.zeros(X_test.shape[0])
    y_submit_pred = np.zeros(X_submit.shape[0])
    Y_pred = np.zeros(X_train.shape[0])
    Y_pred_test = np.zeros(X_test.shape[0])
    Y_submit_pred = np.zeros(X_submit.shape[0])
    
    for t in range(T):
        print("Tree:",t,"  Depth:",D)
        clf = DecisionTreeClassifier(max_depth=D)
        clf.fit(X_train, y_train, sample_weight=w)
        y_pred = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)   
        y_submit_pred = clf.predict(X_submit)
        diff = (y_pred[t]!=y_train)
        gamma= np.dot(w,diff)/sum(w)
        alpha = np.log((1-gamma)/gamma)
        w = w * np.exp(diff*alpha)
        Y_pred += alpha*y_pred
        Y_pred_test += alpha*y_pred_test
        Y_submit_pred += alpha*y_submit_pred
    
    #Etant donné que je fais du gini cette étape est-elle juste ou le résultat sera pourri?
    
    for i in range(len(Y_pred_test)):
        if(Y_pred_test[i]>0):
            Y_pred_test[i] = 1
        else:
            Y_pred_test[i] = -1
    
    for i in range(len(Y_pred)):
        if(Y_pred[i]>0):
            Y_pred[i] = 1
        else:
            Y_pred[i] = -1
            
    return gini_normalized(y_test,Y_pred_test), gini_normalized(y_train,Y_pred), Y_submit_pred
#
#D = [1,3,10,20,30,100]
#test_errors = np.zeros(len(D))
#training_errors = np.zeros(len(D))
#Y_submit = [np.zeros(X_submit.shape[0]) for i in range(len(D))]
#for i in range(len(D)):
#    test, train, Y_submit[i] = myTree(D[i])
#    test_errors[i] = test
#    training_errors[i] = train
#
#plt.plot(D,training_errors, label="training error")
#plt.plot(D,test_errors, label="test error")
#plt.legend()
#plt.show()
#
#y_submit = Y_submit[np.argmin(test_errors)]
#res=[['id','target']]
#for i in range(len(X_submit)):
#    pair=[X_submit_id[i],y_submit[i]]
#    res.append(pair)
#
#submission = pd.DataFrame(res)
#submission.to_csv('submission.csv', index=False, header=False) 
