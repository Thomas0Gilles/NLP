from numpy import *
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from gaussianKernel import gaussianKernel
from pylab import scatter, show, legend, xlabel, ylabel, contour, title, plot
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# # Load the dataset
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = None)

def ginic(actual, pred):
    n = actual.shape[0]
    a_s = actual[np.argsort(pred)]
    a_c = a_s.cumsum()
    giniSum = a_c.sum() / a_c[-1] - (n + 1) / 2.0
    return giniSum / n
 
def gini_normalizedc(a, p):
    return ginic(a, p) / ginic(a, a)


sigma = 0.2 # Gaussian kernel variance

C = 1.0  # SVM regularization parameter
VC = [1.0,3.0,10.0,30.0]
Vsigma = [0.01,0.03,0.1,0.3,1,3,10,30]
Gini = zeros((len(VC),len(Vsigma)))

for i in range(len(VC)):
    for j in range(len(Vsigma)):
        C = VC[i]
        sigma = Vsigma[j]
        svc = SVC(C = C, kernel="precomputed")
        svc.fit(gaussianKernel(X,X,sigma),y)
        
        p = svc.predict(gaussianKernel(Xval,X,sigma))
        counter = 0
        for k in range(yval.size):
            if p[k] == yval[k]:
                counter += 1            
        Acc[i,j] = counter/yval.size
        print('Training accuracy with C=',C,'  and sigma=',sigma,' : ',Acc[i,j])
print(Acc)


