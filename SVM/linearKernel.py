import numpy as np

def linearKernel(X1, X2):
    m = X1.shape[0]
    K = np.zeros((m,X2.shape[0]))
    
    for i in range(m):
        K[i,:] = np.dot(X2, X1[i,:])
    return K
