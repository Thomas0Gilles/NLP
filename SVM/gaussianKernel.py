import numpy as np

def gaussianKernel(X1, X2, sigma = 0.1):
    m = X1.shape[0]
    K = np.zeros((m,X2.shape[0]))
    
    for i in range(m):
        K[i,:] = np.exp((-(np.linalg.norm(X1[i,:]-X2, axis=1)**2))/(2*sigma**2))
    
    return K