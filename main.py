import numpy as np
import matplotlib.pyplot as plt
import syntheticdata

def center(A):
    return A - np.mean(A, axis=0)

def compute_cov(A):
    return np.cov(A.T)

def compute_eig(A):
    eigval, eigvec = np.linalg.eig(A)
    eigval = eigval.real
    eigvec = eigvec.real
    return eigval, eigvec

def sort_eigenvalue_eigenvectors(eigval, eigvec):
    sorted_eigval = np.array(sorted(eigval, reverse=True))
    sorted_eigvec = eigvec[:, eigval.argsort()[::-1]]
    return sorted_eigval, sorted_eigvec

def pca(A,m):
    #Center the data and compute the covariance matrix
    A_centered = center(A)
    cov = compute_cov(A_centered)

    #Compute eigenvectors and sort by eigenvalues
    eigval, eigvec = compute_eig(cov)
    sort_eigval, sort_eigvec = sort_eigenvalue_eigenvectors(eigval, eigvec)

    #Keep the m dim with highest variance.
    pca_eigvec = sort_eigvec[:, :m]
    P = pca_eigvec.T @ A_centered.T

    return pca_eigvec, P.T

#Compression using PCA
def pca_encode(A,m):
    A_centered = center(A)
    cov = compute_cov(A_centered)
    eigval, eigvec = compute_eig(cov)
    sort_eigval, sort_eigvec = sort_eigenvalue_eigenvectors(eigval, eigvec)
    pca_eigvec = sort_eigvec[:, :m]
    P = pca_eigvec.T @ A_centered.T

    A = (pca_eigvec @ P).T + np.mean(A, axis=0)

    return A

if __name__ == "__main__":
    X = syntheticdata.get_synthetic_data1()
    X = center(X)

    
    pca_eigvec = pca(X, 1)[0]
    first_eigvec = pca_eigvec[:, 0]
    
    #Plotting the first eigenvec on the data
    plt.scatter(X[:,0], X[:, 1])

    x = np.linspace(-5, 5, 1000)
    y = first_eigvec[1]/first_eigvec[0] * x
    plt.plot(x,y, c='green')

    #Plotting data projected on the dimension with highest variance
    P = center(pca(X, 1)[1])
    plt.scatter(P[:, 0],[1 for val in P[:, 0]])


    #labeled data set
    plt.scatter(X[:, 0],X[:, 1],c=y[:,0])
    plt.figure()
    X = center(X)
    eignevec, P = pca(X, 1)
    plt.scatter(P[:, 0],np.ones(P.shape[0]),c=y[:,0])

    #Image compression using PCA
    X,y,h,w = syntheticdata.get_lfw_data()
    plt.imshow(X[0,:].reshape((h, w)), cmap=plt.cm.gray)

    X_comp = pca_encode(X,200)
    plt.imshow(X[0,:].reshape((h, w)), cmap=plt.cm.gray)
    plt.figure()
    plt.imshow(X_comp[0,:].reshape((h, w)), cmap=plt.cm.gray)
