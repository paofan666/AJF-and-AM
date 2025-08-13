import numpy as np
import scipy.io
import scipy.linalg
from sklearn.metrics import pairwise

"""
Solve the nuclear matrix K
    ker: A method for solving a kernel function
    X1: The feature matrix of the source domain data
    X2: Feature matrix of the target domain data
    gamma: The parameter of when the kernel function method selects rbf
"""
def kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2 is not None:
            K = pairwise.linear_kernel(np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2 is not None:
            K = pairwise.rbf_kernel(np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = pairwise.rbf_kernel(np.asarray(X1).T, None, gamma)
    return K

class TCA:
    def __init__(self, kernel_type='primal', dim=36, lamb=1, gamma=1):
        """
        :param kernel_type:
        :param dim:
        :param lamb:
        :param gamma:
        """
        self.kernel_type = kernel_type  #选用核函数的类型
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma

    def fit(self, Xs, Xt):
        """
        :param Xs: Feature matrix of the source domain (number of samples x number of features)
        :param Xt: Feature matrix of the target domain (number of samples x number of features)
        :return: After TCA transformation Xs_new,Xt_new
        """
        X = np.hstack((Xs.T, Xt.T))     #X.T means transpose; hstack is to put data of the same dimension together
        X = X/np.linalg.norm(X, axis=0)  #The default is l2 norm, that is, the sum of squares and squares, and is processed as a column vector
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        #Construct the MMD matrix L
        e = np.vstack((1 / ns*np.ones((ns, 1)), -1 / nt*np.ones((nt, 1))))
        M = e * e.T
        M = M / np.linalg.norm(M, 'fro')
        #Construct the center matrix H
        H = np.eye(n) - 1 / n*np.ones((n, n))
        #Construct the kernel function matrix K
        K = kernel(self.kernel_type, X, None, gamma=self.gamma)
        n_eye = m if self.kernel_type == 'primal' else n

        #Note that the kernel function K is the X feature at the end, but it is expressed in the form of a kernel function
        a = np.linalg.multi_dot([K, M, K.T]) + self.lamb * np.eye(n_eye)#XMX_T+lamb*I
        b = np.linalg.multi_dot([K, H, K.T])#XHX_T

        w, V = scipy.linalg.eig(a, b)  #Here we solve the generalized eigenvalue, eigenvector
        ind = np.argsort(w)#The argsort() function arranges the elements in x from small to large, extracts their corresponding indexes, and outputs them to ind
        A = V[:, ind[:self.dim]]#Take the first dim eigenvectors to obtain the transformation matrix A, and arrange them according to the size of the eigenvectors
        Z = np.dot(A.T, K)#Map the data feature to A
        Z /= np.linalg.norm(Z, axis=0)#Unit vector words
        Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T#Source domain features and target domain features are obtained
        return Xs_new, Xt_new

