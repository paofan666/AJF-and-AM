import numpy as np
import scipy.io
import scipy.linalg
from sklearn.metrics import pairwise

"""
求解核矩阵K
    ker:求解核函数的方法
    X1:源域数据的特征矩阵
    X2:目标域数据的特征矩阵
    gamma:当核函数方法选择rbf时，的参数
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
        :param Xs: 源域的特征矩阵 （样本数x特征数）
        :param Xt: 目标域的特征矩阵 （样本数x特征数）
        :return: 经过TCA变换后的Xs_new,Xt_new
        """
        X = np.hstack((Xs.T, Xt.T))     #X.T是转置的意思；hstack则是将数据的相同维度数据放在一起
        X = X/np.linalg.norm(X, axis=0)  #求范数默认为l2范数即平方和开方，按列向量处理，这里相当于
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        #构造MMD矩阵 L
        e = np.vstack((1 / ns*np.ones((ns, 1)), -1 / nt*np.ones((nt, 1))))
        M = e * e.T
        M = M / np.linalg.norm(M, 'fro')
        #构造中心矩阵H
        H = np.eye(n) - 1 / n*np.ones((n, n))
        #构造核函数矩阵K
        K = kernel(self.kernel_type, X, None, gamma=self.gamma)
        n_eye = m if self.kernel_type == 'primal' else n

        #注意核函数K就是后边的X特征，只不过用核函数的形式表示了
        a = np.linalg.multi_dot([K, M, K.T]) + self.lamb * np.eye(n_eye)#XMX_T+lamb*I
        b = np.linalg.multi_dot([K, H, K.T])#XHX_T

        w, V = scipy.linalg.eig(a, b)  #这里求解的是广义特征值，特征向量
        ind = np.argsort(w)#argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到ind
        A = V[:, ind[:self.dim]]#取前dim个特征向量得到变换矩阵A，按照特征向量的大小排列好,
        Z = np.dot(A.T, K)#将数据特征*映射A
        Z /= np.linalg.norm(Z, axis=0)#单位向量话
        Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T#得到源域特征和目标域特征
        return Xs_new, Xt_new

