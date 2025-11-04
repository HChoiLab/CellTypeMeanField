import numpy as np

def wrapped_gaussian(mu,sigma,npoints=2,xpoints=100,xleft=0,xright=1):
    x=np.linspace(xleft,xright,xpoints+1)[:-1]
#     mu = 0.5
#     sigma = 0.5
#     npoints = 2
    xar = np.array([x+n for n in range(-npoints,npoints+1)])
    xn = np.exp(-np.power(xar-mu,2)/(2*sigma*sigma))
    #     print(xn)
    gx = np.sum(xn, axis=0)/np.sqrt(2*np.pi)/sigma
    return x, gx

def l2_deviation(vector1, vector2):
    difference = np.array(vector1) - np.array(vector2)
    l2_dev = np.sqrt(np.sum(difference ** 2))
    return l2_dev

def minor(matrix, i, j):
    """Return the minor of the matrix after removing the i-th row and j-th column."""
    minor_matrix = np.delete(np.delete(matrix, i, axis=0), j, axis=1)
    return np.linalg.det(minor_matrix)

def cofactor_matrix(matrix):
    """Return the cofactor matrix of the given square matrix."""
    n = matrix.shape[0]
    cofactors = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            minor_ij = minor(matrix, i, j)
            cofactors[i, j] = (-1) ** (i + j) * minor_ij
    return cofactors

def g_bar(n, mu, sigma):
    return np.exp(-2*n**2*np.pi**2*sigma**2 - 2j*n*np.pi*mu)