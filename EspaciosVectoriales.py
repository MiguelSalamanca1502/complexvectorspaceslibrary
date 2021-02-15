import numpy as np
import math


def mat_sum(mat1, mat2):
    """
    Sum of matrices
    :param mat1: 1st matrix
    :param mat2: 2nd matrix
    :return: matrix
    """
    if mat1.ndim == 1:
        res = np.full(len(mat1))  # res: result of the sum
        for i in range(len(mat1)):
            res[i] = mat1[i] + mat2[i]
    else:
        res = np.full((len(mat1), len(mat1[0])), 0j)
        for i in range(len(mat1)):
            for j in range(len(mat1[0])):
                res[i][j] = mat1[i][j] + mat2[i][j]

    return res


def inverse(mat):
    """
    Inverse of a matrix
    :param mat: Matrix
    :return: Matriz
    """
    for i in range(len(mat)):
        mat[i] = -1 * mat[i]

    return mat


def sc_mult(sca, mat):  # Scalar multiplication
    """
    Scalar multiplication over a matrix
    :param sca: Scalar
    :param mat: Matrix
    :return: Matrix
    """
    if mat.ndim > 1:
        for i in range(len(mat)):
            for j in range(len(mat[0])):
                mat[i][j] = sca * mat[i][j]
        return mat
    else:
        for i in range(len(mat)):
            mat[i] = sca * mat[i]
        return mat


def conjugate(mat):
    """
    Conjugate of a matrix
    :param mat: mat
    :return: mat
    """
    if mat.ndim > 1:
        res = np.full((len(mat), len(mat[0])), 0j)
        for i in range(len(mat)):
            for k in range(len(mat[0])):
                (a, b) = mat[i][k].real, -1*mat[i][k].imag
                res[i][k] = complex(a, 0)+complex(0, b)
    else:
        res = np.full(len(mat), 0j)
        for i in range(len(mat)):
            a, b = mat[i].real, -1*mat[i].imag
            res[i] = complex(a, 0)+complex(0, b)
    return res


def transpose(mat):
    """
    Transpose of a matrix
    :param mat: matrix
    :return: matrix
    """
    if mat.ndim > 1:
        res = np.full((len(mat[0]), len(mat)), 0j)  # res: result of the
        if len(mat[0]) > 1:                         # tranpose of the given
            for i in range(len(mat)):               # matrix
                for j in range(len(mat[0])):
                    res[j][i] = mat[i][j]
            return res
        else:
            for i in range(len(mat)):
                for j in range(len(mat[0])):
                    res[j][i] = mat[i][j]
            return res[0]
    else:
        res = np.full((len(mat), 1), 0j)
        for i in range(len(mat)):
            res[i][0] = mat[i]
        return res


def adjoint(mat):  # Adjoint of a matrix
    """
    Adjoint of a matrix
    :param mat: matrix
    :return: matrix
    """
    return transpose(conjugate(mat))


def mat_mult(mat1, mat2):
    """
    Matrix multiplication
    :param mat1: 1st matrix
    :param mat2: 2nd matrix
    :return: matrix
    """
    if mat1.ndim > 1 and mat2.ndim > 1:
        res = np.full((len(mat1), len(mat2[0])), 0j)
        for k in range(len(mat1)):
            for i in range(len(mat2[0])):
                for j in range(len(mat2)):
                    res[k][i] += mat1[k][j]*mat2[j][i]
    elif mat1.ndim > 1 and mat2.ndim == 1:
        res = np.full((len(mat1), len(mat2)), 0j)
        for k in range(len(mat1)):
            for j in range(len(mat2)):
                res[k][j] = mat1[k][0]*mat2[j]
    else:
        print("here")
        res = np.full(len(mat2[0]), 0j)
        for i in range(len(mat2[0])):
            for j in range(len(mat2)):
                res[i] += mat1[j]*mat2[j][i]
    return res


def inner_prod(mat1, mat2):
    """
    Inner product between two matrices
    :param mat1: 1st Matrix
    :param mat2: 2nd Matrix
    :return: Matrix
    """
    return mat_mult(adjoint(mat1), mat2)


def norm(mat):
    """
    Norm (lenght) of a matrix
    :param mat: matrix
    :return: float
    """
    return math.sqrt(inner_prod(mat, mat).real)


def distance(mat1, mat2):
    """
    Distance between two matrices
    :param mat1: 1st matrix
    :param mat2: 2nd matrix
    :return: float
    """
    return norm(mat_sum(mat1, inverse(mat2)))


def unitary(mat):
    """
    Checks if a matrix is unitary
    :param mat: Matrix
    :return: Boolean
    """
    k = True
    adj = adjoint(mat)
    unit = mat_mult(adj, mat)
    for i in range(len(unit)):
        if np.round(unit[i][i].real) != 1:
            k = False
    return k


def hermitian(mat):
    """
    Checks if a matrix is hermitian
    :param mat: Matrix
    :return: Boolean
    """
    k = True
    adj = adjoint(mat)
    for i in range(len(mat)):
        for j in range(len(mat)):
            if np.round(adj[i][j]) != np.round(mat[i][j]):
                k = False

    return "Is hermitian" if k else "Is not hermitian"


def tensor_prod(mat1, mat2):
    """
    Calculates the tensor product of two matrices
    :param mat1: 1st matrix
    :param mat2: 2nd matrix
    :return: matrix
    """
    res = []
    for i in range(len(mat1)):
        row = []
        for j in range(len(mat1[i])):
            mat = sc_mult(mat1[i][j], mat2)
            for k in range(len(mat)):
                for n in range(len(mat[k])):
                    row.append(mat[k][n])
        res.append(row)
    res = np.array(res)
    return res
