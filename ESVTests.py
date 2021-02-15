import unittest
import EspaciosVectoriales as esv
import numpy as np

class TestStringMethods(unittest.TestCase):


    def test_sum_mat(self):
        mat1 = np.array([[1+2j], [5+1j], [7+8j]])
        mat2 = np.array([[4+3j], [1+2j], [3+2j]])
        res = np.array([[5.+5.j], [6.+3.j], [10.+10.j]])
        self.assertTrue(
            np.testing.assert_almost_equal(esv.mat_sum(mat1, mat2), res) is None)


    def test_inverse(self):
        mat = np.array([[2+5j], [8+2j], [6+4j]])
        res = np.array([[-2.-5.j], [-8.-2.j], [-6.-4.j]])
        self.assertTrue(
            np.testing.assert_almost_equal(esv.inverse(mat), res) is None
        )


    def test_sc_mult(self):
        scalar = 2+8j
        mat = np.array([[3+2j ,5+6j ,1+3j],
                        [5+8j, 8+5j, 9+1j],
                        [3+7j, 3+1j, 2+4j]])
        res = np.array([[-10.+28.j, -38.+52.j, -22.+14.j],
                        [-54.+56.j, -24.+74.j,  10.+74.j],
                        [-50.+38.j,  -2.+26.j, -28.+24.j]])
        self.assertTrue(
            np.testing.assert_almost_equal(esv.sc_mult(scalar, mat), res) is None
        )


    def test_conj(self):
        mat = np.array([[2+1j, 5+2j], [9+2j, 6+2j]])
        res = np.array([[2.-1.j, 5.-2.j], [9.-2.j, 6.-2.j]])
        self.assertTrue(
            np.testing.assert_almost_equal(esv.conjugate(mat), res) is None
        )


    def test_tran(self):
        mat = np.array([1+2j, 6+2j, 2+9j, 7+1j])
        res = np.array([[1+2j], [6+2j], [2+9j], [7+1j]])
        self.assertTrue(
            np.testing.assert_almost_equal(esv.transpose(mat), res) is None
        )


    def test_adj(self):
        mat = np.array([[1+2j, 8+3j], [4+4j, 2+5j], [9+4j, 9+3j]])
        res = np.array([[1-2j, 4-4j, 9-4j], [8-3j, 2-5j, 9-3j]])
        self.assertTrue(
            np.testing.assert_almost_equal(esv.adjoint(mat), res) is None
        )


    def test_mat_mult(self):
        mat1 = np.array([[1+2j, 4+7j], [3+9j, 3+3j]])
        mat2 = np.array([[7+2j, 5+8j], [2+6j, 2+2j]])
        res = np.array([[-31.+54.j, -17.+40.j],
                        [-9.+93.j, -57.+81.j]])
        self.assertTrue(
            np.testing.assert_almost_equal(esv.mat_mult(mat1, mat2), res)
            is None
        )


    def test_inner_pr(self):
        mat1 = np.array([2+5j, 8+5j])
        mat2 = np.array([1+1j, 3+6j])
        res = np.array([[ 7. -3.j, 36. -3.j], [13. +3.j, 54.+33.j]])
        self.assertTrue(
            np.testing.assert_almost_equal(esv.inner_prod(mat1, mat2),res) is None
        )


    def test_norm(self):
        mat = np.array([[1 + 9j], [4 + 7j], [2+6j], [9+2j]])
        res = 16.492422502470642
        self.assertTrue(
            np.testing.assert_almost_equal(esv.norm(mat), res) is None
        )


    def test_unitary(self):
        mat = np.array([[1j, 0, 0], [0, 1j, 0], [0, 0, 1j]])
        res = True
        self.assertTrue(esv.unitary(mat), res)


    def test_hermitian(self):
        mat = np.array([[5, 3-7j], [3+7j, 2]])
        res = False
        self.assertTrue(esv.hermitian(mat), res)

    def test_tensor_p(self):
        mat1 = np.array([[2 + 3j], [5 + 2j], [8 + 2j]])
        mat2 = np.array([[1 + 4j], [8 + 7j], [2 + 9j]])
        res = np.array([[-10.+11.j, -5.+38.j, -23. +24.j],
                        [-72.+35.j, -101.+180.j, -163.+74.j],
                        [-646.+136.j, -1168.+1238.j, -1452.+266.j]])
        self.assertTrue(
            np.testing.assert_almost_equal(esv.tensor_prod(mat1, mat2), res) is None
        )


if __name__ == '__main__':
    unittest.main()