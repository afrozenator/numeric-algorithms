import numpy as np
import cholesky as cd

import unittest


class CholeskyTest(unittest.TestCase):

  def test_elimination_matrix(self):
    A = np.array([
        [4., -2., 4.],
        [-2., 5., -4.],
        [4., -4., 14.]
    ], dtype=np.float64)
    E = cd.cholesky_elimination_matrix(A)
    E_inv = cd.cholesky_elimination_matrix_inverse(A)
    I = E @ E_inv
    np.testing.assert_allclose(I, np.eye(A.shape[0]))

  def test_cholesky(self):
    A = np.array([
        [4., -2., 4.],
        [-2., 5., -4.],
        [4., -4., 14.]
    ], dtype=np.float64)
    L, I = cd.cholesky_decomposition(A)
    np.testing.assert_allclose(A, L @ L.T)
    np.testing.assert_allclose(np.eye(A.shape[0]), I)

  def test_ldlt(self):
    A = np.array([
        [4., -2., 4.],
        [-2., 5., -4.],
        [4., -4., 14.]
    ], dtype=np.float64)
    L, D = cd.ldlt_decomposition(A)
    # Assert that D is diagonal.
    np.testing.assert_allclose(D, np.diag(D) * np.eye(D.shape[0]))
    np.testing.assert_allclose(A, L @ D @ L.T)


if __name__ == '__main__':
  unittest.main()
