import qr

import unittest
import numpy as np


class QRTest(unittest.TestCase):
  def test_gram_schmidt_1(self):
    A = np.array([
        [1., 1., 1.],
        [0., 1., 1.],
        [0., 1., 0.],
    ], dtype=np.float64)
    Q, R = qr.gram_schmidt_1(A, debug=False)
    # Assert that R is upper triangular.
    n = A.shape[1]
    for i in range(n):
      np.testing.assert_allclose(R[i + 1:, i], np.zeros(n - i - 1))
    # Multiply back and check.
    np.testing.assert_allclose(A, Q @ R, atol=1e-10)

  def test_gram_schmidt_0(self):
    A = np.array([
        [1., 1., 1.],
        [0., 1., 1.],
        [0., 1., 0.],
    ], dtype=np.float64)
    Q, R = qr.gram_schmidt_0(A, debug=False)
    # Assert that R is upper triangular.
    n = A.shape[1]
    for i in range(n):
      np.testing.assert_allclose(R[i + 1:, i], np.zeros(n - i - 1))
    # Multiply back and check.
    np.testing.assert_allclose(A, Q @ R, atol=1e-10)

  # def test_householder_matrix(self):
  #   v = np.array([1., 0.])[..., np.newaxis]
  #   n = 2
  #   Hv = qr.householder_matrix(v, n)

  #   y = np.array([0., 1.])[..., np.newaxis]
  #   np.testing.assert_allclose(-1. * y, Hv @ y)

  #   z = np.array([1., 1.])[..., np.newaxis]
  #   z /= np.linalg.norm(z)
  #   # z = [1/sqrt(2), 1/sqrt(2)]
  #   zp = np.array([z[0], -z[1]])
  #   # zp = [1/sqrt(2), -1/sqrt(2)]
  #   np.testing.assert_allclose(zp, Hv @ z)

  def test_householder_transformation(self):
    # Example 5.3
    A_ = np.array([
        [2., -1., 5.],
        [2., 1., 2.],
        [1., 0., -2.],
    ], dtype=np.float64)

    A = np.copy(A_)
    Q, R = qr.householder_transformation(A, debug=False)

    # R is same memory as A.
    assert R is A

    n = A.shape[1]
    # Assert R is upper triangular.
    for i in range(n):
      np.testing.assert_allclose(R[i + 1:, i], np.zeros(n - i - 1), atol=1e-10)

    # Multiply back and check the decomposition.
    np.testing.assert_allclose(A_, Q @ R, atol=1e-10)

  def test_householder_transformation_nonsquare(self):
    # Example 5.3
    A_ = np.array([
        [-1., 5.],
        [1., 2.],
        [0., -2.],
    ], dtype=np.float64)
    m, n = A_.shape
    A = np.copy(A_)
    Q, R = qr.householder_transformation(A, debug=False)

    # R is same memory as A.
    self.assertTrue(R is A)
    self.assertEqual(Q.shape, (m, m))

    # TODO(afro): Assert R is upper triangular.
    # n = A.shape[1]
    # for i in range(n):
    #   np.testing.assert_allclose(R[i + 1:, i], np.zeros(n - i - 1), atol=1e-10)

    # Multiply back and check the decomposition.
    np.testing.assert_allclose(A_, Q @ R, atol=1e-10)

    with np.printoptions(precision=3):
      print(f'{A_=}')
      print(f'{Q=}')
      print(f'{R=}')

  # def test_exercise_5p1(self):
  #   A = np.array([
  #       [1., 1., 1.],
  #       [0., 1., 1.],
  #       [0., 1., 0.],
  #   ], dtype=np.float64)

  #   A_ = A.copy()
  #   Q_h, R_h = qr.householder_transformation(A_, debug=False)

  #   A_ = A.copy()
  #   Q_gs, R_gs = qr.gram_schmidt_1(A_, debug=False)

  #   # Multiply back and check.
  #   np.testing.assert_allclose(A, Q_h @ R_h, atol=1e-10)
  #   np.testing.assert_allclose(A, Q_gs @ R_gs, atol=1e-10)


if __name__ == '__main__':
  unittest.main()
