import unittest

import gaussian_elimination as ge
import numpy as np


class GaussianEliminationTest(unittest.TestCase):
  def test_pivoting(self):
    A = np.array([[0, 1, -1], [1, 1, -2], [3, -1, 1]], dtype=np.float64)
    b = np.array([-1, -3, 4], dtype=np.float64)
    ge.pivot(A, b, partial_pivot=True)
    print(A)
    print()
    print(b)

  def test_forward_substitution(self):
    A = np.array([[1, 1, -2], [0, 1, -1], [3, -1, 1]], dtype=np.float64)
    b = np.array([-3, -1, 4], dtype=np.float64)
    U, y = ge.forward_substitution(A, b)

    np.testing.assert_allclose(
        U,
        np.array([[1, 1, -2], [0, 1, -1], [0, 0, 1]], dtype=np.float64)
    )
    np.testing.assert_allclose(y, np.array([-3, -1, 4./3], dtype=np.float64))

  def test_backward_substitution(self):
    U = np.array([[1, 1, -2], [0, 1, -1], [0, 0, 1]], dtype=np.float64)
    y = np.array([-3, -1, 3], dtype=np.float64)

    A, b = ge.back_substitution(U, y)
    np.testing.assert_allclose(A, np.eye(3))
    np.testing.assert_allclose(b, np.array([1, 2, 3], dtype=np.float64))


if __name__ == '__main__':
  unittest.main()
