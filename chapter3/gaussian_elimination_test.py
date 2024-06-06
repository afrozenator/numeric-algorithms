import unittest

import gaussian_elimination as ge
import numpy as np


# TODO(afro): Parametrize the tests.
class GaussianEliminationTest(unittest.TestCase):
  def test_pivoting(self):
    A = np.array([[0, -3, 5], [1, 2, 4], [2, 7, 6]], dtype=np.float64)
    b = np.array([0, 1, 2], dtype=np.float64)
    ge.pivot(A, b, partial_pivot=True)
    np.testing.assert_allclose(
        A,
        np.array([[2, 7, 6], [0, -3, 5], [1, 2, 4]], dtype=np.float64)
    )
    np.testing.assert_allclose(b, np.array([2, 0, 1], dtype=np.float64))

  def test_pivoting_batched_b(self):
    A_nn = np.array(
        [[1, 1, -2],
         [0, 1, -1],
         [3, -1, 1]],
        dtype=np.float64)
    # Does `nb` work?
    b_nb = np.array(
        [
            [-3, 3],
            [-1, 1],
            [4, 8],
        ],
        dtype=np.float64)
    ge.pivot(A_nn, b_nb, partial_pivot=True)
    np.testing.assert_allclose(
        A_nn,
        np.array(
            [
                [3, -1, 1],
                [0, 1, -1],
                [1, 1, -2],
            ],
            dtype=np.float64)
    )
    np.testing.assert_allclose(
        b_nb,
        np.array(
            [
                [4, 8],
                [-1, 1],
                [-3, 3],
            ],
            dtype=np.float64)
    )

  def test_forward_substitution(self):
    A = np.array(
        [[1, 1, -2],
         [0, 1, -1],
         [3, -1, 1]],
        dtype=np.float64)
    b = np.array([-3, -1, 4], dtype=np.float64)
    U, y = ge.forward_substitution(A, b)
    np.testing.assert_allclose(
        U,
        np.array(
            [[1, -1./3, 1/3],
             [0, 1, -1],
             [0, 0, 1]],
            dtype=np.float64)
    )
    y = np.squeeze(y)
    np.testing.assert_allclose(y, np.array([4./3, -1, 3], dtype=np.float64))

    # Testing a different `b`
    b = np.array([3, 1, 8], dtype=np.float64)
    U, y = ge.forward_substitution(A, b)
    np.testing.assert_allclose(
        U,
        np.array(
            [[1, -1./3, 1/3],
             [0, 1, -1],
             [0, 0, 1]],
            dtype=np.float64)
    )
    y = np.squeeze(y)
    np.testing.assert_allclose(y, np.array([8./3, 1, 1], dtype=np.float64))

  def test_forward_substitution_batched_b(self):
    A_nn = np.array(
        [[1, 1, -2],
         [0, 1, -1],
         [3, -1, 1]],
        dtype=np.float64)
    b_nb = np.array(
        [
            [-3, 3],
            [-1, 1],
            [4, 8],
        ],
        dtype=np.float64)
    U, y = ge.forward_substitution(A_nn, b_nb)
    np.testing.assert_allclose(
        U,
        np.array(
            [[1, -1./3, 1/3],
             [0, 1, -1],
             [0, 0, 1]],
            dtype=np.float64)
    )
    np.testing.assert_allclose(
        y,
        np.array(
            [
                [4./3, 8./3],
                [-1, 1],
                [3, 1],
            ],
            dtype=np.float64)
    )

  def test_backward_substitution(self):
    U = np.array([[1, 1, -2], [0, 1, -1], [0, 0, 1]], dtype=np.float64)
    y = np.array([-3, -1, 3], dtype=np.float64)

    A, b = ge.back_substitution(U, y)
    np.testing.assert_allclose(A, np.eye(3))
    b = np.squeeze(b)
    np.testing.assert_allclose(b, np.array([1, 2, 3], dtype=np.float64))

  def test_gaussian_elimination(self):
    A = np.array([[1, 1, -2], [0, 1, -1], [3, -1, 1]], dtype=np.float64)
    b = np.array([-3, -1, 4], dtype=np.float64)
    U, y = ge.forward_substitution(A, b)
    # y = np.squeeze(y)
    I, s = ge.back_substitution(U, y)
    np.testing.assert_allclose(I, np.eye(3))
    s = np.squeeze(s)
    np.testing.assert_allclose(s, np.array([1, 2, 3], dtype=np.float64))

    # Check with a different `b` but the same `A`
    b = np.array([3, 1, 8], dtype=np.float64)
    U, y = ge.forward_substitution(A, b)
    I, s = ge.back_substitution(U, y)
    np.testing.assert_allclose(I, np.eye(3))
    s = np.squeeze(s)
    np.testing.assert_allclose(s, np.array([3, 2, 1], dtype=np.float64))

  def test_gaussian_elimination_batched(self):
    A_nn = np.array(
        [[1, 1, -2],
         [0, 1, -1],
         [3, -1, 1]],
        dtype=np.float64)
    b_nb = np.array(
        [
            [-3, 3],
            [-1, 1],
            [4, 8],
        ],
        dtype=np.float64)
    U, y = ge.forward_substitution(A_nn, b_nb)
    I, s = ge.back_substitution(U, y)
    np.testing.assert_allclose(I, np.eye(3))
    np.testing.assert_allclose(
        s,
        np.array(
            [
                [1, 3],
                [2, 2],
                [3, 1],
            ],
            dtype=np.float64)
    )


if __name__ == '__main__':
  unittest.main()
