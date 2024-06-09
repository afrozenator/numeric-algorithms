import numpy as np


def cholesky_elimination_matrix(A):
  n = A.shape[0]
  E = np.zeros_like(A)
  E[0, 0] = 1. / np.sqrt(A[0, 0])
  E[1:, 1:] = np.eye(n - 1)
  E[1:, 0] = -A[1:, 0] / A[0, 0]
  return E


def cholesky_elimination_matrix_inverse(A):
  # No real need to call this, E_inv can be constructed directly from A.
  E = cholesky_elimination_matrix(A)
  E_inv = E
  E_inv[0, 0] = np.sqrt(A[0, 0])
  E_inv[1:, 0] = -1. * np.sqrt(A[0, 0]) * E[1:, 0]
  return E_inv


# A is positive definite, return L such that A = LL^T
def cholesky_decomposition(A):
  A = A.copy()
  # A is a square matrix.
  assert A.shape[0] == A.shape[1], f'A is not square: {A.shape}'
  n = A.shape[0]
  L = np.eye(n)
  for i in range(n):
    E = cholesky_elimination_matrix(A[i:, i:])
    E_inv = cholesky_elimination_matrix_inverse(A[i:, i:])
    # Update the submatrix of A.
    A[i:, i:] = E @ A[i:, i:] @ E.T
    full_e_inv = np.eye(n)
    full_e_inv[i:, i:] = E_inv
    L = L @ full_e_inv
  return L
