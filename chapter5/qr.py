import numpy as np


def gram_schmidt_0(A, debug=False):
  m, n = A.shape
  del debug, m
  Q = np.copy(A)
  R = np.zeros((n, n))
  for j in range(n):
    R[j, j] = np.linalg.norm(Q[:, j])
    Q[:, j] /= R[j, j]
    for i in range(j + 1, n):
      R[j, i] = np.dot(Q[:, i], Q[:, j])
      Q[:, i] -= R[j, i] * Q[:, j]
  return Q, R


def gram_schmidt_1(A, debug=False):
  # We will do this column by column and make both Q and R.
  m, n = A.shape
  del m
  Q = np.copy(A)
  R = np.zeros((n, n))

  for j in range(n):
    col_j = Q[:, j]
    norm_col_j = np.linalg.norm(col_j)
    # Overwrites.
    Q[:, j] = col_j / norm_col_j  # (m,)
    R[j, j] = norm_col_j

    if debug:
      print(f'At column {j} norm is {norm_col_j}')
      print(f'Normalized column: {Q[:, j]=}')

    post_j_columns = Q[:, j + 1:]  # (m, n-j-1)
    dot_products = np.einsum('i,ij->j', Q[:, j], post_j_columns)  # (n-j-1,)

    if debug:
      print(f"Column {j}'s {dot_products=}")

    R[j, j + 1:] = dot_products
    Q[:, j + 1:] -= np.einsum('m,n->mn', Q[:, j], dot_products)

    if debug:
      print(f'After column {j}\n{Q=}\n{R=}')

  return Q, R


def householder_matrix(v, n):
  """Matrix that reflects a vector over v.

  i.e. Hv = householder_matrix(v, n)
  Hv @ a = reflection of a over v.
  """
  return np.eye(n) - (2 * v @ v.T / (v.T @ v))


def householder_transformation(A, debug=False):
  m, n = A.shape
  # First column of A.
  a = A[:, 0]
  c = np.linalg.norm(a)
  e1 = np.eye(n)[:, 0]
  v = a - c * e1          # (n,)
  v = v[..., np.newaxis]  # (n, 1)
  H = householder_matrix(v, n)
  if debug:
    print(f'norm: {c=}')
    print(f'a - ce1: {v}')
    print(f'{H=}')
  return H @ A, v
