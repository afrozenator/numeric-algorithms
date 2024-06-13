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

  # NOTE: Assumes v is a column vector.

  i.e. Hv = householder_matrix(v, n)
  Hv @ a = reflection of a over v.
  """
  return np.eye(n) - (2 * v @ v.T / (v.T @ v))


# TODO(afro): Write this better.
def householder_transformation(A, debug=False):
  m, n = A.shape
  I = np.eye(m)
  Q = np.eye(m)
  for i in range(n):
    if debug:
      print(f'{i=} {A=}')
    a = A[i:, i]              # (m-i,)
    c = np.linalg.norm(a)     # scalar
    ei = I[i:, i]             # (m-i,)
    v = a - c * ei            # (m-i,)
    v = v[..., np.newaxis]    # (m-i, 1)
    H = householder_matrix(v, m - i)  # (m-i, m-i)
    H_full = np.eye(m)        # (m, m)
    H_full[i:, i:] = H
    Q = Q @ H_full.T          # (m, m)
    if debug:
      with np.printoptions(precision=3):
        print(f'{i=} norm: {c=}')
        print(f'{i=} a - ce1: {v}')
        print(f'{i=} {H=}')
        maybe_I = np.round(H_full @ H_full.T, 3)
        print(f'{i=} H_full @ H_full.T: {maybe_I=}')
    # Overwrite into A.
    np.dot(H_full, A, out=A)
    if debug:
      with np.printoptions(precision=3):
        print(f'{i=} HvA: {np.round(A, 5)}')
      print(f'---')

  # Since we're modifying A in place.
  return Q, A  # A is R.


# TODO(afro): Reduced Householder QR for the least squares case.
