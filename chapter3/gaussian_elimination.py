import numpy as np


def _assert_square_and_return_dim(A):
  assert A.ndim == 2, f'A is not dim 2, {A.ndim=}'
  assert A.shape[0] == A.shape[1], f'A is not square, {A.shape=}'
  n = A.shape[0]
  return n


# Only does row re-orderings, not column - and only for one column.
def _partial_pivot(A, b, col=0):
  remaining_column = A[col:, col]

  max_abs_idx = np.argmax(np.abs(remaining_column))
  # Since the above is indexed from `col`
  max_abs_idx += col

  if max_abs_idx == col:
    # i.e. this is already in the correct order.
    return

  # swap max_abs_idx and col.
  tmp = np.copy(A[col])
  A[col] = A[max_abs_idx]
  A[max_abs_idx] = tmp
  tmp = np.copy(b[col])
  b[col] = b[max_abs_idx]
  b[max_abs_idx] = tmp


# Pivoting the matrix for stability.
def pivot(A, b, partial_pivot=True):
  if partial_pivot:
    n = A.shape[0]
    for i in range(n):
      _partial_pivot(A, b, col=i)
  else:
    print('Full pivoting is not implemented.')
  return A, b


# Converts `Ax = b` into `Ux = y` where `U` is upper triangular.
def forward_substitution(A_nn, b_n, debug=False):
  A = np.copy(A_nn)
  b = np.copy(b_n)
  A, b = pivot(A, b, partial_pivot=True)
  n = _assert_square_and_return_dim(A)
  if debug:
    print(f'Init: {A=}')
    print(f'Init: {b=}')
  for i in range(n):
    v = A[i, i]
    # Make element at the diagonal on the i-th row as 1.
    A[i] /= v
    b[i] /= v
    # Let's update for the rows below this.
    coeffs = A[i+1:, i]  # shape: (n - i - 1,)
    subtract_A = A[i] * coeffs[..., None]  # (n,) x (n-i-1, 1) -> (n-i-1, n)
    subtract_b = b[i] * coeffs  # shape: (n - i - 1,)
    # Do all the updates in the very end, since `coeffs` etc are views into
    # the array and not copies.
    A[i+1:] -= subtract_A
    b[i+1:] -= subtract_b
    if debug:
      print(f'{coeffs=}')
      print(f'{subtract_A=}')
      print(f'{b[i]=}')
      print(f'{subtract_b=}')
      print(f'{i}: {A=}')
      print(f'{i}: {b=}')
  return A, b


# Solves `Ux = y`  where U is upper triangular.
def back_substitution(U, y):
  A, b = np.copy(U), np.copy(y)

  assert U.ndim == 2
  assert U.shape[0] == U.shape[1]
  n = U.shape[0]

  for i in range(n - 1, 0, -1):
    # print(f'Processing {i=}')
    # print(f'Pre A: \n{A=}')
    # print(f'Pre b: \n{b=}')
    # Coefficients of A[i, i] in the rows above it.
    c = A[:i, i]  # (i-1,)
    # print(f'c: \n{c=}')
    r = A[i]  # (n,)
    dA = r[None, :] * c[:, None]  # (i-1, n)
    # print(f'Updates to A: \n{r}')
    db = b[i] * c
    # print(f'bu = b[i] * c: \n{b[i]=}\n{c=}\n{db}')
    A[:i] -= dA
    b[:i] -= db
    # print(f'A: \n{A=}')
    # print(f'b: \n{b=}')

  return A, b
