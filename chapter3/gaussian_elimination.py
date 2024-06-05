import numpy as np


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

  print('Currently pivoting is a no-op')
  return A, b


# Converts `Ax = b` into `Ux = y` where `U` is upper triangular.
def forward_substitution(A, b):
  # Copy and modify the copies.
  A, b = np.copy(A), np.copy(b)

  # Pivot A and b if needed
  A, b = pivot(A, b, partial_pivot=False)

  # Assuming a square matrix A.
  assert A.ndim == 2, f'A is not dim 2, {A.ndim=}'
  assert A.shape[0] == A.shape[1], f'A is not square, {A.shape=}'
  n = A.shape[0]

  for i in range(n):
    # Make the diagonal value as 1.
    # NOTE: Could be numerically unstable, do pivoting as needed.
    v = A[i, i]
    # print(f'For {i=} we see diagonal {v=}')
    if abs(v) < 1e-8:
      # Swap with the largest absolute value that comes after i.
      swap_idx = np.argmax(np.abs(A[i+1:, i]))
      # print(
      #     f'For {i=} since v < 1e-8, we see: {swap_idx=} and we will add {i + 1}')
      swap_idx += (i + 1)
      tmp = np.copy(A[i])
      A[i] = A[swap_idx]
      A[swap_idx] = tmp
      # This one didn't seem to work ?!
      # A[i], A[swap_idx] = A[swap_idx], A[i]
      b[i], b[swap_idx] = b[swap_idx], b[i]
      v = A[i, i]
      # print(f'The current {v=}')
      # print(f'Current A and b are: \n {A=} \n {b=}')
    A[i] /= v
    b[i] /= v

    # for i + 1 onwards, do the subtraction to make the lower triangular
    # part hold.
    for j in range(i+1, n):
      A[j] -= A[j, i] * A[i]
      b[j] -= A[j, i] * b[i]

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
