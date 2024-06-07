import numpy as np

# TODO(afro): Do this in jax.
# TODO(afro): Do full pivoting.
# TODO(afro): Continuously pivoting.


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
def forward_substitution(A_nn, b_nb, debug=False):
  A = np.copy(A_nn)
  b = np.copy(b_nb)
  if b.ndim == 1:
    b = b[..., np.newaxis]
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
    if i < n - 1:
      # Let's update for the rows below this.
      coeffs = A[i+1:, i]  # shape: (n - i - 1,)
      subtract_A = A[i] * coeffs[..., None]  # (n,) x (n-i-1, 1) -> (n-i-1, n)
      subtract_b = b[i] * coeffs[..., None]  # (b,) x (n-i-1, 1) -> (n-i-1, b)
      if debug:
        print(f'{i}: {coeffs=}')
        print(f'{i}: {subtract_A=}')
        print(f'{i}: {b[i]=}')
        print(f'{i}: {subtract_b=}')
      # Do all the updates in the very end, since `coeffs` etc are views into
      # the array and not copies.
      A[i+1:] -= subtract_A
      b[i+1:] -= subtract_b
    if debug:
      print(f'{i}: {A=}')
      print(f'{i}: {b=}')
  return A, b


# Solves `Ux = y`  where U is upper triangular.
def back_substitution(U, y):
  A, b = np.copy(U), np.copy(y)
  if b.ndim == 1:
    b = b[..., np.newaxis]

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
    db = b[i] * c[:, None]
    # print(f'bu = b[i] * c: \n{b[i]=}\n{c=}\n{db}')
    A[:i] -= dA
    b[:i] -= db
    # print(f'A: \n{A=}')
    # print(f'b: \n{b=}')

  return A, b


def move_and_scale(from_row, to_row, n, scale=1):
  em = np.zeros((n, n))
  em[to_row, from_row] = scale
  return em


def move_and_scale_with_unit_vectors(from_row, to_row, n, scale=1):
  I = np.eye(n)
  from_unit_vec = I[from_row]  # (n,)
  to_unit_vec = I[to_row]      # (n,)
  from_unit_vec = from_unit_vec[None, ...]  # (1,n)
  to_unit_vec = to_unit_vec[..., None]      # (n,1)
  return scale * (to_unit_vec @ from_unit_vec)        # (n,n)


def elimination_matrix(from_row, to_row, n, scale=1):
  return np.eye(n) + move_and_scale(from_row, to_row, n, scale)


def permutation_matrix(permutation_vec):
  I = np.eye(permutation_vec.shape[0])
  return I[permutation_vec]


def scaling_matrix(index, scalar, n, inverse=False):
  I = np.eye(n)
  I[index, index] = scalar if not inverse else 1./scalar
  return I


# Since A = LU, and we-pre-multiply A with scaling and elimination matrices, we eventually get:
# M_{n-1}...M_1M_0A = U
# Which basically means that A = M_0^{-1}M_1^{-1}...M_{n-1}^{-1}U
# Which implies that L = M_0^{-1}M_1^{-1}...M_{n-1}^{-1}
# So, we just keep post-multiplying the inverse of the matrices we used to get U, i.e. the recurrence is:
# L = L @ M_i^{-1}
# U = M_i @ U
def lu_factorization(A_nn, debug=False):
  # This will gradually become U.
  U = np.copy(A_nn)
  n = _assert_square_and_return_dim(U)
  L = np.eye(n)
  if debug:
    print(f'Init: {L=}')
    print(f'Init: {U=}')
  # `L` will gradually become L.
  for i in range(n):
    v = U[i, i]

    # Make element at the diagonal on the i-th row as 1.
    S = scaling_matrix(i, 1./v, n)
    S_inv = scaling_matrix(i, v, n)

    U = S @ U      # U gets pre-multiplied with S.
    L = L @ S_inv  # L gets post-multiplied with S_inv.
    if debug:
      print(f'{i}: After scaling: {L=}')
      print(f'{i}: After scaling: {U=}')

    if i < n - 1:
      # Let's update for the rows below this.
      coeffs = U[i+1:, i]  # shape: (n - i - 1,)

      em = np.eye(n)
      em[i+1:, i] = -coeffs

      em_inv = np.eye(n)
      em_inv[i+1:, i] = coeffs

      U = em @ U
      L = L @ em_inv

    if debug:
      print(f'{i}: After elimination: {L=}')
      print(f'{i}: After elimination: {U=}')

  if debug:
    print(f'{i} Final: {L=}')
    print(f'{i} Final: {U=}')
  return L, U
