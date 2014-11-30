import numpy as np
import scipy.linalg


class RowEchelonError(Exception):
    def __init__(self, col):
        self.col = col

    def __str__(self):
        return 'elimination failed at column '+str(self.col)


def swap_rows(a, i, j):
    """Swap rows i and j in the numpy matrix a."""
    # Note that a[i],a[j] = a[j],a[i] will _not_ work here
    temp = a[i].copy()
    a[i] = a[j]
    a[j] = temp


def partial_lu(a, ncols):
    """Compute an LU decomposition whether only the first N columns of U are upper triangular."""
    assert a.shape[0] <= a.shape[1], \
        'partial_lu not implemented for matrices with nr > nc'
    p, l, u = scipy.linalg.lu(a)
    ll = l.copy()
    ll[:, ncols:] = np.eye(*l.shape)[:, ncols:]
    uu = scipy.linalg.solve_triangular(ll, np.dot(p.T, a), lower=True)
    return p, ll, uu


def partial_row_echelon_form(a, ncols, tol=1e-8, allow_rank_defficient=True):
    """Eliminate the first N columns of A, with row pivoting."""
    a = np.asarray(a)
    assert ncols <= a.shape[1]
    if a.dtype.kind == 'i':
        a = a.astype(float)

    row = 0
    u = a.copy()
    for col in range(ncols):
        # move the row with the largest element in col i to the top
        pivot_row = row + np.argmax(np.abs(u[row:, col]))
        swap_rows(u, row, pivot_row)
        if abs(u[row, col]) < tol:
            if allow_rank_defficient:
                # this column is already eliminated, which is fine
                u[row, col] = 0.
            else:
                raise RowEchelonError(col)
        else:
            u[row, col+1:] /= u[row, col]
            u[row, col] = 1.
            u[row+1:, col+1:] -= u[row+1:, col:col+1] * u[row, col+1:]
            u[row+1:, col] = 0.
            row += 1
    return u, row
